import torch
from accelerate import Accelerator
import os
from tqdm import tqdm
from diffusers import DDPMScheduler
from accelerate.utils import ProjectConfiguration
# from led.models import build_model
import torch.nn.functional as F
# from led.data import build_dataset
# from led.models.ema import EMAModel
import inspect
from diffusers.optimization import get_scheduler
from LED.led.trainers.utils import extract_into_tensor
import wandb
import math
import numpy as np
from accelerate.logging import get_logger
from diffusers.utils import  is_wandb_available
import logging
import diffusers
import datasets
from LED.led.pipelines.led_pipeline import LEDPipeline
from typing import Optional, List

# Replace these lines:
# from led.models import build_model
# from led.data import build_dataset
# from led.models.ema import EMAModel

# With direct imports:
def import_model():
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    sys.path.insert(0, project_root)
    from LED.led.models.unet import UNet2DGenerator
    return UNet2DGenerator

UNet2DGenerator = import_model()

logger = get_logger(__name__, log_level="INFO")

class FineTunedTrainer:
    # def __init__(self, config) -> None:
    #     self.config = config
    #     self._build_accelerator()
    #     self._build_data_loaders()      
    #     self._build_model()
    #     self._build_optimizer()
    #     self._build_lr_scheduler()
    #     self._build_noise_scheduler()
    #     self._build_logger()
    def __init__(self, model,train_loader, learning_rate=1e-5, device=None):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            # Move data to device
            input_images = batch[0].to(self.device)
            target_images = batch[1].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_images)
            loss = torch.nn.functional.mse_loss(outputs, target_images)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")
    
    def _build_data_loaders(self):
        good_train_dataset, bad_train_dataset = build_dataset(good_image_dir=self.config.data.train_good_image_dir, bad_image_dir=self.config.data.train_bad_image_dir, degraded_image_dir=self.config.data.train_degraded_image_dir, config=self.config, train=True)
        self.good_train_dataloader = torch.utils.data.DataLoader(good_train_dataset, batch_size=self.config.train_batch_size, shuffle=True, num_workers=self.config.num_worker, pin_memory=True)
        self.bad_train_dataloader = torch.utils.data.DataLoader(bad_train_dataset, batch_size=self.config.train_batch_size, shuffle=True, num_workers=self.config.num_worker, pin_memory=True)

        _, bad_val_dataset = build_dataset(good_image_dir=self.config.data.val_good_image_dir, bad_image_dir=self.config.data.val_bad_image_dir, config=self.config, train=False)
        self.bad_val_dataloader = torch.utils.data.DataLoader(bad_val_dataset, batch_size=self.config.test_batch_size, shuffle=False, num_workers=self.config.num_worker, pin_memory=True) 
       

    def _build_noise_scheduler(self):
        accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
        if accepts_prediction_type:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.config.diffusion.num_train_steps,
                beta_schedule=self.config.diffusion.beta_schedule,
                prediction_type=self.config.diffusion.prediction_type,
            )
        else:
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.config.diffusion.num_train_steps, beta_schedule=self.config.diffusion.beta_schedule)
    
    
    def _build_optimizer(self):
        if self.config.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.train.lr, betas=(self.config.train.adam_beta1, self.config.train.adam_beta2), eps=self.config.train.adam_eps, weight_decay=self.config.train.weight_decay)
        elif self.config.optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr, betas=(self.config.train.adam_beta1, self.config.train.adam_beta2), eps=self.config.train.adam_eps, weight_decay=self.config.train.weight_decay)

        elif self.config.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr)


    def _build_lr_scheduler(self):
        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.train.lr_warmup_steps * self.config.train.gradient_accumulation_steps,
            num_training_steps=(len(self.good_train_dataloader) * self.config.train.num_epochs),
        )


    def _build_model(self):
        self.model, self.model_class = build_model(self.config)
        if self.config.use_ema:
            self.ema_model = EMAModel(self.model.parameters(),
                                decay=self.config.train.ema_max_decay,
                                use_ema_warmup=True,
                                inv_gamma=self.config.train.ema_inv_gamma,
                                power=self.config.train.ema_power,
                                model_cls=self.model_class,
                                model_config=self.model.config)

    def resume_from_checkpoint(self):
        if self.config.resume is None:
            return
        if self.config.resume == 'last': # get the latest checkpoint
            dirs = os.listdir(self.config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            checkpoint_path = dirs[-1] if len(dirs) > 0 else None
        else:
            checkpoint_path = os.path.basename(os.path.normpath(self.config.resume))
        if checkpoint_path is None:
            self.accelerator.print(
                f"Checkpoint '{checkpoint_path}' does not exist. Starting a new training run."
            )
        else:
            self.accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            self.accelerator.load_state(os.path.join(self.config.output_dir, checkpoint_path))
            self.global_step = int(checkpoint_path.split("-")[1])
            self.resume_global_step = self.global_step * self.config.train.gradient_accumulation_steps
            self.first_epoch = self.global_step // self.num_update_steps_per_epoch
            self.resume_step = self.resume_global_step % (self.num_update_steps_per_epoch * self.config.train.gradient_accumulation_steps)

            
    def _build_accelerator(self):
        accelerator_project_config = ProjectConfiguration(total_limit=self.config.train.checkpointing_steps_total_limit)
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps, 
            log_with="tensorboard",
            logging_dir=os.path.join(self.config.output_dir, "logs"), 
            project_config=accelerator_project_config)
        
        def save_model_hook(models, weights, output_dir):
            if self.config.use_ema:
                self.ema_model.save_pretrained(os.path.join(output_dir, "model_ema"))
            for i, model in enumerate(models):
                #print(model.name)
                # get the class name of the model
                model_name = model.__class__.__name__
                model.save_pretrained(os.path.join(output_dir, model_name))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if self.config.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "model_ema"), self.model_class)
                self.ema_model.load_state_dict(load_model.state_dict())
                self.ema_model.to(self.accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                # load diffusers style into model
                model_name = model.__class__.__name__
                load_model = model.from_pretrained(input_dir, subfolder=model_name)
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model
        
        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def _build_logger(self):
        if self.config.logger_name == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError("You want to use `tensorboard` logger which is not installed yet, run `pip install tensorboard`.")

        elif self.config.logger_name == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb
        
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()


    def get_new_dataset_loader():
        from torchvision import transforms
        from torchvision.datasets import FakeData
        import torch

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),  # match your model input
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = FakeData(
            size=20,
            image_size=(3, 128, 128),
            num_classes=1,
            transform=transform
        )

        # Simulate degraded images by adding noise
        class PairedFakeData(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __getitem__(self, idx):
                img, _ = self.dataset[idx]
                noise = torch.randn_like(img) * 0.2
                degraded = torch.clamp(img + noise, -1, 1)
                return {
                    'image': img,  # clean
                    'degraded_image': degraded  # noisy
                }

            def __len__(self):
                return len(self.dataset)

        paired_dataset = PairedFakeData(dataset)
        return torch.utils.data.DataLoader(paired_dataset, batch_size=4, shuffle=True)


    def train(self):
        model, optimizer, good_train_dataloader, bad_train_dataloader, bad_val_dataloader, lr_scheduler= self.accelerator.prepare(
            self.model, self.optimizer, self.good_train_dataloader, self.bad_train_dataloader, self.bad_val_dataloader, self.lr_scheduler)
        if self.config.use_ema:
            self.ema_model.to(self.accelerator.device)

        if self.accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            run = os.path.split(__file__)[-1].split(".")[0]
            self.accelerator.init_trackers(run)

        self.global_step = 0
        self.first_epoch = 0
        self.total_batch_size = self.config.train_batch_size * self.accelerator.num_processes * self.config.train.gradient_accumulation_steps
        self.num_update_steps_per_epoch = math.ceil(len(good_train_dataloader) / self.config.train.gradient_accumulation_steps)
        self.max_train_steps = self.config.train.num_epochs *self.num_update_steps_per_epoch

        self.resume_from_checkpoint()

        bad_val_loader_iter = iter(bad_val_dataloader)
        
        # Now you train the model
        model.train()
        for epoch in range(self.first_epoch, self.config.train.num_epochs):
            progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(good_train_dataloader):
                # Skip steps until we reach the resumed step
                if self.config.resume and epoch == self.first_epoch and step < self.resume_step:
                    if step % self.config.train.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue
                # train the generator
                good_images = batch['image']
                degraded_images = batch['degraded_image']
                # Sample noise to add to the images
                noise = torch.randn(good_images.shape).to(good_images.device)
                bs = good_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=good_images.device).long()
                noisy_images = self.noise_scheduler.add_noise(good_images, noise, timesteps)
                with self.accelerator.accumulate(model):
                    # Predict the noise residual
                    noisy_inputs = torch.cat([noisy_images, degraded_images], dim=1)
                    model_output = model(noisy_inputs, timesteps, return_dict=False)[0]
                    if self.config.diffusion.prediction_type == "epsilon":
                        loss = F.mse_loss(model_output, noise)  # this could have different weights!
                    elif self.config.diffusion.prediction_type == "sample":
                        alpha_t = extract_into_tensor(
                            self.noise_scheduler.alphas_cumprod, timesteps, (good_images.shape[0], 1, 1, 1)
                        )
                        snr_weights = alpha_t / (1 - alpha_t)
                        loss = snr_weights * F.mse_loss(
                            model_output, good_images, reduction="none"
                        )  # use SNR weighting from distillation paper
                        loss = loss.mean()
                    else:
                        raise ValueError(f"Unsupported prediction type: {self.config.diffusion.prediction_type}")
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()               
                if self.accelerator.sync_gradients:
                    if self.config.use_ema:
                        self.ema_model.step(model.parameters())
                    progress_bar.update(1)
                    self.global_step += 1
                    if self.global_step % self.config.train.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                        
                        torch.save({
                            'model_state': self.model.state_dict(),
                            'optimizer_state': self.optimizer.state_dict(),
                            'epoch': epoch,
                            'global_step': self.global_step,
                        }, os.path.join(save_path, "model_ft_ready.pth"))


                logs = {"denoise_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": self.global_step} #"ref_denoise_loss": ref_loss.detach().item(),
                if self.config.use_ema:
                    logs["ema_decay"] = self.ema_model.get_decay(self.global_step)
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)
            progress_bar.close()
            self.accelerator.wait_for_everyone()
            # Generate sample images for visual inspection
            torch.set_grad_enabled(False)
            unwrap_model = self.accelerator.unwrap_model(model).eval()
            if self.accelerator.is_main_process:
                if epoch == self.first_epoch or epoch % self.config.train.save_images_epochs == 0 or epoch == self.config.train.num_epochs - 1:
                    if self.config.use_ema:
                        self.ema_model.copy_to(unwrap_model.parameters())
                    pipeline = LEDPipeline(
                        unet=unwrap_model,
                        scheduler=self.noise_scheduler,
                        image_size=self.config.data.image_size,
                    )
                    generator = torch.Generator(device=pipeline.device).manual_seed(0)
                    # run pipeline in inference (sample random noise and denoise)
                    try:
                        bad_eval_batch = next(bad_val_loader_iter)
                    except StopIteration:
                        bad_val_loader_iter = iter(bad_val_dataloader)
                        bad_eval_batch = next(bad_val_loader_iter)
                    images_processed = pipeline(
                        cond_image=bad_eval_batch['image'].to(pipeline.device),
                        generator=generator,
                        num_inference_steps=self.config.diffusion.num_inference_steps,
                        output_type="numpy",
                        output_max_val=255.0,
                    )
                    # denormalize the images and save to tensorboard
                    real_bad = ((bad_eval_batch['image'].cpu().numpy() * 0.5 + 0.5) * 255.0).astype(np.uint8)
                    images = np.concatenate([real_bad, images_processed.transpose(0, 3, 1, 2)], axis=0)
                    #print(images_processed.shape, real_bad.shape)
                    if self.config.logger_name == "tensorboard":
                        self.accelerator.get_tracker("tensorboard").add_images(
                            "samples", images, epoch
                        )
                    elif self.config.logger_name == "wandb":
                        self.accelerator.get_tracker("wandb").log(
                            {"test_samples": [wandb.Image(img) for img in images.transpose(0, 3, 1, 2)], "epoch": epoch},
                            step=self.global_step,
                        )   
                if epoch == self.first_epoch or epoch % self.config.train.save_model_epochs == 0 or epoch == self.config.train.num_epochs - 1:
                    # save the model
                    pipeline.save_pretrained(self.config.output_dir)
            torch.set_grad_enabled(True)
            self.accelerator.unwrap_model(unwrap_model).train()
            # clear the cache
            torch.cuda.empty_cache()

        if self.config.get("fine_tune", {}).get("enabled", False):
            new_dataloader = self.get_new_dataset_loader()
            self.fine_tune(
                new_dataloader=new_dataloader,
                checkpoint_path=self.config.fine_tune.checkpoint,
                ft_epochs=self.config.fine_tune.epochs,
            )

        
        self.accelerator.end_training()



    

    def fine_tune(self, data_dirs, epochs=5, freeze_encoder=True):
        """Minimal fine-tuning for fundus images
        Args:
            data_dirs: dict with keys 'good', 'bad', 'degraded' (optional)
            epochs: number of fine-tuning epochs
            freeze_encoder: whether to freeze down_blocks
        """
        # 1. Load latest checkpoint if no specific path given
        checkpoint_dirs = [d for d in os.listdir(self.config.output_dir) 
                        if d.startswith("checkpoint")]
        latest_checkpoint = os.path.join(self.config.output_dir, sorted(checkpoint_dirs)[-1])
        checkpoint = torch.load(os.path.join(latest_checkpoint, "pytorch_model.bin"))
        
        # 2. Prepare model
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(checkpoint)
        
        if freeze_encoder:
            for name, param in model.named_parameters():
                if 'down_blocks' in name:
                    param.requires_grad = False
        
        # 3. Create fundus dataloader (minimal changes)
        good_dataset, _ = build_dataset(
            good_image_dir=data_dirs['good'],
            bad_image_dir=data_dirs['bad'],
            degraded_image_dir=data_dirs.get('degraded'),
            config=self.config,
            train=True
        )
        ft_loader = torch.utils.data.DataLoader(
            good_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True
        )
        
        # 4. Fine-tune loop (barebones)
        model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-5  # Fixed small LR
        )
        
        for epoch in range(epochs):
            for batch in ft_loader:
                with self.accelerator.accumulate(model):
                    # Original loss computation
                    good = batch['image']
                    degraded = batch.get('degraded_image', torch.zeros_like(good))
                    
                    noise = torch.randn_like(good)
                    timesteps = torch.randint(0, 1000, (good.shape[0],), device=good.device)
                    noisy = self.noise_scheduler.add_noise(good, noise, timesteps)
                    
                    pred = model(torch.cat([noisy, degraded], dim=1), timesteps).sample
                    loss = F.mse_loss(pred, noise)
                    
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                    
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
        
        # 5. Save only what's needed
        torch.save(model.state_dict(), os.path.join(self.config.output_dir, "fundus_ft.pth"))
                






                        
