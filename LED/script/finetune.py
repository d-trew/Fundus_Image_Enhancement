import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import time
import datetime
from tqdm import tqdm

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
led_path = os.path.join(project_root, "LED")
sys.path.extend([project_root, led_path])

# Patch OmegaConf.load to handle correct paths
original_load = OmegaConf.load
def patched_load(file_):
    if "default_config.yaml" in file_:
        actual_path = os.path.join(led_path, "led", "models", "default_config.yaml")
        return original_load(actual_path)
    return original_load(file_)
OmegaConf.load = patched_load


# Import model components
from led.models.unet import UNet2DGenerator
from led.data.finetune_dataset import FineTuneDataset


def load_pretrained_weights(model, pretrained_path, model_type="led"):
    """Load weights with handling for different model types"""
    checkpoint = torch.load(pretrained_path)
    
    # Handle different model types
    if model_type == "led":
        state_dict = checkpoint
    elif model_type == "scrnet":
        state_dict = checkpoint['model_state_dict']
    elif model_type == "pcenet":
        state_dict = checkpoint['state_dict']
    elif model_type == "isecret":
        state_dict = checkpoint['model']
    elif model_type == "arcnet":
        state_dict = checkpoint
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load with strict=False to ignore non-matching keys
    model.load_state_dict(state_dict, strict=False)
    
    print(f"Loaded {model_type} weights from {pretrained_path}")
    return model

def load_config():
    """Load and validate all configurations with comprehensive error handling"""
    # 1. Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    led_path = os.path.join(project_root, "LED")
    config_dir = os.path.join(led_path, "configs")
    
    # 2. Default configuration (fallback)
    default_config = {
        "model": {
            "sample_size": 512,
            "in_channels": 6,
            "out_channels": 3,
            "block_out_channels": [128, 128, 256, 256, 512, 512],
            "layers_per_block": 2,
            "attention_head_dim": 8,
            "norm_num_groups": 32,
            "down_block_types": ["DownBlock2D"]*4 + ["AttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "AttnUpBlock2D"] + ["UpBlock2D"]*3,
            "mid_block_scale_factor": 1,
            "act_fn": "silu"
        },
        "train": {
            "lr": 1e-5,
            "num_epochs": 50,
            "train_batch_size": 4,
            "eval_batch_size": 4,
            "num_worker": 4,
            "weight_decay": 1e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "save_model_epochs": 10
        },
        "data": {
            "image_size": 512,
            "train_good_image_dir": os.path.join(project_root, "finetune_images", "good_images"),
            "train_bad_image_dir": os.path.join(project_root, "finetune_images", "degraded_images")
        },
        "output_dir": os.path.join(project_root, "logs", "finetune")
    }

    # 3. Load YAML configs
    def load_yaml(path):
        try:
            with open(path) as f:
                config = yaml.safe_load(f)
                print(f"Loaded config from {path}")
                return config if config else {}
        except Exception as e:
            print(f"Warning: Failed to load {path}: {str(e)}")
            return {}

    train_config = load_yaml(os.path.join(config_dir, "train_led.yaml"))
    finetune_config = load_yaml(os.path.join(config_dir, "finetune_led.yaml"))

    # 4. Deep merge configurations
    def deep_merge(base, update):
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = deep_merge(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    config = deep_merge(default_config.copy(), train_config)
    config = deep_merge(config, finetune_config)

    # 5. Process special values and convert types
    def process_config(cfg):
        """Recursively convert string numbers to floats/ints and handle booleans"""
        if isinstance(cfg, dict):
            return {k: process_config(v) for k, v in cfg.items()}
        elif isinstance(cfg, list):
            return [process_config(v) for v in cfg]
        elif isinstance(cfg, str):
            # Handle boolean strings
            if cfg.lower() == 'true':
                return True
            if cfg.lower() == 'false':
                return False
            # Handle scientific notation (e.g., 1e-5)
            if 'e' in cfg.lower():
                try:
                    return float(cfg)
                except ValueError:
                    return cfg
            # Handle regular numbers
            try:
                return float(cfg) if '.' in cfg else int(cfg)
            except ValueError:
                return cfg
        return cfg

    config = process_config(config)
    
    # Specifically ensure critical training parameters are numeric
    for param in ['lr', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_eps']:
        if param in config['train'] and isinstance(config['train'][param], str):
            try:
                config['train'][param] = float(config['train'][param])
            except ValueError:
                raise ValueError(f"Could not convert {param} to float: {config['train'][param]}")

    # 6. Validate paths and required fields
    def validate():
        # Check required model parameters
        required_model_params = ["sample_size", "in_channels", "out_channels", 
                               "block_out_channels", "layers_per_block"]
        for param in required_model_params:
            if param not in config["model"]:
                raise ValueError(f"Missing required model parameter: {param}")

        # Verify data directories exist
        for path_key in ["train_good_image_dir", "train_bad_image_dir"]:
            path = config["data"][path_key]
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Data directory not found: {path}\n"
                    f"Current working directory: {os.getcwd()}\n"
                    f"Project root: {project_root}"
                )

        # Convert relative paths to absolute
        for path_key in ["train_good_image_dir", "train_bad_image_dir", "output_dir"]:
            if path_key in config["data"]:
                config["data"][path_key] = os.path.abspath(
                    os.path.join(project_root, config["data"][path_key])
                )

    validate()

    # 7. Separate model architecture from training params
    final_config = {
        "model_config": {  # For model initialization
            k: v for k, v in config["model"].items()
            if k not in ["freeze_backbone", "unfreeze_layers", "pretrained_path", "model_type"]
        },
        "training_config": {  # For training parameters
            **config["train"],
            "freeze_backbone": config["model"].get("freeze_backbone", True),
            "unfreeze_layers": config["model"].get("unfreeze_layers", []),
        },
        "data_config": config["data"],
        "paths": {
            "pretrained": os.path.abspath(
                os.path.join(led_path, config["model"].get("pretrained_path", ""))
            ) if config["model"].get("pretrained_path") else None,
            "output_dir": config["output_dir"]
        }
    }

    print("\n=== Final Config Structure ===")
    print(f"1. Model Config: {len(final_config['model_config'])} parameters")
    print(f"2. Training Config: {final_config['training_config'].keys()}")
    print(f"3. Data Config: {final_config['data_config'].keys()}")
    print(f"4. Paths: {final_config['paths']}")

    return final_config

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_TIMESTEPS = 1000  # Diffusion timesteps

    # 1. Initialize model
    model = UNet2DGenerator(**config["model_config"]).to(device)
    
    # 2. Load pretrained weights
    if config["paths"]["pretrained"]:
        model = load_pretrained_weights(model, config["paths"]["pretrained"], "led")
    
    # 3. Freeze layers if specified
    if config["training_config"]["freeze_backbone"]:
        for name, param in model.named_parameters():
            param.requires_grad = False
            for layer in config["training_config"]["unfreeze_layers"]:
                if layer in name:
                    param.requires_grad = True
    
    # 4. Initialize optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training_config"]["lr"],
        weight_decay=config["training_config"]["weight_decay"]
    )
    
    # 5. Prepare data loaders
    train_dataset = FineTuneDataset(
        good_dir=config["data_config"]["train_good_image_dir"],
        bad_dir=config["data_config"]["train_bad_image_dir"],
        image_size=config["data_config"]["image_size"]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training_config"]["train_batch_size"],
        shuffle=True,
        num_workers=0,  # Set to 0 to disable multiprocessing
        pin_memory=False  # Disable if using CPU
    )
    
    model.train()
    print("\n=== Starting Training ===")
    print(f"Total epochs: {config['training_config']['num_epochs']}")
    print(f"Batch size: {config['training_config']['train_batch_size']}")
    print(f"Checkpoints every {config['training_config']['save_model_epochs']} epochs")
    print(f"Device: {device}\n")

    start_time = time.time()

    for epoch in range(config["training_config"]["num_epochs"]):
        epoch_start_time = time.time()
        print(f"\n=== Epoch {epoch+1}/{config['training_config']['num_epochs']} ===")
        
        # Initialize tqdm progress bar
        batch_iterator = tqdm(train_loader, 
                            desc=f"Epoch {epoch+1}", 
                            unit="batch",
                            leave=True)
        
        batch_times = []
        total_loss = 0
        
        for batch_idx, batch in enumerate(batch_iterator):
            batch_start_time = time.time()
            
            # --- Data Loading ---
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            
            # --- Forward Pass ---
            t = torch.randint(0, NUM_TIMESTEPS, (x.size(0),), device=device).long()
            model_output = model(x, t)
            
            # Handle different output types
            if hasattr(model_output, 'sample'):
                pred = model_output.sample
            elif hasattr(model_output, 'pred'):
                pred = model_output.pred
            else:
                pred = model_output
            
            # --- Loss Calculation ---
            loss = torch.nn.functional.mse_loss(pred, y)
            total_loss += loss.item()
            
            # --- Backward Pass ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            batch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{(total_loss/(batch_idx+1)):.4f}"
            })
            
            # --- Timing ---
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
        
        # --- Epoch Completion ---
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed in {datetime.timedelta(seconds=int(epoch_time))}")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average batch time: {sum(batch_times)/len(batch_times):.2f}s")
        
        # --- Checkpoint Saving ---
        if (epoch + 1) % config["training_config"]["save_model_epochs"] == 0:
            save_start = time.time()
            os.makedirs(config["paths"]["output_dir"], exist_ok=True)
            save_path = os.path.join(config["paths"]["output_dir"], f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint saved to {save_path} in {time.time()-save_start:.2f}s")

    # --- Training Completion ---
    total_time = time.time() - start_time
    print("\n=== Training Complete ===")
    print(f"Total training time: {datetime.timedelta(seconds=int(total_time))}")
    print(f"Final average loss: {avg_loss:.4f}")
    print(f"Model saved to: {config['paths']['output_dir']}")

if __name__ == "__main__":
    main()