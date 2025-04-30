# test_finetune.py
from led.trainers.led_trainer import Trainer
from omegaconf import OmegaConf

config = OmegaConf.load("configs/train_led.yaml") 
trainer = Trainer(config)

# Test with dummy paths (use your existing dataset)
trainer.fine_tune(
    data_dirs={
        'good': config.data.train_good_image_dir,
        'bad': config.data.train_bad_image_dir,
        'degraded': config.data.train_degraded_image_dir or config.data.train_bad_image_dir
    },
    epochs=1,
    freeze_encoder=True
)

print("Test completed - check for fundus_ft.pth in output_dir")