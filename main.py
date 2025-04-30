import argparse
import os
import sys
from omegaconf import OmegaConf
from LED.led.trainers.led_trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='LED Fine-Tuning')
    parser.add_argument('--config', type=str, required=True, help='Base config file')
    parser.add_argument('--ft_config', type=str, required=True, help='Fine-tuning config file')
    parser.add_argument('--resume', type=str, default='', help='Checkpoint to resume from')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load base config
    config = OmegaConf.load(args.config)
    
    # Load fine-tuning config and merge
    ft_config = OmegaConf.load(args.ft_config)
    config = OmegaConf.merge(config, ft_config)
    
    if args.resume:
        config.resume = args.resume
    else:
        config.resume = None
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Run fine-tuning
    trainer.fine_tune(config.fine_tune)

if __name__ == '__main__':
    main()