import torch
from omegaconf import OmegaConf
import sys
import os
import importlib.util

# --------------------------------------------------
# SETUP PATHS - NO HARDCODED ASSUMPTIONS
# --------------------------------------------------

# Get absolute path to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate project root (two levels up from script)
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Define all required paths using project root
led_module_path = os.path.join(project_root, "LED", "led")
config_path = os.path.join(led_module_path, "models", "default_config.yaml")
weights_path = os.path.join(project_root, "LED", "pretrained_weights", "led.bin")
output_dir = os.path.join(project_root, "LED", "outputs", "dummy-checkpoint")
output_path = os.path.join(output_dir, "model_ft_ready.pth")

# --------------------------------------------------
# IMPORT UNET WITHOUT CIRCULAR IMPORTS
# --------------------------------------------------

# Temporarily add LED directory to Python path
sys.path.insert(0, os.path.join(project_root, "LED"))

# Monkey-patch OmegaConf.load to handle the config path
original_omega_load = OmegaConf.load
def patched_omega_load(path):
    if path == "led/models/default_config.yaml":
        return original_omega_load(config_path)
    return original_omega_load(path)
OmegaConf.load = patched_omega_load

try:
    # Import UNet using the original import path
    from led.models.unet import UNet2DGenerator
    
    # --------------------------------------------------
    # MODEL SETUP
    # --------------------------------------------------
    
    # Load config (will use our patched loader)
    config = OmegaConf.load("led/models/default_config.yaml")
    
    # Create model instance
    model = UNet2DGenerator(
        sample_size=config.sample_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        center_input_sample=config.center_input_sample,
        time_embedding_type=config.time_embedding_type,
        freq_shift=config.freq_shift,
        flip_sin_to_cos=config.flip_sin_to_cos,
        down_block_types=tuple(config.down_block_types),
        up_block_types=tuple(config.up_block_types),
        block_out_channels=tuple(config.block_out_channels),
        layers_per_block=config.layers_per_block,
        mid_block_scale_factor=config.mid_block_scale_factor,
        downsample_padding=config.downsample_padding,
        act_fn=config.act_fn,
        attention_head_dim=config.attention_head_dim,
        norm_num_groups=config.norm_num_groups,
        norm_eps=config.norm_eps,
        resnet_time_scale_shift=config.resnet_time_scale_shift,
        add_attention=config.add_attention,
    )
    
    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    
    # --------------------------------------------------
    # CREATE CHECKPOINT
    # --------------------------------------------------
    
    optimizer = torch.optim.AdamW(model.parameters())
    
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': 0,
        'global_step': 0,
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, output_path)
    print(f"Successfully created dummy checkpoint at:\n{output_path}")

finally:
    # Restore original OmegaConf.load
    OmegaConf.load = original_omega_load
    # Remove temporary path
    sys.path.pop(0)