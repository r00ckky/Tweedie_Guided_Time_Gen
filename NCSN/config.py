from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class NCSNConfig:
    vq_vae_checkpoint: str
    checkpoint_dir: str = "./checkpoints"
    num_epochs: int = 100                
    denoiser_hidden_dim: int = 256
    sigma_max: float = 1.0
    sigma_min: float = 0.01
    num_scales: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    ncsn_num_blocks: int = 3
    denoiser_model:str = "dit"  # Options: "dit", "conv_next", "conv", "resnet"