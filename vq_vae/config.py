"""
Configuration module for VQ-VAE with Transformer architecture.
All hyperparameters are defined here for easy experimentation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VQVAEConfig:
    """Configuration for VQ-VAE with Transformer backbone."""
    
    # Input and output dimensions
    input_dim: int = 512
    output_dim: int = 512
    
    # Quantization parameters
    num_embeddings: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5

    # Classification token parameters
    encoder_class_token: bool = True
    encoder_class_proj_dim: Optional[int] = 1  # Set to None to disable
    
    # Encoder parameters
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 2
    encoder_num_heads: int = 8
    encoder_ff_dim: int = 512
    encoder_dropout: float = 0.1
    encoder_activation: str = "relu"
    
    # Decoder parameters
    decoder_hidden_dim: int = 256
    decoder_num_layers: int = 2
    decoder_num_heads: int = 8
    decoder_ff_dim: int = 512
    decoder_dropout: float = 0.1
    decoder_activation: str = "relu"
    
    # Projection layers
    use_encoder_projection: bool = True
    use_decoder_projection: bool = True
    projection_dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Loss weights
    reconstruction_loss_weight: float = 1.0
    commitment_loss_weight: float = 0.25
    classification_loss_weight: float = 0.5  # Weight for classification loss
    
    # Device
    device: str = "cuda"
    
    # Seed for reproducibility
    seed: int = 42
    
    # Sequence parameters
    max_seq_length: int = 512
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
        }
    
    def __repr__(self):
        """Pretty print configuration."""
        lines = ["\nVQ-VAE Configuration:"]
        for key, value in sorted(self.__dict__.items()):
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class TransformerConfig:
    """Configuration for Transformer components."""
    
    hidden_dim: int = 256
    num_heads: int = 8
    ff_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
    layer_norm_eps: float = 1e-6
    bias: bool = True
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
        }
