"""
Configuration module for VAE with Transformer architecture.
All hyperparameters are defined here for easy experimentation.
Streamlined with unified parameters for encoder/decoder and patch embedding for time-series.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VAEConfig:
    """Streamlined configuration for VAE with Transformer backbone and Patch Embedding."""
    
    # ============================================================
    # Input and Patch Embedding Parameters
    # ============================================================
    input_dim: int = 294  # Feature dimension of raw input
    patch_size: int = 2  # Patch kernel size for Conv1d embedding
    patch_stride: int = 2  # Patch stride for Conv1d embedding
    patch_embed_dim: int = 256  # Output dimension of patch embedding
    
    # ============================================================
    # Latent Space Parameters
    # ============================================================
    latent_dim: int = 256  # Dimension of latent space (mu and log_var)
    
    # ============================================================
    # Transformer Parameters (Unified for Encoder and Decoder)
    # ============================================================
    hidden_dim: int = 256  # Transformer hidden dimension (encoder and decoder)
    num_layers: int = 6  # Number of transformer layers (encoder and decoder)
    num_heads: int = 8  # Number of attention heads
    ff_multiplier: int = 2  # Feed-forward dimension = hidden_dim * ff_multiplier
    dropout: float = 0.1
    activation: str = "relu"
    
    # ============================================================
    # Classification Token Parameters
    # ============================================================
    use_class_token: bool = True
    class_proj_dim: Optional[int] = 1  # Set to None to disable classification head
    class_func: str = 'knn'
    
    # ============================================================
    # Loss Weights
    # ============================================================
    reconstruction_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0
    classification_loss_weight: float = 0.5
    koleo_penalty_weight: float = 0.1
    
    # ============================================================
    # Training and Device Parameters
    # ============================================================
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    max_seq_length: int = 512
    
    # ============================================================
    # Derived Properties
    # ============================================================
    @property
    def ff_dim(self) -> int:
        """Feed-forward dimension derived from hidden_dim."""
        return self.hidden_dim * self.ff_multiplier
    
    @property
    def encoder_num_layers(self) -> int:
        """Alias for backwards compatibility."""
        return self.num_layers
    
    @property
    def decoder_num_layers(self) -> int:
        """Alias for backwards compatibility."""
        return self.num_layers
    
    @property
    def encoder_num_heads(self) -> int:
        """Alias for backwards compatibility."""
        return self.num_heads
    
    @property
    def decoder_num_heads(self) -> int:
        """Alias for backwards compatibility."""
        return self.num_heads
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def __repr__(self):
        """Pretty print configuration."""
        lines = ["\n" + "="*60]
        lines.append("VAE Configuration (Streamlined)")
        lines.append("="*60)
        
        sections = {
            "Patch Embedding": ["input_dim", "patch_size", "patch_stride", "patch_embed_dim"],
            "Latent Space": ["latent_dim"],
            "Transformer (Unified)": ["hidden_dim", "num_layers", "num_heads", "ff_multiplier", "dropout"],
            "Classification": ["use_class_token", "class_proj_dim"],
            "Loss Weights": ["reconstruction_loss_weight", "kl_loss_weight", "classification_loss_weight"],
            "Training": ["learning_rate", "weight_decay", "device", "seed"],
        }
        
        for section, keys in sections.items():
            lines.append(f"\n{section}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "="*60)
        return "\n".join(lines)
