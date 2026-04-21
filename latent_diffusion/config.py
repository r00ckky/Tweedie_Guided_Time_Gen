from dataclasses import dataclass
from typing import Optional


@dataclass
class DiTConfig:
    """
    All hyperparameters for the Latent Diffusion Transformer.

    Typical AMEX defaults (for latent_dim=64, seq_len=13):
        hidden_dim=256, cond_dim=256, num_layers=6, num_heads=8
    """

    # ── DiT architecture ──────────────────────────────────────────────────────
    hidden_dim: int   = 256   # Internal transformer width H
    cond_dim: int     = 256   # Timestep/label conditioning width C
    num_layers: int   = 6     # Number of DiTBlock stacks
    num_heads: int    = 8     # Self-attention heads
    ff_multiplier: int = 4    # FFN hidden = hidden_dim * ff_multiplier
    dropout: float    = 0.1   # Dropout rate

    # ── Label conditioning (classifier-free guidance) ─────────────────────────
    num_classes: Optional[int] = 2    # 2 for binary AMEX; None = unconditional
    cfg_dropout_prob: float = 0.10    # Probability to drop label (CFG training)

    # ── Diffusion schedule ────────────────────────────────────────────────────
    timesteps: int  = 1_000           # Total diffusion steps T
    schedule: str   = "cosine"        # "cosine" or "linear"
    beta_start: float = 1e-4          # For linear schedule only
    beta_end: float   = 0.02          # For linear schedule only

    # ── Training ──────────────────────────────────────────────────────────────
    num_epochs: int       = 100
    batch_size: int       = 256
    learning_rate: float  = 1e-4
    weight_decay: float   = 1e-4
    max_grad_norm: float  = 1.0       # Gradient clip threshold
    warmup_epochs: int    = 5         # Linear LR warmup

    # ── Encoder settings ──────────────────────────────────────────────────────
    freeze_encoder: bool         = True   # Keep encoder frozen during DiT training
    vqvae_checkpoint: Optional[str] = None  # Path to pretrained VQ-VAE .pt file

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int            = 42
    device: str          = "cuda"
    output_dir: str      = "./dit_output"
    log_freq: int        = 50          # Log every N gradient steps
    checkpoint_freq: int = 10          # Save checkpoint every N epochs
    save_best: bool      = True
    use_wandb: bool      = False
    wandb_project: str   = "amex-latent-diffusion"
    wandb_run_name: Optional[str] = None
    debug: bool          = False
    debug_size: int      = 512