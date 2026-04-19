from __future__ import annotations

import gc
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ── Local imports (adjust paths to match your project structure) ───────────────
# The three new modules live alongside the existing vq_vae package.
# Adjust these imports if you've placed them inside the vq_vae/ directory.
from dit_model import DiT
from latent_diffusion import GaussianDiffusion, LatentDiffusionModel

# Reuse existing components from VQ-VAE codebase
from vq_vae.encoder import TransformerEncoder
from vq_vae.vq_vae import PatchEmbedding
from vq_vae.config import VQVAEConfig

# Your existing dataset class — adjust as needed
from data.data import AmexDataset


# ─── Config ───────────────────────────────────────────────────────────────────


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


# ─── Model Factory ────────────────────────────────────────────────────────────


def build_latent_diffusion_model(
    vqvae_cfg: VQVAEConfig,
    dit_cfg: DiTConfig,
    device: torch.device,
    logger: logging.Logger,
) -> LatentDiffusionModel:
    """
    Construct the full LatentDiffusionModel.

    1. Builds PatchEmbedding + TransformerEncoder from vqvae_cfg
    2. Loads pretrained encoder weights from vqvae_checkpoint (if provided)
    3. Builds DiT noise predictor from dit_cfg
    4. Wraps everything in LatentDiffusionModel

    Args:
        vqvae_cfg: VQ-VAE config (defines encoder architecture/dims)
        dit_cfg:   DiT training config
        device:    Target torch device
        logger:    Python logger

    Returns:
        model: LatentDiffusionModel on `device`
    """

    # ── 1. Build encoder backbone (mirrors VQ_VAE.__init__) ──────────────────
    patch_embedding = PatchEmbedding(
        input_dim=vqvae_cfg.input_dim,
        patch_size=vqvae_cfg.patch_size,
        patch_stride=vqvae_cfg.patch_stride,
        patch_embed_dim=vqvae_cfg.patch_embed_dim,
    )
    logger.info(
        f"PatchEmbedding: input_dim={vqvae_cfg.input_dim}, "
        f"patch_size={vqvae_cfg.patch_size}, "
        f"patch_embed_dim={vqvae_cfg.patch_embed_dim}"
    )

    encoder = TransformerEncoder(
        input_dim=vqvae_cfg.patch_embed_dim,    # encoder sees patch-embedded features
        hidden_dim=vqvae_cfg.hidden_dim,
        embedding_dim=vqvae_cfg.embedding_dim,  # = latent_dim for DiT
        num_layers=vqvae_cfg.num_layers,
        num_heads=vqvae_cfg.num_heads,
        ff_dim=vqvae_cfg.ff_dim,
        dropout=vqvae_cfg.dropout,
        activation="relu",
        class_token=vqvae_cfg.use_class_token,
        class_proj_dim=vqvae_cfg.class_proj_dim,
        class_func=vqvae_cfg.class_func,
        koleo_penalty_weight=vqvae_cfg.koleo_penalty_weight,
    )
    logger.info(
        f"Encoder: hidden_dim={vqvae_cfg.hidden_dim}, "
        f"embedding_dim={vqvae_cfg.embedding_dim}, "
        f"num_layers={vqvae_cfg.num_layers}"
    )

    # ── 2. Load pretrained encoder weights ───────────────────────────────────
    if dit_cfg.vqvae_checkpoint is not None:
        ckpt_path = Path(dit_cfg.vqvae_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {ckpt_path}")

        logger.info(f"Loading pretrained VQ-VAE weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)

        # Extract only the relevant submodule weights
        pe_state  = {k.replace("patch_embedding.", ""): v
                     for k, v in state.items() if k.startswith("patch_embedding.")}
        enc_state = {k.replace("encoder.", ""): v
                     for k, v in state.items() if k.startswith("encoder.")}

        patch_embedding.load_state_dict(pe_state, strict=True)
        encoder.load_state_dict(enc_state, strict=True)
        logger.info("✓ Pretrained encoder weights loaded successfully.")
    else:
        logger.warning(
            "No VQ-VAE checkpoint provided — encoder initialized from scratch. "
            "For best results, train VQ-VAE first, then pass its checkpoint here."
        )

    # ── 3. Build DiT noise predictor ──────────────────────────────────────────
    dit = DiT(
        latent_dim=vqvae_cfg.embedding_dim,   # Must match encoder output dim D
        hidden_dim=dit_cfg.hidden_dim,
        cond_dim=dit_cfg.cond_dim,
        num_layers=dit_cfg.num_layers,
        num_heads=dit_cfg.num_heads,
        ff_multiplier=dit_cfg.ff_multiplier,
        dropout=dit_cfg.dropout,
        num_classes=dit_cfg.num_classes,
        cfg_dropout_prob=dit_cfg.cfg_dropout_prob,
    )
    dit_params = dit.get_param_count()
    logger.info(
        f"DiT: hidden_dim={dit_cfg.hidden_dim}, layers={dit_cfg.num_layers}, "
        f"heads={dit_cfg.num_heads}, total_params={dit_params['total']:,}"
    )

    # ── 4. Build diffusion process ────────────────────────────────────────────
    diffusion = GaussianDiffusion(
        timesteps=dit_cfg.timesteps,
        schedule=dit_cfg.schedule,
        beta_start=dit_cfg.beta_start,
        beta_end=dit_cfg.beta_end,
    )
    logger.info(
        f"GaussianDiffusion: timesteps={dit_cfg.timesteps}, "
        f"schedule={dit_cfg.schedule}"
    )

    # ── 5. Assemble and move to device ────────────────────────────────────────
    model = LatentDiffusionModel(
        patch_embedding=patch_embedding,
        encoder=encoder,
        dit=dit,
        diffusion=diffusion,
        freeze_encoder=dit_cfg.freeze_encoder,
    ).to(device)

    logger.info(model.summary())
    return model


# ─── Data ─────────────────────────────────────────────────────────────────────


def create_dataloaders(
    dit_cfg: DiTConfig,
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val DataLoaders using your existing AmexDataset.

    The dataset must yield batches of (x, y, time_tensor) where:
        x:           (B, T, F)   float32 features
        y:           (B,)        int64 labels
        time_tensor: (B, T, 1)   float32 month deltas

    Adapt this function to match your actual data pipeline.
    """
    import pandas as pd

    # Reuse data loading logic from your existing train_vq_vae.py
    # (abbreviated here — plug in your full implementation)
    logger.info("Loading AMEX dataset...")

    # Placeholder: replace with your actual dataset instantiation
    train_dataset = AmexDataset(split="train", debug=dit_cfg.debug, debug_size=dit_cfg.debug_size)
    val_dataset   = AmexDataset(split="val",   debug=dit_cfg.debug, debug_size=dit_cfg.debug_size // 5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=dit_cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dit_cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader


# ─── Training ─────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: LatentDiffusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    dit_cfg: DiTConfig,
    logger: logging.Logger,
    wandb_run=None,
) -> Dict[str, float]:
    """
    Train DiT for one epoch.

    Batch flow:
        x, y, time_tensor → model.training_step() → loss.backward()

    Shape annotations (AMEX example):
        x           : (256, 13, 188)
        y           : (256,)
        time_tensor : (256, 13, 1)
        z_e         : (256, 13, 64)   encoder output (frozen)
        t           : (256,)          sampled timesteps
        z_t         : (256, 13, 64)   noisy latent
        noise_pred  : (256, 13, 64)   DiT prediction
        loss        : scalar

    Args:
        model:     LatentDiffusionModel
        loader:    Training DataLoader
        optimizer: AdamW optimizer
        device:    Target device
        epoch:     Current epoch index (0-based)
        dit_cfg:   Config
        logger:    Logger
        wandb_run: Optional W&B run

    Returns:
        dict: {"loss": float}
    """
    model.dit.train()          # DiT in train mode (dropout active)
    model.encoder.eval()       # Encoder frozen — keep BN/LN in eval mode
    model.patch_embedding.eval()

    running_loss = 0.0
    n_batches    = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for step, batch in enumerate(pbar):
        # ── Unpack batch ──────────────────────────────────────────────────────
        # Adjust to your dataset's __getitem__ return signature
        x, y, time_tensor = batch
        # x:           (B, T, F)  — raw features
        # y:           (B,)       — int labels
        # time_tensor: (B, T, 1)  — time delta

        x           = x.to(device, dtype=torch.float32, non_blocking=True)
        time_tensor = time_tensor.to(device, dtype=torch.float32, non_blocking=True)
        y           = y.to(device, dtype=torch.long, non_blocking=True) if y is not None else None

        # ── Forward + loss ────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        output = model.training_step(x, y, time_tensor)
        # output["loss"]       : scalar
        # output["noise_pred"] : (B, T', D)
        # output["t"]          : (B,)

        loss = output["loss"]

        # ── Backward + gradient clip ──────────────────────────────────────────
        loss.backward()
        nn.utils.clip_grad_norm_(model.dit.parameters(), dit_cfg.max_grad_norm)
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        n_batches    += 1

        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        # ── Step-level logging ────────────────────────────────────────────────
        if step % dit_cfg.log_freq == 0:
            global_step = epoch * len(loader) + step
            logger.debug(f"  [step {step:>5d}] loss={loss_val:.5f}")

            if wandb_run is not None:
                wandb_run.log({
                    "train/step_loss":  loss_val,
                    "train/t_mean":     output["t"].float().mean().item(),
                    "global_step":      global_step,
                })

    epoch_loss = running_loss / max(n_batches, 1)
    return {"loss": epoch_loss}


@torch.no_grad()
def validate_one_epoch(
    model: LatentDiffusionModel,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    wandb_run=None,
) -> Dict[str, float]:
    """
    Evaluate DiT on the validation set (no gradient updates).

    Returns:
        dict: {"loss": float}
    """
    model.eval()
    running_loss = 0.0
    n_batches    = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)

    for batch in pbar:
        x, y, time_tensor = batch
        x           = x.to(device, dtype=torch.float32, non_blocking=True)
        time_tensor = time_tensor.to(device, dtype=torch.float32, non_blocking=True)
        y           = y.to(device, dtype=torch.long, non_blocking=True) if y is not None else None

        output = model.training_step(x, y, time_tensor)
        loss_val = output["loss"].item()

        running_loss += loss_val
        n_batches    += 1
        pbar.set_postfix({"val_loss": f"{loss_val:.4f}"})

    val_loss = running_loss / max(n_batches, 1)

    if wandb_run is not None:
        wandb_run.log({"val/step_loss": val_loss, "epoch": epoch + 1})

    return {"loss": val_loss}


# ─── Checkpointing ────────────────────────────────────────────────────────────


def save_checkpoint(
    model: LatentDiffusionModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    output_dir: str,
    best: bool = False,
) -> Path:
    """Save full model checkpoint."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = "best_dit.pt" if best else f"dit_epoch_{epoch+1:04d}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":     model.state_dict(),
            "dit_state_dict":       model.dit.state_dict(),     # Easy to extract DiT alone
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics":              metrics,
        },
        out / fname,
    )
    return out / fname


def load_checkpoint(
    path: str,
    model: LatentDiffusionModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[int, dict]:
    """Load checkpoint and return (start_epoch, metrics)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt.get("metrics", {})


# ─── Main Training Loop ───────────────────────────────────────────────────────


def train_latent_diffusion(
    model: LatentDiffusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    dit_cfg: DiTConfig,
    logger: logging.Logger,
    wandb_run=None,
    resume_from: Optional[str] = None,
) -> List[dict]:
    """
    Full DiT training loop.

    Optimizer:  AdamW (only DiT params, encoder frozen)
    Scheduler:  Linear warmup → CosineAnnealingLR
    Loss:       DDPM MSE (predict noise)

    Training objective at each step:
        ε ~ N(0,I);  z_t = √ᾱ_t·z_e + √(1-ᾱ_t)·ε
        L = E_{z_e, t, ε} [ || ε - DiT(z_t, t, y) ||² ]

    Args:
        model:        LatentDiffusionModel
        train_loader: Training DataLoader
        val_loader:   Validation DataLoader
        dit_cfg:      DiT hyperparameters
        logger:       Python logger
        wandb_run:    Optional W&B run
        resume_from:  Optional checkpoint path to resume from

    Returns:
        history: List of per-epoch metric dicts
    """
    device = next(model.parameters()).device

    # ── Optimizer (only DiT params) ────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=dit_cfg.learning_rate,
        weight_decay=dit_cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # ── LR scheduler: warmup → cosine decay ───────────────────────────────────
    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=dit_cfg.warmup_epochs,
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=max(dit_cfg.num_epochs - dit_cfg.warmup_epochs, 1),
        eta_min=dit_cfg.learning_rate * 0.01,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[dit_cfg.warmup_epochs],
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_val_loss = float("inf")
    history: List[dict] = []

    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        start_epoch, _ = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch += 1
        logger.info(f"Resumed at epoch {start_epoch}")

    output_dir = Path(dit_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting Latent Diffusion (DiT) Training")
    logger.info(f"  Epochs       : {dit_cfg.num_epochs}")
    logger.info(f"  Batch size   : {dit_cfg.batch_size}")
    logger.info(f"  LR           : {dit_cfg.learning_rate}")
    logger.info(f"  Timesteps    : {dit_cfg.timesteps}")
    logger.info(f"  Schedule     : {dit_cfg.schedule}")
    logger.info(f"  Freeze enc   : {dit_cfg.freeze_encoder}")
    logger.info(f"  CFG dropout  : {dit_cfg.cfg_dropout_prob}")
    logger.info("=" * 60)

    for epoch in range(start_epoch, dit_cfg.num_epochs):

        # ── Train ─────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            epoch=epoch, dit_cfg=dit_cfg, logger=logger, wandb_run=wandb_run,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        val_metrics = validate_one_epoch(
            model, val_loader, device,
            epoch=epoch, logger=logger, wandb_run=wandb_run,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Log ───────────────────────────────────────────────────────────────
        logger.info(
            f"Epoch {epoch+1:>4d}/{dit_cfg.num_epochs} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"lr={current_lr:.3e}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "epoch":             epoch + 1,
                "train/epoch_loss":  train_metrics["loss"],
                "val/epoch_loss":    val_metrics["loss"],
                "learning_rate":     current_lr,
            })

        epoch_record = {
            "epoch":  epoch,
            "train":  train_metrics,
            "val":    val_metrics,
            "lr":     current_lr,
        }
        history.append(epoch_record)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % dit_cfg.checkpoint_freq == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, epoch, epoch_record, dit_cfg.output_dir
            )
            logger.info(f"  ✓ Saved checkpoint: {ckpt_path.name}")

        if dit_cfg.save_best and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = save_checkpoint(
                model, optimizer, scheduler, epoch, epoch_record,
                dit_cfg.output_dir, best=True,
            )
            logger.info(f"  ★ New best model saved (val_loss={best_val_loss:.5f})")
            if wandb_run is not None:
                wandb_run.save(str(best_path))

    # ── Save history ──────────────────────────────────────────────────────────
    hist_path = output_dir / "dit_training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Training history saved to {hist_path}")

    return history


# ─── Sampling ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def sample_latents(
    model: LatentDiffusionModel,
    batch_size: int,
    seq_len: int,
    latent_dim: int,
    device: torch.device,
    y: Optional[torch.Tensor] = None,
    cfg_scale: float = 1.0,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Generate synthetic encoder latent sequences via reverse diffusion.

    Starts from pure Gaussian noise and iteratively denoises using DiT,
    optionally guided by class labels via classifier-free guidance.

    Args:
        model:         Trained LatentDiffusionModel
        batch_size:    Number of samples to generate
        seq_len:       T' — encoder output seq length (e.g. 13 for AMEX)
        latent_dim:    D  — encoder embedding_dim (e.g. 64)
        device:        Target device
        y:             (batch_size,) class labels for conditioning.
                         None → unconditional (or CFG null branch)
        cfg_scale:     Classifier-free guidance scale:
                         1.0 → no guidance (unconditional)
                         2.0 → moderate guidance
                         4.0 → strong guidance (may sacrifice diversity)
        show_progress: Show tqdm denoising progress bar

    Returns:
        z_samples: (batch_size, seq_len, latent_dim) synthetic latents

    Shape trace:
        z_T ~ N(0,I) : (B, S, D)   start from pure noise
        loop t=T-1…0 :              iterative denoising
        z_0          : (B, S, D)   final synthetic latents

    Example:
        >>> # Unconditional
        >>> z = sample_latents(model, 16, 13, 64, device)
        >>> print(z.shape)   # (16, 13, 64)

        >>> # Class-conditional with CFG guidance
        >>> y = torch.zeros(16, dtype=torch.long, device=device)  # all class 0
        >>> z = sample_latents(model, 16, 13, 64, device, y=y, cfg_scale=2.0)
        >>> print(z.shape)   # (16, 13, 64)
    """
    return model.sample_latents(
        batch_size=batch_size,
        seq_len=seq_len,
        latent_dim=latent_dim,
        device=device,
        y=y,
        cfg_scale=cfg_scale,
        show_progress=show_progress,
    )


# ─── Setup Helpers ────────────────────────────────────────────────────────────


def setup_logging(output_dir: str) -> logging.Logger:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out / "dit_training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("latent_diffusion")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Example Training Snippet ─────────────────────────────────────────────────


def example_training_snippet():
    """
    Minimal self-contained example demonstrating the full pipeline.

    Substitute your real AmexDataset and VQVAEConfig.
    All tensor shapes annotated at every stage.

    Typical AMEX dimensions:
        B = 256   (batch_size)
        T = 13    (months per customer)
        F = 188   (features)
        D = 64    (encoder embedding_dim / latent_dim)
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dummy AMEX-shaped data ────────────────────────────────────────────────
    B, T, F = 256, 13, 188        # batch, months, features
    N = B * 40                    # total synthetic customers

    x_all    = torch.randn(N, T, F)       # (N, 13, 188)  features
    y_all    = torch.randint(0, 2, (N,))  # (N,)          binary labels
    time_all = torch.rand(N, T, 1)        # (N, 13, 1)    time deltas

    dataset      = TensorDataset(x_all, y_all, time_all)
    train_loader = DataLoader(dataset[:int(0.8*N)], batch_size=B, shuffle=True)
    val_loader   = DataLoader(dataset[int(0.8*N):], batch_size=B, shuffle=False)

    # ── VQ-VAE config (must match your pretrained encoder) ───────────────────
    vqvae_cfg = VQVAEConfig()
    vqvae_cfg.input_dim      = F    # 188
    vqvae_cfg.patch_size     = 1    # no striding for seq_len=13
    vqvae_cfg.patch_stride   = 1
    vqvae_cfg.patch_embed_dim = 64
    vqvae_cfg.hidden_dim     = 128
    vqvae_cfg.embedding_dim  = 64   # D — latent dim for DiT
    vqvae_cfg.num_layers     = 2
    vqvae_cfg.num_heads      = 4
    vqvae_cfg.ff_dim         = 256
    vqvae_cfg.dropout        = 0.1
    vqvae_cfg.use_class_token = True
    vqvae_cfg.class_proj_dim  = 1
    vqvae_cfg.class_func      = "entropy"
    vqvae_cfg.koleo_penalty_weight = None

    # ── DiT config ────────────────────────────────────────────────────────────
    dit_cfg = DiTConfig(
        hidden_dim=256,
        cond_dim=256,
        num_layers=4,
        num_heads=8,
        ff_multiplier=4,
        dropout=0.1,
        num_classes=2,              # Binary AMEX labels
        cfg_dropout_prob=0.10,
        timesteps=1_000,
        schedule="cosine",
        num_epochs=3,               # Increase to 100+ for real training
        batch_size=B,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_epochs=1,
        freeze_encoder=True,
        vqvae_checkpoint=None,      # Set to path of your best_model.pt
        output_dir="./dit_output",
        log_freq=10,
        use_wandb=False,
    )

    logger = setup_logging(dit_cfg.output_dir)
    set_seed(dit_cfg.seed)

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_latent_diffusion_model(vqvae_cfg, dit_cfg, device, logger)

    # ── Train ─────────────────────────────────────────────────────────────────
    history = train_latent_diffusion(model, train_loader, val_loader, dit_cfg, logger)
    print(f"\nFinal train loss: {history[-1]['train']['loss']:.5f}")
    print(f"Final val   loss: {history[-1]['val']['loss']:.5f}")

    # ── Sampling ──────────────────────────────────────────────────────────────
    model.eval()
    latent_dim = vqvae_cfg.embedding_dim  # D = 64

    # (A) Unconditional sampling
    z_uncond = sample_latents(
        model,
        batch_size=8,
        seq_len=T,           # 13
        latent_dim=latent_dim,  # 64
        device=device,
        y=None,
        cfg_scale=1.0,
    )
    print(f"\n[Unconditional] z_uncond: {z_uncond.shape}")
    # Expected: (8, 13, 64)

    # (B) Class-conditional sampling — generate class=0 (non-default) customers
    y_class0 = torch.zeros(8, dtype=torch.long, device=device)
    z_class0 = sample_latents(
        model,
        batch_size=8,
        seq_len=T,
        latent_dim=latent_dim,
        device=device,
        y=y_class0,
        cfg_scale=1.0,
    )
    print(f"[Class-0 cond ] z_class0: {z_class0.shape}")
    # Expected: (8, 13, 64)

    # (C) Class-conditional with CFG guidance (stronger class adherence)
    y_class1 = torch.ones(8, dtype=torch.long, device=device)
    z_cfg = sample_latents(
        model,
        batch_size=8,
        seq_len=T,
        latent_dim=latent_dim,
        device=device,
        y=y_class1,
        cfg_scale=2.0,       # > 1.0 activates CFG
    )
    print(f"[CFG scale=2.0] z_cfg:    {z_cfg.shape}")
    # Expected: (8, 13, 64)

    return z_uncond, z_class0, z_cfg


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def main():
    """
    CLI entry point. Reads args and runs full training.

    Minimal usage:
        python train_latent_diffusion.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train Latent Diffusion Transformer on AMEX data")
    parser.add_argument("--vqvae_checkpoint", type=str, default=None,
                        help="Path to pretrained VQ-VAE checkpoint (.pt)")
    parser.add_argument("--output_dir",       type=str, default="./dit_output")
    parser.add_argument("--num_epochs",       type=int, default=100)
    parser.add_argument("--batch_size",       type=int, default=256)
    parser.add_argument("--learning_rate",    type=float, default=1e-4)
    parser.add_argument("--timesteps",        type=int, default=1_000)
    parser.add_argument("--schedule",         type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--num_layers",       type=int, default=6)
    parser.add_argument("--hidden_dim",       type=int, default=256)
    parser.add_argument("--num_heads",        type=int, default=8)
    parser.add_argument("--num_classes",      type=int, default=2)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.1)
    parser.add_argument("--freeze_encoder",   action="store_true", default=True)
    parser.add_argument("--use_wandb",        action="store_true", default=False)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--device",           type=str, default="cuda")
    parser.add_argument("--debug",            action="store_true", default=False)
    parser.add_argument("--resume_from",      type=str, default=None)
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build configs
    vqvae_cfg = VQVAEConfig()   # ← Replace with your actual config or args parsing
    dit_cfg   = DiTConfig(
        vqvae_checkpoint = args.vqvae_checkpoint,
        num_epochs       = args.num_epochs,
        batch_size       = args.batch_size,
        learning_rate    = args.learning_rate,
        timesteps        = args.timesteps,
        schedule         = args.schedule,
        num_layers       = args.num_layers,
        hidden_dim       = args.hidden_dim,
        num_heads        = args.num_heads,
        num_classes      = args.num_classes,
        cfg_dropout_prob = args.cfg_dropout_prob,
        freeze_encoder   = args.freeze_encoder,
        use_wandb        = args.use_wandb,
        output_dir       = args.output_dir,
        seed             = args.seed,
        debug            = args.debug,
    )

    # W&B
    wandb_run = None
    if dit_cfg.use_wandb and HAS_WANDB:
        wandb_run = wandb.init(
            project=dit_cfg.wandb_project,
            name=dit_cfg.wandb_run_name,
            config=asdict(dit_cfg),
        )

    # Data
    train_loader, val_loader = create_dataloaders(dit_cfg, logger)
    gc.collect()

    # Model
    model = build_latent_diffusion_model(vqvae_cfg, dit_cfg, device, logger)

    # Train
    train_latent_diffusion(
        model, train_loader, val_loader, dit_cfg, logger,
        wandb_run=wandb_run,
        resume_from=args.resume_from,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
