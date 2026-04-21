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

from latent_diffusion import DiT, GaussianDiffusion, LatentDiffusionModel, DiTConfig

from vq_vae.vq_vae import VQ_VAE
from vq_vae.config import VQVAEConfig

from data.data import AmexDataset

def build_latent_diffusion_model(
    vqvae_cfg: VQVAEConfig,
    dit_cfg: DiTConfig,
    device: torch.device,
    logger: logging.Logger,
) -> LatentDiffusionModel:
    """
    Construct the full LatentDiffusionModel.

    1. Builds full VQ-VAE from vqvae_cfg
    2. Loads pretrained weights from vqvae_checkpoint (if provided)
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

    # ── 1. Build VQ-VAE backbone ─────────────────────────────────────────────
    vq_vae = VQ_VAE(vqvae_cfg)
    logger.info("Initialized full VQ-VAE backbone.")

    # ── 2. Load pretrained VQ-VAE weights ────────────────────────────────────
    if dit_cfg.vqvae_checkpoint is not None:
        ckpt_path = Path(dit_cfg.vqvae_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {ckpt_path}")

        logger.info(f"Loading pretrained VQ-VAE weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)

        vq_vae.load_state_dict(state, strict=True)
        logger.info("✓ Pretrained VQ-VAE weights loaded successfully.")
    else:
        logger.warning(
            "No VQ-VAE checkpoint provided — VQ-VAE initialized from scratch. "
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
        vq_vae=vq_vae,
        dit=dit,
        diffusion=diffusion,
        freeze_encoder=dit_cfg.freeze_encoder,
    ).to(device)

    logger.info(model.summary())
    return model

def create_dataloaders(
    args,
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader, AmexDataset]:
    """
    Create train/val DataLoaders using your existing AmexDataset.
    """
    import pandas as pd

    logger.info("Loading AMEX dataset...")
    logger.info(f"Label path: {args.train_labels}")

    try:
        train_data = pd.read_csv(
            args.train_labels,
            dtype={'customer_ID': str, 'target': 'int8'},
            low_memory=False,
        )
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise

    if not args.class_imbalance:
        train_data = train_data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        if 'customer_ID' not in train_data.columns or 'target' not in train_data.columns:
            raise ValueError("Missing 'customer_ID' or 'target' column in train data")
        
        train_data = train_data.dropna(subset=['customer_ID', 'target']).reset_index(drop=True)
        
        n_customers = len(train_data)
        val_size = int(0.2 * n_customers)
        indices = np.random.permutation(n_customers)
        
        train_customers = train_data.iloc[indices[val_size:]]
        val_customers = train_data.iloc[indices[:val_size]]
    else:
        logger.info("Using class imbalance handling with weighted sampling.")
        pos_customers = train_data[train_data['target'] == 1]
        neg_customers = train_data[train_data['target'] == 0]
        
        val_pos_size = int(0.2 * len(pos_customers))
        val_neg_size = int(0.2 * len(neg_customers))
        
        val_pos_indices = np.random.choice(pos_customers.index, size=val_pos_size, replace=False)
        val_neg_indices = np.random.choice(neg_customers.index, size=val_neg_size, replace=False)
        
        val_indices = np.concatenate([val_pos_indices, val_neg_indices])
        train_indices = np.setdiff1d(train_data.index, val_indices)
        
        train_customers = train_data.loc[train_indices].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        val_customers = train_data.loc[val_indices].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    if args.debug:
        train_customers = train_customers.iloc[:args.debug_size]
        val_customers = val_customers.iloc[:args.debug_size // 5]

    train_dataset = AmexDataset(
        customer_df=train_customers,
        db_path=str(args.db_path),
        max_seq_len=args.max_seq_len,
    )
    
    val_dataset = AmexDataset(
        customer_df=val_customers,
        db_path=str(args.db_path),
        fill_dict=train_dataset.fill_dict,
        transformer=train_dataset.transformer,
        max_seq_len=args.max_seq_len,
    )

    if args.class_imbalance:
        targets = train_customers['target'].values
        class_counts = np.bincount(targets)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[targets]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    return train_loader, val_loader, train_dataset

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
    model.vq_vae.eval()        # VQ-VAE frozen — keep BN/LN/Dropout in eval mode

    running_loss = 0.0
    n_batches    = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for step, batch in enumerate(pbar):
        # ── Unpack batch ──────────────────────────────────────────────────────
        # Adjust to your dataset's __getitem__ return signature
        x, time_tensor, y = batch

        x           = x.to(device, dtype=torch.float32, non_blocking=True)
        time_tensor = time_tensor.to(device, dtype=torch.float32, non_blocking=True)
        y           = y.to(device, dtype=torch.long, non_blocking=True) if y is not None else None

        # ── Forward + loss ────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        output = model.training_step(x, y, time_tensor)
        loss = output["loss"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.dit.parameters(), dit_cfg.max_grad_norm)
        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        n_batches    += 1

        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

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
        x, time_tensor, y = batch
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

def main():
    """
    CLI entry point. Reads args and runs full training.
    """
    from args import get_dit_args, create_config_from_args, create_dit_config_from_args

    args = get_dit_args()

    logger = setup_logging(args.output_dir)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build configs
    vqvae_cfg = create_config_from_args(args)
    dit_cfg   = create_dit_config_from_args(args)

    # W&B
    wandb_run = None
    if dit_cfg.use_wandb and HAS_WANDB:
        wandb_run = wandb.init(
            project=dit_cfg.wandb_project,
            name=dit_cfg.wandb_run_name,
            config=asdict(dit_cfg),
        )

    train_loader, val_loader, _ = create_dataloaders(args, logger)
    gc.collect()

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
