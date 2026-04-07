"""
Training script for VQ-VAE on AMEX time-series data.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import json

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from data.data import AmexDataset
from vq_vae.vq_vae import VQ_VAE
from vq_vae.config import VQVAEConfig
from args import get_args


def setup_logging(output_dir):
    """Configure logging."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def setup_wandb(args, logger):
    """Initialize Weights & Biases logging."""
    if not args.use_wandb:
        logger.info("Weights & Biases logging disabled")
        return None
    
    if not HAS_WANDB:
        logger.warning("Weights & Biases not installed. Skipping wandb initialization.")
        return None
    
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=None,  # Auto-generate name
            notes=args.wandb_notes,
            tags=args.wandb_tags if args.wandb_tags else None,
            config=vars(args),
            reinit=False,
        )
        logger.info(f"Initialized Weights & Biases run: {run.name}")
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        return None


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(args, logger):
    """Create training and validation dataloaders."""
    logger.info("Loading data...")
    
    # Load customer data
    train_data = pd.read_csv(args.train_data)
    train_labels = pd.read_csv(args.train_labels)
    
    # Split into train and validation (80-20)
    n_customers = len(train_data)
    val_size = int(0.2 * n_customers)
    
    indices = np.random.permutation(n_customers)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_customers = train_data.iloc[train_indices]
    val_customers = train_data.iloc[val_indices]
    
    if args.debug:
        train_customers = train_customers.iloc[:args.debug_size]
        val_customers = val_customers.iloc[:args.debug_size // 5]
    
    logger.info(f"Training customers: {len(train_customers)}, Validation customers: {len(val_customers)}")
    
    # Create datasets
    logger.info("Creating training dataset...")
    train_dataset = AmexDataset(
        customer_df=train_customers,
        db_path=str(args.db_path),
        max_seq_len=args.max_seq_len,
    )
    
    logger.info("Creating validation dataset...")
    val_dataset = AmexDataset(
        customer_df=val_customers,
        db_path=str(args.db_path),
        fill_dict=train_dataset.fill_dict,
        transformer=train_dataset.transformer,
        max_seq_len=args.max_seq_len,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, train_dataset


def create_model_and_optimizer(args, logger):
    """Create VQ-VAE model and optimizer."""
    logger.info("Creating VQ-VAE model...")
    
    config = VQVAEConfig(
        input_dim=args.input_dim,
        output_dim=args.input_dim,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        encoder_num_layers=args.encoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        encoder_dropout=args.dropout,
        decoder_num_layers=args.decoder_num_layers,
        decoder_num_heads=args.decoder_num_heads,
        decoder_dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        commitment_loss_weight=args.commitment_loss_weight,
        device=args.device,
        seed=args.seed,
    )
    
    logger.info(f"Model config:\n{config}")
    
    model = VQ_VAE(config)
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-6,
    )
    
    return model, optimizer, scheduler, config


def train_epoch(model, train_loader, optimizer, device, logger, log_freq, wandb_run=None, epoch=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=True)
    
    for batch_idx, batch in enumerate(pbar):
        # Get batch data
        x = batch.to(device)  # Shape: (batch_size, seq_len, input_dim)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        
        loss = output["total_loss"]
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate losses
        batch_total_loss = output["total_loss"].detach().item()
        batch_recon_loss = output["reconstruction_loss"].detach().item()
        batch_vq_loss = output["vq_loss"].detach().item()
        
        total_loss += batch_total_loss
        total_recon_loss += batch_recon_loss
        total_vq_loss += batch_vq_loss
        num_batches += 1
        
        # Log to wandb at batch level
        if wandb_run is not None and (batch_idx + 1) % log_freq == 0:
            step = epoch * len(train_loader) + batch_idx if epoch is not None else batch_idx
            wandb_run.log({
                "train/batch_total_loss": batch_total_loss,
                "train/batch_reconstruction_loss": batch_recon_loss,
                "train/batch_vq_loss": batch_vq_loss,
                "train/batch": step,
            })
        
        if (batch_idx + 1) % log_freq == 0:
            avg_loss = total_loss / num_batches
            avg_recon = total_recon_loss / num_batches
            avg_vq = total_vq_loss / num_batches
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "recon": f"{avg_recon:.4f}",
                "vq": f"{avg_vq:.4f}",
            })
    
    return {
        "loss": total_loss / num_batches,
        "reconstruction_loss": total_recon_loss / num_batches,
        "vq_loss": total_vq_loss / num_batches,
    }


def validate(model, val_loader, device, logger, wandb_run=None, epoch=None):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            x = batch.to(device)
            
            output = model(x)
            
            batch_total_loss = output["total_loss"].item()
            batch_recon_loss = output["reconstruction_loss"].item()
            batch_vq_loss = output["vq_loss"].item()
            
            total_loss += batch_total_loss
            total_recon_loss += batch_recon_loss
            total_vq_loss += batch_vq_loss
            num_batches += 1
            
            pbar.set_postfix({
                "val_loss": f"{total_loss / num_batches:.4f}",
            })
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_vq_loss = total_vq_loss / num_batches
    
    return {
        "loss": avg_loss,
        "reconstruction_loss": avg_recon_loss,
        "vq_loss": avg_vq_loss,
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, output_dir, best=False):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if best:
        checkpoint_path = output_dir / "best_model.pt"
    else:
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint["epoch"], checkpoint["metrics"]


def main():
    """Main training loop."""
    # Parse arguments
    args = get_args()
    
    # Setup
    logger = setup_logging(args.output_dir)
    wandb_run = setup_wandb(args, logger)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    
    logger.info(f"Starting training with args:\n{vars(args)}")
    
    # Create dataloaders
    train_loader, val_loader, train_dataset = create_dataloaders(args, logger)
    
    # Create model and optimizer
    model, optimizer, scheduler, config = create_model_and_optimizer(args, logger)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")
    
    if args.resume_from and Path(args.resume_from).exists():
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch, _ = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, device
        )
        start_epoch += 1
    
    # Training loop
    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, logger, args.log_freq, wandb_run, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, logger, wandb_run, epoch)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log metrics
        logger.info(f"\nTrain Loss: {train_metrics['loss']:.6f}")
        logger.info(f"  - Reconstruction: {train_metrics['reconstruction_loss']:.6f}")
        logger.info(f"  - VQ Loss: {train_metrics['vq_loss']:.6f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.6f}")
        logger.info(f"  - Reconstruction: {val_metrics['reconstruction_loss']:.6f}")
        logger.info(f"  - VQ Loss: {val_metrics['vq_loss']:.6f}")
        logger.info(f"Learning Rate: {current_lr:.2e}")
        
        # Log to wandb at epoch level
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/total_loss": train_metrics["loss"],
                "train/reconstruction_loss": train_metrics["reconstruction_loss"],
                "train/vq_loss": train_metrics["vq_loss"],
                "val/total_loss": val_metrics["loss"],
                "val/reconstruction_loss": val_metrics["reconstruction_loss"],
                "val/vq_loss": val_metrics["vq_loss"],
                "learning_rate": current_lr,
            })
        
        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": current_lr,
        }
        history.append(epoch_metrics)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_freq == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, epoch_metrics, args.output_dir
            )
            logger.info(f"Saved checkpoint at epoch {epoch + 1}")
            
            # Upload checkpoint to wandb
            if wandb_run is not None:
                wandb_run.save(str(checkpoint_path))
        
        # Save best model
        if args.save_best and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, epoch_metrics, args.output_dir, best=True
            )
            logger.info(f"Saved best model with val loss: {best_val_loss:.6f}")
            
            # Upload best model to wandb
            if wandb_run is not None:
                wandb_run.save(str(checkpoint_path))
    
    # Save training history
    history_file = output_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Saved training history to {history_file}")
    
    # Finalize wandb run
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("Finished Weights & Biases logging")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
