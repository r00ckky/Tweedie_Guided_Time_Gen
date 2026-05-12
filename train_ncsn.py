from __future__ import annotations

import gc
import os
import json
import logging
import math
from dataclasses import asdict, dataclass, field
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from vq_vae.vq_vae import VQ_VAE
from vq_vae.config import VQVAEConfig
from data.data import AmexDataset

from NCSN import AmexGuidedGenerator, TabularDenoiser, NCSNConfig, model

from args import get_ncsn_parser, create_config_from_args, create_ncsn_config_from_args

def build_ncsn(args, logger) -> AmexGuidedGenerator:
    vq_vae_config = create_config_from_args(args)
    ncsn_config = create_ncsn_config_from_args(args)
    vq_vae = VQ_VAE(vq_vae_config)

    try:
        if args.ncsn_weights is None and os.path.exists(ncsn_config.vq_vae_checkpoint):
            logger.info(f"Loading VQ-VAE checkpoint from {ncsn_config.vq_vae_checkpoint}")
            vq_vae.load_state_dict(torch.load(ncsn_config.vq_vae_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])

        elif args.ncsn_weights is not None and os.path.exists(args.ncsn_weights):
            logger.info(f"Loading NCSN checkpoint from {args.ncsn_weights}")
            vq_vae.load_state_dict(torch.load(args.ncsn_weights, map_location="cpu")["model_state_dict"], strict=False)

    except FileNotFoundError:
        raise FileNotFoundError(f"VQ-VAE checkpoint not found.{ncsn_config.vq_vae_checkpoint}")

    ncsn = AmexGuidedGenerator(vq_vae, ncsn_config)
    logger.info(f"Initialized NCSN with VQ-VAE checkpoint: {ncsn_config.vq_vae_checkpoint}")
    logger.info(f"NCSN Config: {asdict(ncsn_config)}")
    logger.info(f"VQ-VAE Config: {vq_vae.summary()}")
    return ncsn

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

    if args.transformer_path is not None and args.fill_dict_path is not None and os.path.exists(args.transformer_path) and os.path.exists(args.fill_dict_path):
        logger.info(f"Loading transformer and fill_dict from {args.transformer_path} and {args.fill_dict_path}")
        transformer = joblib.load(args.transformer_path)
        fill_dict = joblib.load(args.fill_dict_path)
    else:
        transformer = None
        fill_dict = None

    train_dataset = AmexDataset(
        customer_df=train_customers,
        db_path=str(args.db_path),
        max_seq_len=args.max_seq_len,
        transformer=transformer,
        fill_dict=fill_dict,
    )
    
    val_dataset = AmexDataset(
        customer_df=val_customers,
        db_path=str(args.db_path),
        fill_dict=train_dataset.fill_dict,
        transformer=train_dataset.transformer,
        max_seq_len=args.max_seq_len,
    )

    if args.class_imbalance:
        targets = train_customers['target'].values.copy()
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
    args,
    model: AmexGuidedGenerator,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    wandb_run = None,
):
    model.train()
    total_loss = 0.0
    step_idx = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    
    for batch_idx, (features, time_tensor, targets) in enumerate(pbar):
        features, time_tensor, targets = features.to(device), time_tensor.to(device), targets.to(device)
        optimizer.zero_grad()
        
        t = torch.rand(features.size(0), device=device) * (1. - 1e-5) + 1e-5
        sigma = model.get_sigma(t)
        sigma = sigma.view(-1, 1, 1)
        noise = torch.randn_like(features)
        noisy_features = features + (noise * sigma)
        if not args.train_classifier:
            score_pred = model.get_data_score(noisy_features, t)
            target_score = -noise / sigma
            
            loss = torch.mean((sigma ** 2) * (score_pred - target_score) ** 2)
        else:
            score_pred, cls_loss = model(noisy_features, t)
            target_score = -noise / sigma
            
            loss = torch.mean((sigma ** 2) * (score_pred - target_score) ** 2) + cls_loss
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(args, 'max_grad_norm', 1.0))
        optimizer.step()
        total_loss += loss.item()
        
        if step_idx % args.log_freq == 0:
            pbar.set_postfix({"loss": loss.item()})
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), "epoch": epoch})
        step_idx += 1

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    return {"epoch": epoch, "loss": avg_loss}

@torch.no_grad()
def validate_one_epoch(
    args,
    model: AmexGuidedGenerator,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}", dynamic_ncols=True)
    for batch_idx, (features, time_tensor, targets) in enumerate(pbar):
        features, time_tensor, targets = features.to(device), time_tensor.to(device), targets.to(device)
        
        t = torch.rand(features.size(0), device=device) * (1. - 1e-5) + 1e-5
        sigma = model.get_sigma(t)
        sigma = sigma.view(-1, 1, 1)
        
        noise = torch.randn_like(features)
        noisy_features = features + (noise * sigma)
        
        score_pred = model.get_data_score(noisy_features, t)
        target_score = -noise / sigma
        
        loss = torch.mean((sigma ** 2) * (score_pred - target_score) ** 2)
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch} - Val Avg Loss: {avg_loss:.4f}")
    return {"val/loss": avg_loss}

def save_checkpoint(model: AmexGuidedGenerator, optimizer: torch.optim.Optimizer, epoch: int, config: NCSNConfig, logger: logging.Logger):
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"ncsn_epoch_{epoch}.pth"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': asdict(config),
    }, checkpoint_path)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")

def train_ncsn(
        args,
        model: AmexGuidedGenerator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: logging.Logger,
        wandb_run = None,
):
    device = next(model.parameters()).device
    start_epoch = 1
    optimizer = AdamW(model.parameters(), lr=model.config.learning_rate, weight_decay=model.config.weight_decay)
    if args.ncsn_weights is not None and os.path.exists(args.ncsn_weights):
        logger.info(f"Loading NCSN optimizer checkpoint from {args.ncsn_weights} for resuming training")
        checkpoint = torch.load(args.ncsn_weights, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed training from {args.ncsn_weights} at epoch {start_epoch}")
    
    for epoch in range(start_epoch, model.config.num_epochs + 1):
        train_metrics = train_one_epoch(args, model, train_loader, optimizer, device, epoch, logger, wandb_run)
        val_metrics = validate_one_epoch(args, model, val_loader, device, epoch, logger)
        
        if wandb_run is not None:
            wandb_run.log({**train_metrics, **val_metrics})
        
        if epoch % args.checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, model.config, logger)

def setup_logging(output_dir: str) -> logging.Logger:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out / "ncsn_training.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("ncsn")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = get_ncsn_parser()
    args = parser.parse_args()
    
    set_seed(args.seed)
    logger = setup_logging(args.output_dir)
    
    if args.use_wandb and HAS_WANDB:
        if args.wandb_save_dir:
            os.makedirs(args.wandb_save_dir, exist_ok=True)
            
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            dir=args.wandb_save_dir,
            config=vars(args),
        )
    else:
        wandb_run = None
        if args.use_wandb and not HAS_WANDB:
            logger.warning("wandb not installed, proceeding without logging to Weights & Biases.")
    
    model = build_ncsn(args, logger)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_loader, val_loader, _ = create_dataloaders(args, logger)
    
    train_ncsn(args, model, train_loader, val_loader, logger, wandb_run)

if __name__ == "__main__":
    main()