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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from vq_vae.vq_vae import VQ_VAE, PatchEmbedding
from vq_vae.encoder import TransformerEncoder

from data.data import AmexDataset, SyntheticAmexDataset

from NCSN import AmexGuidedGenerator, NCSNConfig

from args import get_ncsn_parser, create_config_from_args, create_ncsn_config_from_args

class TSTREvaluator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.patch_embedding = PatchEmbedding(
            input_dim=config.input_dim,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_embed_dim=config.patch_embed_dim,
        )
        self.encoder = TransformerEncoder(
            input_dim=config.patch_embed_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            activation="relu",
            class_token=config.use_class_token,
            class_proj_dim=config.class_proj_dim,
            class_func=config.class_func,
            koleo_penalty_weight=config.koleo_penalty_weight,
        )

    def forward(self, x, y, time_tensor):
        x = self.patch_embedding(x)
        return self.encoder(x, y, time_tensor)

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

def compute_classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (numpy array or list)
        y_pred: Predicted labels (numpy array or list)
    
    Returns:
        Dictionary containing accuracy, balanced_accuracy, precision, and recall
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    return metrics


def create_confusion_matrix_plot(y_true, y_pred):
    """
    Create a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        matplotlib figure object for logging to wandb
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = np.unique(y_true)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    return fig

def build_ncsn(args, logger) -> AmexGuidedGenerator:
    vq_vae_config = create_config_from_args(args)
    ncsn_config = create_ncsn_config_from_args(args)
    vq_vae = VQ_VAE(vq_vae_config)
    model = TSTREvaluator(config=vq_vae_config)
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
    return ncsn, model

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

def generate_synthetic_dataset_to_disk(
    ncsn_model, real_loader, logger:logging.Logger, device, save_dir="synthetic_data", steps=50, guidance_scale=2.0
):
    os.makedirs(save_dir, exist_ok=True)
    ncsn_model.eval()
    
    logger.info(f"Generating synthetic data using {ncsn_model.config.denoiser_model.upper()}...")
    
    total_samples = len(real_loader.dataset)
    current_idx = 0
    first_batch_x, first_batch_time, _ = next(iter(real_loader))
    seq_len = first_batch_x.size(1)
    input_dim = first_batch_x.size(2)
    x_path = os.path.join(save_dir, 'fake_x.npy')
    time_path = os.path.join(save_dir, 'fake_time.npy')
    target_path = os.path.join(save_dir, 'fake_targets.npy')
    
    fake_x_memmap = np.memmap(x_path, dtype='float32', mode='w+', shape=(total_samples, seq_len, input_dim))
    fake_time_memmap = np.memmap(time_path, dtype='float32', mode='w+', shape=(total_samples, seq_len, 1))
    fake_targets_memmap = np.memmap(target_path, dtype='float32', mode='w+', shape=(total_samples,))

    with torch.no_grad():
        for real_x, time_tensor, targets in tqdm(real_loader, desc="Streaming to Disk"):
            batch_size = real_x.size(0)
            time_tensor = time_tensor.to(device)
            targets = targets.to(device)
            fake_x = ncsn_model.generate(
                batch_size=batch_size,
                seq_len=seq_len,
                target_class=targets,
                time_tensor=time_tensor,
                steps=steps,
                guidance_scale=guidance_scale
            )
            
            fake_x_memmap[current_idx : current_idx + batch_size] = fake_x.cpu().numpy()
            fake_time_memmap[current_idx : current_idx + batch_size] = time_tensor.cpu().numpy()
            fake_targets_memmap[current_idx : current_idx + batch_size] = targets.cpu().numpy()
            
            fake_x_memmap.flush()
            fake_time_memmap.flush()
            fake_targets_memmap.flush()
            
            current_idx += batch_size
            
    metadata = {
        "total_samples": total_samples,
        "seq_len": seq_len,
        "input_dim": input_dim
    }

    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(metadata, f)
        
    logger.info(f"Generation complete. Data safely written to {save_dir}/")
    return save_dir

def save_checkpoint(args, model, optimizer: torch.optim.Optimizer, epoch: int, config: NCSNConfig, logger: logging.Logger):
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"ncsn_epoch_{epoch}.pth"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': asdict(config),
    }, checkpoint_path)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")

def train_model(args, model, optimizer, train_loader, val_loader, logger, wandb_run):
    device = next(model.parameters()).device
    start_epoch = 1
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_metrics = train_one_epoch(args, model, optimizer, train_loader, device, epoch, logger, wandb_run)
        val_metrics = validate_one_epoch(args, model, val_loader, device, epoch, logger)

        if wandb_run is not None:
            wandb_run.log({**train_metrics, **val_metrics})
        
        if epoch % args.checkpoint_freq == 0:
            save_checkpoint(args, model, optimizer, epoch, model.config, logger)
    
    logger.info(f"\nTrain Loss: {train_metrics['train/loss']:.6f}")

    logger.info(f"Val Loss: {val_metrics['val/loss']:.6f}")
    logger.info(f"  - Classification Loss: {val_metrics['val/loss']:.6f}")
    logger.info(f"  - Accuracy: {val_metrics['val/acc']:.4f}")
    logger.info(f"  - Balanced Accuracy: {val_metrics['val/bal_acc']:.4f}")
    logger.info(f"  - Precision: {val_metrics['val/precision']:.4f}")
    logger.info(f"  - Recall: {val_metrics['val/recall']:.4f}")
    
    if wandb_run is not None:
            cm_fig = val_metrics.pop('val/conf_mat', None)
            wandb_log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["train/loss"],
                "val/acc":val_metrics["val/acc"],
                "val/bal_acc": val_metrics["val/bal_acc"],
                "val/precision": val_metrics["val/precision"],
                "val/recall": val_metrics["val/recall"],
                "val/loss":val_metrics["val/loss"],
            }
            wandb_log_dict["val/conf_mat"] = wandb.Image(cm_fig)
            wandb_run.log(wandb_log_dict)
            plt.close(cm_fig)

def train_one_epoch(args, model, optimizer, train_loader, device, epoch, logger, wandb_run):
    model.train()
    total_loss = 0.0
    step_idx = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

    for batch_idx, (features, time_tensor, targets) in enumerate(pbar):
        features, time_tensor, targets = features.to(device), time_tensor.to(device), targets.to(device)
        optimizer.zero_grad()
        z, cls_logits, cls_loss = model(features, targets, time_tensor)
        cls_loss.backward()
        optimizer.step()

        total_loss +=cls_loss.cpu().item()
        
        if step_idx % args.log_freq==0:
            pbar.set_postfix({"loss": cls_loss.item()})
            if wandb_run is not None:
                wandb_run.log({"train_freq/loss": cls_loss.item(), "epoch": epoch})
        step_idx += 1

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}")
    return {"epoch": epoch, "train/loss": avg_loss}

@torch.no_grad()
def validate_one_epoch(args, model, val_loader, device, epoch, logger):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    step_idx =0
    pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}", dynamic_ncols=True)
    for batch_idx, (features, time_tensor, targets) in enumerate(pbar):
        features, time_tensor, targets = features.to(device), time_tensor.to(device), targets.to(device)
        z, cls_logits, cls_loss = model(features, targets, time_tensor)
        total_loss+=cls_loss.item()
        predictions = torch.argmax(cls_logits, dim=1).cpu().numpy()
        all_labels.extend(targets.cpu().numpy())
        all_predictions.extend(predictions)
        if step_idx % args.log_freq==0:
            pbar.set_postfix({"loss": cls_loss.item()})
        step_idx+=1

    cls_metrics = compute_classification_metrics(all_labels, all_predictions)
    metrics_dict = {
        "val/acc":cls_metrics["accuracy"],
        "val/bal_acc": cls_metrics["balanced_accuracy"],
        "val/precision": cls_metrics["precision"],
        "val/recall": cls_metrics["recall"],
        "val/loss":total_loss/len(val_loader)
    }
    cm_fig = create_confusion_matrix_plot(all_labels, all_predictions)
    metrics_dict['val/conf_mat'] = cm_fig
    return metrics_dict

def main():
    args = get_ncsn_parser().parse_args()
    logger = setup_logging(args.output_dir)
    train_loader, val_loader, _ = create_dataloaders(args, logger)
    set_seed(args.seed)
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
    
    ncsn_model, model = build_ncsn(args, logger)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    ncsn_model.to(device)
    synth_dir = generate_synthetic_dataset_to_disk(ncsn_model, train_loader, logger, device)
    model_optim = optim.AdamW(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    synth_data = DataLoader(SyntheticAmexDataset(synth_dir), batch_size=args.batch_size, num_workers=args.num_workers)
    train_model(args, model, model_optim, synth_data, val_loader, logger, wandb_run)

if __name__ == "__main__":
    main()
