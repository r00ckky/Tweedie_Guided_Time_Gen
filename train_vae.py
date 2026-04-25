"""
Training script for VAE on AMEX time-series data.
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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, confusion_matrix

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from data.data import AmexDataset
from vae.vae import VAE
from vae.config import VAEConfig
from args import get_args, create_config_from_args
import gc

torch.autograd.set_detect_anomaly(True)


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
    
    # Plot confusion matrix as heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    
    # Set labels and title
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
        wandb_kwargs = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": args.wandb_run_name,
            "notes": args.wandb_notes,
            "tags": args.wandb_tags if args.wandb_tags else None,
            "config": vars(args),
            "reinit": False,
        }
        
        if args.wandb_save_dir:
            wandb_kwargs["dir"] = str(args.wandb_save_dir)
        
        run = wandb.init(**wandb_kwargs)
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
    logger.info(f"Label path: {args.train_labels}")
    
    try:
        train_data = pd.read_csv(
            args.train_labels,
            dtype={'customer_ID': str, 'target': 'int8'},
            low_memory=False,
            nrows=None,
        )
        logger.info(f"Successfully loaded data with shape: {train_data.shape}")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise
    
    if not args.class_imbalance:
        train_data = train_data.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        
        logger.info("Validating data integrity...")
        if 'customer_ID' not in train_data.columns:
            raise ValueError("Missing 'customer_ID' column in train data")
        if 'target' not in train_data.columns:
            raise ValueError("Missing 'target' column in train data")
        
        null_count = train_data[['customer_ID', 'target']].isnull().sum().sum()
        if null_count > 0:
            logger.warning(f"Found {null_count} null values in critical columns. Dropping them...")
            train_data = train_data.dropna(subset=['customer_ID', 'target']).reset_index(drop=True)
        
        logger.info(f"Data validation complete. Final shape: {train_data.shape}")
        
        n_customers = len(train_data)
        val_size = int(0.2 * n_customers)
        
        indices = np.random.permutation(n_customers)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_customers = train_data.iloc[train_indices]
        val_customers = train_data.iloc[val_indices]
        logger.info(f"Training customers: {len(train_customers)}, Validation customers: {len(val_customers)}")
    
    if args.debug:
        train_customers = train_customers.iloc[:args.debug_size]
        val_customers = val_customers.iloc[:args.debug_size // 5]

    if args.class_imbalance:
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
        
        logger.info(f"Training customers: {len(train_customers)}, Validation customers: {len(val_customers)}")
    
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
    
    if args.class_imbalance:
        logger.info("Configuring WeightedRandomSampler for 50/50 batch generation...")
        targets = train_customers['target'].values
        
        class_counts = np.bincount(targets)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[targets]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
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
    """Create VAE model and optimizer."""
    logger.info("Creating VAE model...")
    
    config = VAEConfig(
        input_dim=args.input_dim,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_embed_dim=args.patch_embed_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        use_class_token=args.use_class_token,
        class_proj_dim=args.class_proj_dim,
        class_func=args.class_func,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        kl_loss_weight=args.kl_loss_weight,
        classification_loss_weight=args.classification_loss_weight,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        seed=args.seed,
        max_seq_length=args.max_seq_len,
    )
    
    logger.info(f"Model config:\n{config}")
    
    model = VAE(config)
    model = model.to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    return model, optimizer, config


def train_epoch(model, train_loader, optimizer, device, args, logger):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_cls_loss = 0.0
    
    # For classification metrics
    all_labels = []
    all_predictions = []
    
    pbar = tqdm(train_loader, desc="Training", unit="batch")
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 3:
            X, time_tensor, y = batch
            X = X.to(device)
            time_tensor = time_tensor.to(device)
            y = y.to(device)
        else:
            continue
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(X, y=y, time_tensor=time_tensor)
        
        loss = output["total_loss"]
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += output["reconstruction_loss"].item()
        total_kl_loss += output["kl_loss"].item()
        if output["classification_loss"] is not None:
            total_cls_loss += output["classification_loss"].item()
        
        # Collect predictions for classification metrics
        if output["classification_logits"] is not None:
            predictions = torch.argmax(output["classification_logits"], dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(y.cpu().numpy())
            
        pbar.set_postfix({
            "loss": total_loss / (batch_idx + 1),
            "recon": total_recon_loss / (batch_idx + 1),
            "kl": total_kl_loss / (batch_idx + 1),
            "cls": (total_cls_loss / (batch_idx + 1)) if total_cls_loss > 0 else 0.0,
        })
        
        # Clear GPU cache periodically
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    avg_loss = total_loss / max(len(train_loader), 1)
    avg_recon_loss = total_recon_loss / max(len(train_loader), 1)
    avg_kl_loss = total_kl_loss / max(len(train_loader), 1)
    avg_cls_loss = total_cls_loss / max(len(train_loader), 1) if total_cls_loss > 0 else 0.0
    
    metrics = {
        "loss": avg_loss,
        "reconstruction_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
        "classification_loss": avg_cls_loss,
    }
    
    if total_cls_loss > 0:
        metrics["classification_loss"] = total_cls_loss / max(len(train_loader), 1)
    
    if len(all_predictions) > 0 and len(all_labels) > 0:
        cls_metrics = compute_classification_metrics(all_labels, all_predictions)
        metrics.update({
            "accuracy": cls_metrics["accuracy"],
            "balanced_accuracy": cls_metrics["balanced_accuracy"],
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
        })
        
    return metrics


def validate_epoch(model, val_loader, device, args, logger):
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_cls_loss = 0.0
    
    # For classification metrics
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", unit="batch")
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                X, time_tensor, y = batch
                X = X.to(device)
                time_tensor = time_tensor.to(device)
                y = y.to(device)
            else:
                continue
            
            # Forward pass
            output = model(X, y=y, time_tensor=time_tensor)
            
            loss = output["total_loss"]
            
            total_loss += loss.item()
            total_recon_loss += output["reconstruction_loss"].item()
            total_kl_loss += output["kl_loss"].item()
            total_cls_loss += output["classification_loss"].item() if output["classification_loss"] is not None else 0.0
            
            # Collect predictions for classification metrics
            if output["classification_logits"] is not None:
                predictions = torch.argmax(output["classification_logits"], dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(y.cpu().numpy())
            
            pbar.set_postfix({
                "loss": total_loss / (batch_idx + 1),
                "cls": total_cls_loss / (batch_idx + 1) if total_cls_loss > 0 else 0.0,
            })
    
    avg_loss = total_loss / max(len(val_loader), 1)
    avg_recon_loss = total_recon_loss / max(len(val_loader), 1)
    avg_kl_loss = total_kl_loss / max(len(val_loader), 1)
    avg_cls_loss = total_cls_loss / max(len(val_loader), 1) if total_cls_loss > 0 else 0.0

    metrics = {
        "loss": avg_loss,
        "reconstruction_loss": avg_recon_loss,
        "kl_loss": avg_kl_loss,
        "classification_loss": avg_cls_loss,
    }
    
    if total_cls_loss > 0:
        metrics["classification_loss"] = total_cls_loss / max(len(val_loader), 1)
    
    if len(all_predictions) > 0 and len(all_labels) > 0:
        cls_metrics = compute_classification_metrics(all_labels, all_predictions)
        metrics.update({
            "accuracy": cls_metrics["accuracy"],
            "balanced_accuracy": cls_metrics["balanced_accuracy"],
            "precision": cls_metrics["precision"],
            "recall": cls_metrics["recall"],
        })
        
        # Create confusion matrix plot
        cm_fig = create_confusion_matrix_plot(all_labels, all_predictions)
        metrics["confusion_matrix_fig"] = cm_fig
        
    return metrics


def main():
    """Main training loop."""
    args = get_args()
    
    set_seed(args.seed)
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    wandb_run = setup_wandb(args, logger)
    
    logger.info(f"Training VAE starting...")
    logger.info(f"Arguments: {args}")
    
    # Create dataloaders
    train_loader, val_loader, train_dataset = create_dataloaders(args, logger)
    
    # Create model and optimizer
    model, optimizer, config = create_model_and_optimizer(args, logger)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = output_dir / "best_model.pt"
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info("=" * 60)
        logger.info(f"Epoch {epoch}/{args.num_epochs}")
        logger.info("=" * 60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, args, logger)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, args.device, args, logger)
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.6f}")
        if train_metrics.get('classification_loss', 0) > 0:
            logger.info(f"  - Classification Loss: {train_metrics['classification_loss']:.6f}")
        if train_metrics.get('accuracy') is not None:
            logger.info(f"  - Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"  - Balanced Accuracy: {train_metrics['balanced_accuracy']:.4f}")
            logger.info(f"  - Precision: {train_metrics['precision']:.4f}")
            logger.info(f"  - Recall: {train_metrics['recall']:.4f}")
            
        logger.info(f"Val Loss: {val_metrics['loss']:.6f}")
        if val_metrics.get('classification_loss', 0) > 0:
            logger.info(f"  - Classification Loss: {val_metrics['classification_loss']:.6f}")
        if val_metrics.get('accuracy') is not None:
            logger.info(f"  - Accuracy: {val_metrics['accuracy']:.4f}")
            logger.info(f"  - Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
            logger.info(f"  - Precision: {val_metrics['precision']:.4f}")
            logger.info(f"  - Recall: {val_metrics['recall']:.4f}")
        
        if wandb_run:
            wandb_log_dict = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_reconstruction_loss": train_metrics["reconstruction_loss"],
                "train_kl_loss": train_metrics["kl_loss"],
                "val_loss": val_metrics["loss"],
                "val_reconstruction_loss": val_metrics["reconstruction_loss"],
                "val_kl_loss": val_metrics["kl_loss"],
            }
            
            if train_metrics.get('classification_loss', 0) > 0:
                wandb_log_dict["train_classification_loss"] = train_metrics["classification_loss"]
                wandb_log_dict["val_classification_loss"] = val_metrics["classification_loss"]
            
            if train_metrics.get('accuracy') is not None:
                wandb_log_dict.update({
                    "train_accuracy": train_metrics["accuracy"],
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_precision": train_metrics["precision"],
                    "train_recall": train_metrics["recall"]
                })
            
            if val_metrics.get('accuracy') is not None:
                wandb_log_dict.update({
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"]
                })
                
                if val_metrics.get('confusion_matrix_fig') is not None:
                    wandb_log_dict["val_confusion_matrix"] = wandb.Image(val_metrics["confusion_matrix_fig"])
            
            wandb_run.log(wandb_log_dict)
            
            if val_metrics.get('confusion_matrix_fig') is not None:
                plt.close(val_metrics["confusion_matrix_fig"])
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            logger.info(f"Saving best model with val loss: {best_val_loss:.6f}")
            torch.save(model.state_dict(), best_model_path)
        
        # Save checkpoint
        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.to_dict(),
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    if wandb_run:
        wandb_run.finish()
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
