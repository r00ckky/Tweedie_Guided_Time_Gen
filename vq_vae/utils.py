"""
Utility functions for VQ-VAE.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional
import random
import numpy as np


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal attention mask for decoder.
    
    Args:
        seq_len: Sequence length
        device: Device to create tensor on
    
    Returns:
        Causal mask of shape (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.
    
    Args:
        lengths: Tensor of sequence lengths, shape (batch_size,)
        max_len: Maximum sequence length (default: max of lengths)
        device: Device to create tensor on
    
    Returns:
        Padding mask of shape (batch_size, max_len)
    """
    if device is None:
        device = lengths.device
    
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.shape[0]
    mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask


def adjust_learning_rate(
    optimizer,
    epoch: int,
    total_epochs: int,
    initial_lr: float,
    schedule: str = "cosine",
):
    """
    Adjust learning rate based on schedule.
    
    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch
        total_epochs: Total number of epochs
        initial_lr: Initial learning rate
        schedule: Learning rate schedule ("cosine", "linear", "step")
    """
    if schedule == "cosine":
        # Cosine annealing
        lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    elif schedule == "linear":
        # Linear decay
        lr = initial_lr * (1 - epoch / total_epochs)
    elif schedule == "step":
        # Step decay
        lr = initial_lr * (0.1 ** (epoch // (total_epochs // 3)))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get device.
    
    Args:
        device: Device string or None
    
    Returns:
        torch.device
    """
    if device is None or device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    save_path: str,
    **kwargs,
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
        save_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: nn.Module,
    optimizer,
    checkpoint_path: str,
    map_location: Optional[str] = None,
):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer
        checkpoint_path: Path to checkpoint
        map_location: Device to map to
    
    Returns:
        Dictionary with checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs with no improvement to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
    
    def __call__(self, loss: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            loss: Current loss value
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def print_model_info(model: nn.Module, config):
    """Print model information."""
    print(model.summary())
    print(f"Config: {config}")
