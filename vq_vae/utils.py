"""
Utility functions for VQ-VAE.
"""

import torch
import torch.nn as nn
from typing import Callable, Optional
import random
import numpy as np
import traceback
import inspect
from contextlib import contextmanager


# ============================================================================
# NaN Detection Utilities
# ============================================================================

def check_nan(tensor: torch.Tensor, var_name: str = "tensor", raise_error: bool = True) -> bool:
    """
    Check if a tensor contains NaN values and raise an informative error.
    
    Args:
        tensor: Tensor to check for NaN
        var_name: Name of the variable (for error message)
        raise_error: If True, raise error when NaN found; if False, return True/False
    
    Returns:
        True if NaN found, False otherwise (only if raise_error=False)
    
    Raises:
        ValueError: If NaN is found and raise_error=True
    """
    if not isinstance(tensor, torch.Tensor):
        return False
    
    has_nan = torch.isnan(tensor).any().item()
    
    if has_nan:
        if raise_error:
            # Get caller information
            frame = inspect.currentframe().f_back
            filename = frame.f_code.co_filename
            line_no = frame.f_lineno
            func_name = frame.f_code.co_name
            
            # Calculate NaN statistics
            nan_count = torch.isnan(tensor).sum().item()
            total_count = tensor.numel()
            nan_percentage = (nan_count / total_count) * 100
            
            error_msg = (
                f"\n{'='*70}\n"
                f"NaN DETECTED in variable: {var_name}\n"
                f"{'='*70}\n"
                f"Location: {filename}:{line_no} in function '{func_name}'\n"
                f"Tensor Shape: {tensor.shape}\n"
                f"Tensor Device: {tensor.device}\n"
                f"Tensor Dtype: {tensor.dtype}\n"
                f"NaN Count: {nan_count}/{total_count} ({nan_percentage:.2f}%)\n"
                f"Tensor Stats:\n"
                f"  Min (ignoring NaN): {torch.nanmin(tensor).item():.6e}\n"
                f"  Max (ignoring NaN): {torch.nanmax(tensor).item():.6e}\n"
                f"  Mean (ignoring NaN): {torch.nanmean(tensor).item():.6e}\n"
                f"  Std (ignoring NaN): {torch.nanstd(tensor).item():.6e}\n"
                f"{'='*70}\n"
                f"Call Stack:\n"
            )
            
            # Add call stack
            for line in traceback.format_stack()[:-1]:
                error_msg += line
            
            error_msg += f"{'='*70}\n"
            
            raise ValueError(error_msg)
        else:
            return True
    
    return False


def check_nan_dict(data_dict: dict, raise_error: bool = True) -> dict:
    """
    Check all tensors in a dictionary for NaN values.
    
    Args:
        data_dict: Dictionary potentially containing tensors
        raise_error: If True, raise error when NaN found
    
    Returns:
        Dictionary with NaN status for each tensor
    
    Raises:
        ValueError: If NaN is found and raise_error=True
    """
    nan_status = {}
    
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            try:
                check_nan(value, var_name=f"dict['{key}']", raise_error=raise_error)
                nan_status[key] = False
            except ValueError:
                nan_status[key] = True
                if raise_error:
                    raise
        elif isinstance(value, dict):
            nested_status = check_nan_dict(value, raise_error=raise_error)
            nan_status[key] = nested_status
    
    return nan_status


@contextmanager
def nan_detector(operation_name: str = "operation"):
    """
    Context manager for NaN detection around operations.
    
    Usage:
        with nan_detector("forward_pass"):
            output = model(input)
    
    Args:
        operation_name: Name of the operation being monitored
    
    Yields:
        None
    """
    try:
        yield
    except Exception as e:
        raise ValueError(
            f"\nNaN or error detected during '{operation_name}':\n{str(e)}"
        )


class NaNCheckHook:
    """
    Forward hook for detecting NaN in module outputs.
    
    Usage:
        hook = NaNCheckHook("encoder_output")
        module.register_forward_hook(hook)
    """
    
    def __init__(self, name: str = "module_output", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.nan_detected = False
    
    def __call__(self, module, input, output):
        if not self.enabled:
            return
        
        if isinstance(output, torch.Tensor):
            try:
                check_nan(output, var_name=self.name, raise_error=True)
            except ValueError as e:
                self.nan_detected = True
                raise
        elif isinstance(output, dict):
            try:
                check_nan_dict(output, raise_error=True)
            except ValueError as e:
                self.nan_detected = True
                raise
        elif isinstance(output, (tuple, list)):
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    try:
                        check_nan(item, var_name=f"{self.name}[{i}]", raise_error=True)
                    except ValueError as e:
                        self.nan_detected = True
                        raise


def register_nan_hooks(model: nn.Module, enabled: bool = True):
    """
    Register NaN detection hooks on all modules.
    
    Args:
        model: Model to register hooks on
        enabled: Whether to enable NaN checking
    
    Returns:
        List of hook handles
    """
    handles = []
    
    for name, module in model.named_modules():
        if name:  # Skip the root module
            hook = NaNCheckHook(name=f"{name}_output", enabled=enabled)
            handle = module.register_forward_hook(hook)
            handles.append(handle)
    
    return handles

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
