"""
Utility functions for VAE training and inference.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np


def kl_weight_schedule(epoch: int, start_epoch: int = 0, max_epochs: int = 50, schedule_type: str = "linear") -> float:
    """
    Compute KL weight for beta-VAE annealing.
    
    Args:
        epoch: Current epoch
        start_epoch: Epoch to start KL annealing
        max_epochs: Total epochs for full KL weight
        schedule_type: Type of schedule ('linear', 'sigmoid', 'constant')
    
    Returns:
        KL weight (between 0 and 1)
    """
    if epoch < start_epoch:
        return 0.0
    
    if schedule_type == "linear":
        progress = min((epoch - start_epoch) / (max_epochs - start_epoch), 1.0)
        return progress
    elif schedule_type == "sigmoid":
        progress = (epoch - start_epoch) / (max_epochs - start_epoch)
        return 1.0 / (1.0 + np.exp(-10 * (progress - 0.5)))
    else:  # constant
        return 1.0


def compute_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute reconstruction loss.
    
    Args:
        reconstruction: Reconstructed tensor
        target: Target tensor
        loss_type: Type of loss ('mse', 'l1', 'huber')
        reduction: Reduction type ('mean', 'sum')
    
    Returns:
        Loss value
    """
    if loss_type == "mse":
        return nn.functional.mse_loss(reconstruction, target, reduction=reduction)
    elif loss_type == "l1":
        return nn.functional.l1_loss(reconstruction, target, reduction=reduction)
    elif loss_type == "huber":
        return nn.functional.huber_loss(reconstruction, target, reduction=reduction)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def sample_from_latent(
    vae: nn.Module,
    num_samples: int,
    device: torch.device,
    seq_len: int = 13,
) -> torch.Tensor:
    """
    Generate samples from the latent space.
    
    Args:
        vae: VAE model
        num_samples: Number of samples to generate
        device: Device to generate samples on
        seq_len: Sequence length for decoder
    
    Returns:
        Generated samples of shape (num_samples, seq_len, input_dim)
    """
    vae.eval()
    with torch.no_grad():
        # Sample from standard normal
        z = torch.randn(num_samples, vae.config.latent_dim, device=device)
        # Decode
        samples = vae.decode(z)
    return samples


def interpolate_latent(
    vae: nn.Module,
    z1: torch.Tensor,
    z2: torch.Tensor,
    num_steps: int = 10,
) -> torch.Tensor:
    """
    Interpolate between two latent codes.
    
    Args:
        vae: VAE model
        z1: First latent code
        z2: Second latent code
        num_steps: Number of interpolation steps
    
    Returns:
        Interpolated reconstructions
    """
    vae.eval()
    with torch.no_grad():
        alphas = torch.linspace(0, 1, num_steps, device=z1.device)
        interpolations = []
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            recon = vae.decode(z_interp.unsqueeze(0))
            interpolations.append(recon)
        return torch.cat(interpolations, dim=0)


def get_reconstruction_quality_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute reconstruction quality metrics.
    
    Args:
        reconstruction: Reconstructed tensor
        target: Target tensor
    
    Returns:
        Dictionary of metrics
    """
    mse = nn.functional.mse_loss(reconstruction, target).item()
    mae = nn.functional.l1_loss(reconstruction, target).item()
    
    # Compute correlation coefficient
    recon_flat = reconstruction.view(-1)
    target_flat = target.view(-1)
    
    mean_recon = recon_flat.mean()
    mean_target = target_flat.mean()
    
    cov = ((recon_flat - mean_recon) * (target_flat - mean_target)).mean()
    std_recon = (recon_flat - mean_recon).std()
    std_target = (target_flat - mean_target).std()
    
    corr = cov / (std_recon * std_target + 1e-8)
    corr = corr.item()
    
    return {
        "mse": mse,
        "mae": mae,
        "correlation": corr,
    }
