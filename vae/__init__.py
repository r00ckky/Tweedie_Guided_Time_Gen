"""
VAE (Variational Autoencoder) module for time-series data.
"""

from .config import VAEConfig
from .vae import VAE, PatchEmbedding
from .encoder import TransformerEncoder, TransformerEncoderBlock
from .decoder import TransformerDecoder, TransformerDecoderBlock
from .utils import (
    kl_weight_schedule,
    compute_reconstruction_loss,
    sample_from_latent,
    interpolate_latent,
    get_reconstruction_quality_metrics,
)

__all__ = [
    "VAE",
    "VAEConfig",
    "PatchEmbedding",
    "TransformerEncoder",
    "TransformerEncoderBlock",
    "TransformerDecoder",
    "TransformerDecoderBlock",
    "kl_weight_schedule",
    "compute_reconstruction_loss",
    "sample_from_latent",
    "interpolate_latent",
    "get_reconstruction_quality_metrics",
]
