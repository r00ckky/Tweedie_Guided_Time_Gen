"""
VQ-VAE with Transformer backbone.

This package provides a modular implementation of Vector Quantized Variational
AutoEncoder (VQ-VAE) using Transformer architecture.

Main Components:
- VQ_VAE: Main model combining encoder, quantizer, and decoder
- VQVAEConfig: Configuration class for all hyperparameters
- TransformerEncoder: Encoder with transformer blocks
- TransformerDecoder: Decoder with transformer blocks
- VectorQuantizer: Vector quantization module with EMA updates
"""

from .config import VQVAEConfig
from .vq_vae import VQ_VAE
from .encoder import TransformerEncoder, TransformerEncoderBlock
from .decoder import TransformerDecoder, TransformerDecoderBlock
from .vector_quantizer import VectorQuantizer
from .codebook_tracker import CodebookTracker
from .utils import (
    set_seed,
    count_parameters,
    count_trainable_parameters,
    create_causal_mask,
    create_padding_mask,
    adjust_learning_rate,
    get_device,
    save_checkpoint,
    load_checkpoint,
    EarlyStopping,
    print_model_info,
)

__version__ = "1.0.0"

__all__ = [
    # Config
    "VQVAEConfig",
    "TransformerConfig",
    # Main model
    "VQ_VAE",
    # Components
    "TransformerEncoder",
    "TransformerEncoderBlock",
    "TransformerDecoder",
    "TransformerDecoderBlock",
    "VectorQuantizer",
    "CodebookTracker",
    # Utilities
    "set_seed",
    "count_parameters",
    "count_trainable_parameters",
    "create_causal_mask",
    "create_padding_mask",
    "adjust_learning_rate",
    "get_device",
    "save_checkpoint",
    "load_checkpoint",
    "EarlyStopping",
    "print_model_info",
]
