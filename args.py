"""
Argument parser for VQ-VAE training script.
"""

import argparse
from pathlib import Path


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VQ-VAE model on AMEX time-series data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/scratch/s25090/Amex_data"),
        help="Path to the data directory containing train_data.csv and train_labels.csv",
    )
    data_group.add_argument(
        "--train-data",
        type=Path,
        default=Path("/scratch/s25090/Amex_data/train_data.csv"),
        help="Path to training data CSV file",
    )
    data_group.add_argument(
        "--train-labels",
        type=Path,
        default=Path("/scratch/s25090/Amex_data/train_labels.csv"),
        help="Path to training labels CSV file",
    )
    data_group.add_argument(
        "--db-path",
        type=Path,
        default=Path("/scratch/s25090/Amex_data/amex.db"),
        help="Path to SQLite database with time-series data",
    )
    data_group.add_argument(
        "--max-seq-len",
        type=int,
        default=13,
        help="Maximum sequence length for time-series samples",
    )
    
    # Model arguments - Patch Embedding
    model_group = parser.add_argument_group("Model - Patch Embedding")
    model_group.add_argument(
        "--input-dim",
        type=int,
        default=294,
        help="Input feature dimension (raw time-series features)",
    )
    model_group.add_argument(
        "--patch-size",
        type=int,
        default=2,
        help="Patch kernel size for Conv1d embedding",
    )
    model_group.add_argument(
        "--patch-stride",
        type=int,
        default=2,
        help="Patch stride for Conv1d embedding",
    )
    model_group.add_argument(
        "--patch-embed-dim",
        type=int,
        default=256,
        help="Output dimension of patch embedding",
    )
    
    # Model arguments - Vector Quantization
    vq_group = parser.add_argument_group("Model - Vector Quantization")
    vq_group.add_argument(
        "--num-embeddings",
        type=int,
        default=256,
        help="Number of codebook embeddings (VQ size)",
    )
    vq_group.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Dimension of VQ embeddings (should match hidden_dim)",
    )
    vq_group.add_argument(
        "--commitment-cost",
        type=float,
        default=0.25,
        help="Commitment cost for VQ loss",
    )
    
    # Model arguments - Transformer (Unified)
    trans_group = parser.add_argument_group("Model - Transformer (Unified for Encoder & Decoder)")
    trans_group.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Transformer hidden dimension (encoder and decoder)",
    )
    trans_group.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (encoder and decoder)",
    )
    trans_group.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    trans_group.add_argument(
        "--ff-multiplier",
        type=int,
        default=2,
        help="Feed-forward dimension multiplier (ff_dim = hidden_dim * ff_multiplier)",
    )
    trans_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for all layers",
    )
    
    # Model arguments - Classification
    class_group = parser.add_argument_group("Model - Classification")
    class_group.add_argument(
        "--use-class-token",
        action="store_true",
        help="Whether to use a classification token in the encoder",
    )
    class_group.add_argument(
        "--class-proj-dim",
        type=int,
        default=1,
        help="Dimension of the classification projection layer",
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    train_group.add_argument(
        "--val-batch-size",
        type=int,
        default=64,
        help="Batch size for validation",
    )
    train_group.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer",
    )
    train_group.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer",
    )
    train_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Loss weights
    loss_group = parser.add_argument_group("Loss Weights")
    loss_group.add_argument(
        "--reconstruction-loss-weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction loss",
    )
    loss_group.add_argument(
        "--commitment-loss-weight",
        type=float,
        default=0.25,
        help="Weight for commitment loss",
    )
    loss_group.add_argument(
        "--classification-loss-weight",
        type=float,
        default=0.5,
        help="Weight for classification loss (if classification head is used)",
    )
    
    # Logging and checkpointing arguments
    logging_group = parser.add_argument_group("Logging and Checkpointing")
    logging_group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./checkpoints"),
        help="Directory to save model checkpoints and logs",
    )
    logging_group.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    logging_group.add_argument(
        "--log-freq",
        type=int,
        default=10,
        help="Log metrics every N batches",
    )
    logging_group.add_argument(
        "--save-best",
        action="store_true",
        help="Save best model based on validation loss",
    )
    
    # Device and precision arguments
    device_group = parser.add_argument_group("Device and Precision")
    device_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    device_group.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training (FP16)",
    )
    
    # Weights & Biases logging arguments
    wandb_group = parser.add_argument_group("Weights & Biases Logging")
    wandb_group.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    wandb_group.add_argument(
        "--wandb-project",
        type=str,
        default="vq-vae-amex",
        help="Weights & Biases project name",
    )
    wandb_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (team/user) name",
    )
    wandb_group.add_argument(
        "--wandb-notes",
        type=str,
        default="",
        help="Notes for the Weights & Biases run",
    )
    wandb_group.add_argument(
        "--wandb-tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags for the Weights & Biases run",
    )
    wandb_group.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Run name for Weights & Biases (auto-generated if not provided)",
    )
    wandb_group.add_argument(
        "--wandb-save-dir",
        type=Path,
        default=None,
        help="Directory to save Weights & Biases data",
    )
    
    # Debugging arguments
    debug_group = parser.add_argument_group("Debugging")
    debug_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with smaller dataset",
    )
    debug_group.add_argument(
        "--debug-size",
        type=int,
        default=100,
        help="Dataset size when in debug mode",
    )
    debug_group.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    
    return parser.parse_args()


def create_config_from_args(args):
    """Convert command-line arguments to VQVAEConfig object."""
    from vq_vae.config import VQVAEConfig
    
    config = VQVAEConfig(
        # Patch Embedding
        input_dim=args.input_dim,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        patch_embed_dim=args.patch_embed_dim,
        # Vector Quantization
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        commitment_cost=args.commitment_cost,
        # Transformer (Unified)
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        # Classification
        use_class_token=args.use_class_token,
        class_proj_dim=args.class_proj_dim,
        # Loss Weights
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        commitment_loss_weight=args.commitment_loss_weight,
        classification_loss_weight=args.classification_loss_weight,
        # Training
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        # Device
        device=args.device,
        seed=args.seed,
    )
    return config


if __name__ == "__main__":
    args = get_args()
    print(args)
