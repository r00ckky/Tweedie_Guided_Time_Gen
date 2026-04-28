"""
Argument parser for VQ-VAE training script.
"""

import argparse
from pathlib import Path


def get_base_parser():
    """Build the base argument parser for VQ-VAE."""
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
    data_group.add_argument(
        "--class_imbalance",
        action="store_true",
        help="Whether to use class imbalance handling (e.g., weighted sampling)",
    )
    data_group.add_argument(
        "--transformer_path",
        type=Path,
        default=Path("quantile_transformer.pkl"),
        help="Path to saved QuantileTransformer for data preprocessing",
    )
    data_group.add_argument(
        "--fill-dict-path",
        type=Path,
        default=Path("fill_dict.pkl"),
        help="Path to saved fill dictionary for handling missing values",
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
    
    # Model arguments - Latent Space (for VAE)
    latent_group = parser.add_argument_group("Model - Latent Space (VAE)")
    latent_group.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Dimension of latent space (for VAE)",
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
    class_group.add_argument(
        "--class_func",
        type=str,
        choices=['entropy', 'knn'],
        help="Classification methodology",
        default='knn',
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
        help="Weight for commitment loss (VQ-VAE)",
    )
    loss_group.add_argument(
        "--kl-loss-weight",
        type=float,
        default=1.0,
        help="Weight for KL divergence loss (VAE)",
    )
    loss_group.add_argument(
        "--classification-loss-weight",
        type=float,
        default=0.5,
        help="Weight for classification loss (if classification head is used)",
    )
    loss_group.add_argument(
        "--koleo_penalty_weight",
        type=float,
        default=0.01,
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
        default=100,
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
    
    return parser


def get_args():
    """Parse and return command-line arguments for VQ-VAE."""
    return get_base_parser().parse_args()


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
        koleo_penalty_weight=args.koleo_penalty_weight,
        # Transformer (Unified)
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        # Classification
        use_class_token=args.use_class_token,
        class_proj_dim=args.class_proj_dim,
        class_func=args.class_func,
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

def get_ncsn_parser():
    """Build the argument parser for NCSN."""
    parser = get_base_parser()
    parser.description = "Train Noise Conditional Score Network (NCSN) on AMEX time-series data"
    
    ncsn_group = parser.add_argument_group("Model - NCSN Specific")
    ncsn_group.add_argument(
        "--vq-vae-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained VQ-VAE .pt file for NCSN training",
    )
    ncsn_group.add_argument(
        "--denoiser-hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for the TabularDenoiser MLP",
    )
    ncsn_group.add_argument(
        "--num-scales",
        type=int,
        default=10,
        help="Number of noise scales for NCSN",
    )
    ncsn_group.add_argument(
        "--sigma-max", 
        type=float,
        default=1.0,
        help="Maximum noise scale (sigma_max) for NCSN",
    )
    ncsn_group.add_argument(
        "--sigma-min",
        type=float,
        default=0.01,
        help="Minimum noise scale (sigma_min) for NCSN",
    )
    ncsn_group.add_argument(
        "--ncsn_weights",
        type=str,
        default=None,
        help="Weights to NCSN"
    )
    ncsn_group.add_argument(
        "--ncsn_num_blocks",
        type=int,
        default=3,
        help="Number of SeqResBlock layers in the TabularDenoiser",
    )
    ncsn_group.add_argument(
        "--denoiser-model",
        type=str,
        default="dit",
        choices=["dit", "conv_next", "conv", "resnet"],
        help="Architecture for the NCSN denoiser",
    )
    return parser

def create_ncsn_config_from_args(args):
    from NCSN.config import NCSNConfig
    config = NCSNConfig(
        vq_vae_checkpoint=args.vq_vae_checkpoint,
        denoiser_hidden_dim=args.denoiser_hidden_dim,
        checkpoint_dir=str(args.output_dir),
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        num_scales=args.num_scales,
        ncsn_num_blocks=args.ncsn_num_blocks,
        denoiser_model=args.denoiser_model,
    )
    return config

def get_dit_parser():
    """Build the argument parser for Latent Diffusion / DiT."""
    parser = get_base_parser()
    parser.description = "Train Latent Diffusion Transformer (DiT) on AMEX time-series data"
    
    # Override defaults that make more sense for DiT
    parser.set_defaults(
        learning_rate=1e-4,
        output_dir=Path("./dit_output"),
        wandb_project="amex-latent-diffusion",
        num_epochs=100,
    )
    
    # DiT Architecture arguments
    dit_group = parser.add_argument_group("Model - DiT Architecture")
    dit_group.add_argument("--dit-hidden-dim", type=int, default=256, help="Internal transformer width H")
    dit_group.add_argument("--dit-cond-dim", type=int, default=256, help="Timestep/label conditioning width C")
    dit_group.add_argument("--dit-num-layers", type=int, default=6, help="Number of DiTBlock stacks")
    dit_group.add_argument("--dit-num-heads", type=int, default=8, help="Self-attention heads for DiT")
    dit_group.add_argument("--dit-ff-multiplier", type=int, default=4, help="FFN multiplier for DiT")
    dit_group.add_argument("--dit-dropout", type=float, default=0.1, help="Dropout rate for DiT")
    dit_group.add_argument("--dit-num-classes", type=int, default=2, help="Number of classes for conditioning")
    dit_group.add_argument("--cfg-dropout-prob", type=float, default=0.10, help="Probability to drop label for CFG")
    
    # Diffusion Schedule
    diff_group = parser.add_argument_group("Model - Diffusion Schedule")
    diff_group.add_argument("--timesteps", type=int, default=1000, help="Total diffusion steps")
    diff_group.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "linear"], help="Beta schedule")
    diff_group.add_argument("--beta-start", type=float, default=1e-4, help="Beta start (linear only)")
    diff_group.add_argument("--beta-end", type=float, default=0.02, help="Beta end (linear only)")
    
    # DiT Training
    train_dit_group = parser.add_argument_group("Training - DiT Specific")
    train_dit_group.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping threshold")
    train_dit_group.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs")
    train_dit_group.add_argument("--unfreeze-encoder", action="store_false", dest="freeze_encoder", help="Fine-tune encoder weights instead of freezing them")
    train_dit_group.add_argument("--vqvae-checkpoint", type=str, default=None, help="Path to pretrained VQ-VAE .pt file")
    train_dit_group.set_defaults(freeze_encoder=True)
    
    return parser


def get_dit_args():
    """Parse and return command-line arguments for Latent Diffusion."""
    return get_dit_parser().parse_args()


def create_dit_config_from_args(args):
    """Convert command-line arguments to DiTConfig object."""
    from latent_diffusion.config import DiTConfig
    
    return DiTConfig(
        hidden_dim=args.dit_hidden_dim,
        cond_dim=args.dit_cond_dim,
        num_layers=args.dit_num_layers,
        num_heads=args.dit_num_heads,
        ff_multiplier=args.dit_ff_multiplier,
        dropout=args.dit_dropout,
        num_classes=args.dit_num_classes,
        cfg_dropout_prob=args.cfg_dropout_prob,
        timesteps=args.timesteps,
        schedule=args.schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=args.warmup_epochs,
        freeze_encoder=args.freeze_encoder,
        vqvae_checkpoint=args.vqvae_checkpoint,
        seed=args.seed,
        device=args.device,
        output_dir=str(args.output_dir),
        log_freq=args.log_freq,
        checkpoint_freq=args.checkpoint_freq,
        save_best=args.save_best,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        debug=args.debug,
        debug_size=args.debug_size,
    )

if __name__ == "__main__":
    args = get_args()
    print(args)
