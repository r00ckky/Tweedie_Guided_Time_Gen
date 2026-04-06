"""
Example usage of VQ-VAE with Transformer.

This script demonstrates how to use the VQ-VAE model with different configurations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple

# Import VQ-VAE components
from vq_vae import VQ_VAE, VQVAEConfig, set_seed, print_model_info


class VQVAETrainer:
    """Trainer for VQ-VAE model."""
    
    def __init__(
        self,
        model: VQ_VAE,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        """
        Initialize trainer.
        
        Args:
            model: VQ-VAE model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """
        Train one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_perplexity = 0
        
        for batch_idx, (batch,) in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            output = self.model(batch)
            loss = output["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += output["reconstruction_loss"].item()
            total_vq_loss += output["vq_loss"].item()
            total_perplexity += output["perplexity"].item()
        
        num_batches = len(train_loader)
        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "vq_loss": total_vq_loss / num_batches,
            "perplexity": total_perplexity / num_batches,
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        total_perplexity = 0
        
        for batch_idx, (batch,) in enumerate(val_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            output = self.model(batch)
            
            # Accumulate metrics
            total_loss += output["total_loss"].item()
            total_recon_loss += output["reconstruction_loss"].item()
            total_vq_loss += output["vq_loss"].item()
            total_perplexity += output["perplexity"].item()
        
        num_batches = len(val_loader)
        return {
            "loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "vq_loss": total_vq_loss / num_batches,
            "perplexity": total_perplexity / num_batches,
        }


def create_dummy_data(
    num_samples: int = 100,
    seq_len: int = 64,
    input_dim: int = 512,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dummy data for testing.
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        input_dim: Input dimension
        batch_size: Batch size
    
    Returns:
        Training and validation data loaders
    """
    # Create random data
    X = torch.randn(num_samples, seq_len, input_dim)
    
    # Split into train and val
    split = int(0.8 * num_samples)
    X_train = X[:split]
    X_val = X[split:]
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def example_basic_usage():
    """Example: Basic VQ-VAE usage."""
    print("\n" + "="*60)
    print("Example 1: Basic VQ-VAE Usage")
    print("="*60)
    
    # Set seed
    set_seed(42)
    
    # Create config
    config = VQVAEConfig(
        input_dim=512,
        output_dim=512,
        embedding_dim=64,
        num_embeddings=512,
        encoder_num_layers=2,
        decoder_num_layers=2,
        encoder_num_heads=8,
        decoder_num_heads=8,
    )
    
    # Create model
    model = VQ_VAE(config)
    print(model.summary())
    
    # Create dummy data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    seq_len = 64
    
    x = torch.randn(batch_size, seq_len, config.input_dim).to(device)
    model = model.to(device)
    
    # Forward pass
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {output['reconstruction'].shape}")
    print(f"Latent shape: {output['z'].shape}")
    print(f"Quantized shape: {output['z_q'].shape}")
    print(f"Total loss: {output['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {output['reconstruction_loss'].item():.4f}")
    print(f"VQ loss: {output['vq_loss'].item():.4f}")
    print(f"Perplexity: {output['perplexity'].item():.4f}")


def example_training():
    """Example: Training VQ-VAE."""
    print("\n" + "="*60)
    print("Example 2: Training VQ-VAE")
    print("="*60)
    
    # Set seed
    set_seed(42)
    
    # Configuration
    config = VQVAEConfig(
        input_dim=512,
        output_dim=512,
        embedding_dim=64,
        num_embeddings=256,
        encoder_num_layers=2,
        decoder_num_layers=2,
        encoder_hidden_dim=128,
        decoder_hidden_dim=128,
        encoder_num_heads=4,
        decoder_num_heads=4,
        encoder_ff_dim=256,
        decoder_ff_dim=256,
    )
    
    # Create model and trainer
    model = VQ_VAE(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = VQVAETrainer(model, device=device, learning_rate=1e-3)
    
    # Create data
    train_loader, val_loader = create_dummy_data(
        num_samples=200,
        seq_len=32,
        input_dim=config.input_dim,
        batch_size=32,
    )
    
    # Training loop
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs on device: {device}")
    
    for epoch in range(num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"VQ: {train_metrics['vq_loss']:.4f}, "
              f"Perplexity: {train_metrics['perplexity']:.2f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Recon: {val_metrics['recon_loss']:.4f}, "
              f"VQ: {val_metrics['vq_loss']:.4f}, "
              f"Perplexity: {val_metrics['perplexity']:.2f}")


def example_custom_config():
    """Example: Custom configuration."""
    print("\n" + "="*60)
    print("Example 3: Custom Configuration")
    print("="*60)
    
    # Create custom config
    config = VQVAEConfig(
        # Input/output
        input_dim=256,
        output_dim=256,
        
        # Quantizer
        num_embeddings=128,
        embedding_dim=32,
        commitment_cost=0.5,
        
        # Encoder
        encoder_hidden_dim=96,
        encoder_num_layers=3,
        encoder_num_heads=4,
        encoder_ff_dim=192,
        encoder_dropout=0.2,
        
        # Decoder
        decoder_hidden_dim=96,
        decoder_num_layers=3,
        decoder_num_heads=4,
        decoder_ff_dim=192,
        decoder_dropout=0.2,
        
        # Training
        learning_rate=5e-4,
        weight_decay=1e-5,
    )
    
    print(config)
    
    # Create model
    model = VQ_VAE(config)
    total_params = model.get_total_parameters_count()
    trainable_params = model.get_trainable_parameters_count()
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def example_inference():
    """Example: Inference and reconstruction."""
    print("\n" + "="*60)
    print("Example 4: Inference and Reconstruction")
    print("="*60)
    
    set_seed(42)
    
    # Create model
    config = VQVAEConfig(
        input_dim=128,
        output_dim=128,
        embedding_dim=16,
        num_embeddings=64,
    )
    model = VQ_VAE(config)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    with torch.no_grad():
        # Sample input
        x = torch.randn(4, 16, config.input_dim).to(device)
        
        # Forward pass
        output = model(x)
        
        # Get indices
        indices = output["indices"]
        
        # Reconstruct from indices
        reconstruction_from_indices = model.reconstruct_from_indices(indices)
        
        print(f"Original shape: {x.shape}")
        print(f"Reconstruction from model: {output['reconstruction'].shape}")
        print(f"Reconstruction from indices: {reconstruction_from_indices.shape}")
        print(f"Indices shape: {indices.shape}")
        print(f"Indices range: [{indices.min().item()}, {indices.max().item()}]")


if __name__ == "__main__":
    print("VQ-VAE with Transformer - Examples")
    print("====================================")
    
    # Run examples
    example_basic_usage()
    example_custom_config()
    example_inference()
    example_training()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)
