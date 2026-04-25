"""
Main VAE module combining Encoder and Decoder.
Also includes PatchEmbedding for time-series data preprocessing.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .config import VAEConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer using 1D Convolution for time-series data.
    
    Converts raw time-series features to patch embeddings.
    Example: input (batch, seq_len, 294) -> output (batch, new_seq_len, patch_embed_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        patch_stride: int,
        patch_embed_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_embed_dim = patch_embed_dim
        
        # Conv1d: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        # then apply Conv1d to get (batch, patch_embed_dim, new_seq_len)
        # then transpose back to (batch, new_seq_len, patch_embed_dim)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=patch_embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patch embedding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Patch embeddings of shape (batch_size, new_seq_len, patch_embed_dim)
        """
        # x: (batch, seq_len, input_dim)
        # Permute to (batch, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply convolution: (batch, input_dim, seq_len) -> (batch, patch_embed_dim, new_seq_len)
        x = self.conv(x)
        
        # Permute back to (batch, new_seq_len, patch_embed_dim)
        x = x.permute(0, 2, 1)
        
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder with Transformer backbone.
    
    Combines:
    - Patch Embedding: converts raw input to patch embeddings
    - Transformer Encoder: compresses patches to latent distribution (mu, logvar)
    - Reparameterization: samples from latent distribution
    - Transformer Decoder: reconstructs from latent samples
    
    Args:
        config: VAEConfig instance containing all hyperparameters
    """
    
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        
        # Patch Embedding Layer
        self.patch_embedding = PatchEmbedding(
            input_dim=config.input_dim,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_embed_dim=config.patch_embed_dim,
        )
        
        # Calculate sequence length after patch embedding
        # seq_len = floor((seq_len - patch_size) / patch_stride) + 1
        # For example: (13 - 2) / 2 + 1 = 6.5 -> 6
        self.patch_seq_len = None  # Will be calculated on first forward pass
        
        # Encoder
        self.encoder = TransformerEncoder(
            input_dim=config.patch_embed_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
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
        
        # Decoder
        self.decoder = TransformerDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.input_dim,
            seq_len=config.max_seq_length,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            activation="relu",
        )
    
    def encode(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None, time_tensor: Optional[torch.Tensor] = None) -> Dict:
        """
        Encode input to latent distribution.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
            time_tensor: Optional tensor of shape (batch_size, seq_len) containing time differences

        Returns:
            Dictionary containing:
                - mu: Mean of latent distribution
                - logvar: Log variance of latent distribution
                - z: Sampled latent codes
                - classification_logits: Classification logits, or None
                - classification_loss: Classification loss, or None
        """
        # Apply patch embedding first
        x_patch = self.patch_embedding(x)
        
        # Store patch sequence length
        if self.patch_seq_len is None:
            self.patch_seq_len = x_patch.size(1)
        
        # Encode to latent distribution
        mu, logvar, cls_logits, cls_loss = self.encoder(x_patch, y, mask=mask, time_tensor=time_tensor)
        
        # Reparameterization trick: sample z from N(mu, exp(logvar))
        z = self.reparameterize(mu, logvar)
        
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "classification_logits": cls_logits,
            "classification_loss": cls_loss,
            "x_patch": x_patch,
        }
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, sigma^2) efficiently.
        
        Args:
            mu: Mean tensor of shape (batch_size, latent_dim)
            logvar: Log variance tensor of shape (batch_size, latent_dim)
        
        Returns:
            Sampled z of shape (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode latent codes to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            mask: Optional attention mask
        
        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, input_dim)
        """
        # Decode: (batch, latent_dim) -> (batch, seq_len, input_dim)
        reconstruction = self.decoder(z, self_attn_mask=mask)
        return reconstruction
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        time_tensor: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
            time_tensor: Optional tensor of shape (batch_size, seq_len) containing time differences
        
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed output
                - mu: Mean of latent distribution
                - logvar: Log variance of latent distribution
                - z: Sampled latent codes
                - total_loss: Total training loss (reconstruction + KL + classification)
                - reconstruction_loss: MSE reconstruction loss
                - kl_loss: KL divergence loss
                - classification_logits: Classification logits, or None
                - classification_loss: Classification loss, or None
        """
        # Encode and sample
        encode_output = self.encode(x, y, mask=mask, time_tensor=time_tensor)
        z = encode_output["z"]
        mu = encode_output["mu"]
        logvar = encode_output["logvar"]
        
        # Decode
        reconstruction = self.decode(z, mask=mask)
        
        # Compute reconstruction loss
        # Reconstruction loss between original input and decoded output
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x)
        
        # Compute KL divergence loss
        # KL(N(mu, logvar) || N(0, 1))
        kl_loss = self._kl_divergence_loss(mu, logvar)
        
        # Compute total loss
        total_loss = (
            self.config.reconstruction_loss_weight * reconstruction_loss
            + self.config.kl_loss_weight * kl_loss
        )
        
        # Add classification loss if it exists
        classification_loss = encode_output["classification_loss"]
        if classification_loss is not None:
            classification_weight = getattr(self.config, 'classification_loss_weight', 1.0)
            total_loss = total_loss + classification_weight * classification_loss
        
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "classification_logits": encode_output["classification_logits"],
            "classification_loss": classification_loss,
        }
    
    @staticmethod
    def _kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence: KL(N(mu, logvar) || N(0, 1))
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
        
        Returns:
            KL divergence loss (scalar)
        """
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # where sigma^2 = exp(logvar)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def freeze_patch_embedding(self):
        """Freeze patch embedding parameters."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
