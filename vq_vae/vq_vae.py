"""
Main VQ-VAE module combining Encoder, Vector Quantizer, and Decoder.
Also includes PatchEmbedding for time-series data preprocessing.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .config import VQVAEConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .vector_quantizer import VectorQuantizer


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


class VQ_VAE(nn.Module):
    """
    Vector Quantized Variational AutoEncoder with Transformer backbone.
    
    Combines:
    - Patch Embedding: converts raw input to patch embeddings
    - Transformer Encoder: compresses patches to latent codes
    - Vector Quantizer: discretizes latent space
    - Transformer Decoder: reconstructs from quantized codes
    
    Args:
        config: VQVAEConfig instance containing all hyperparameters
    """
    
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        
        # Patch Embedding Layer
        self.patch_embedding = PatchEmbedding(
            input_dim=config.input_dim,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_embed_dim=config.patch_embed_dim,
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            input_dim=config.patch_embed_dim,  # Input is patch_embed_dim, not raw input_dim
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            activation="relu",
            class_token=config.use_class_token,
            class_proj_dim=config.class_proj_dim,
        )
        
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
            epsilon=config.epsilon,
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.patch_embed_dim,  # Decoder outputs patch_embed_dim
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            activation="relu",
        )
        
        # Reconstruction projection from patch embeddings back to input
        self.output_projection = nn.Linear(config.patch_embed_dim, config.input_dim)
    
    def encode(self, x: torch.Tensor, y: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None, time_tensor: Optional[torch.Tensor] = None) -> Dict:
        """
        Encode input to latent codes.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
            time_tensor: Optional tensor of shape (batch_size, seq_len) containing time differences

        Returns:
            Dictionary containing:
                - z: Latent embeddings before quantization
                - z_q: Quantized latent embeddings
                - indices: Codebook indices
                - loss: VQ loss
                - perplexity: Codebook perplexity
                - classification_logits: Classification logits, or None
                - classification_loss: Classification loss, or None
        """
        # Apply patch embedding first
        x_patch = self.patch_embedding(x)
        
        # Encode to latent space - always returns (z, cls_logits, cls_loss)
        z, cls_logits, cls_loss = self.encoder(x_patch, y, mask=mask, time_tensor=time_tensor)
        
        # Quantize
        quantization_output = self.vector_quantizer(z)
        z_q = quantization_output["quantized"]
        
        return {
            "z": z,
            "z_q": z_q,
            "indices": quantization_output["encoding_indices"],
            "loss": quantization_output["loss"],
            "perplexity": quantization_output["perplexity"],
            "encodings": quantization_output["encodings"],
            "classification_logits": cls_logits,
            "classification_loss": cls_loss,
            "x_patch": x_patch,
        }
    
    def decode(
        self,
        z_q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode quantized latent codes to reconstruction at patch level.
        
        Args:
            z_q: Quantized latent tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional attention mask
        
        Returns:
            Patch-level reconstructed tensor of shape (batch_size, num_patches, patch_embed_dim)
        """
        # Decode: (batch, num_patches, embedding_dim) -> (batch, num_patches, patch_embed_dim)
        patch_recon = self.decoder(z_q, self_attn_mask=mask)
        
        # Return patch-level reconstruction for loss computation
        return patch_recon
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        time_tensor: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass through VQ-VAE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
            time_tensor: Optional tensor of shape (batch_size, seq_len) containing time differences
        
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed output
                - z: Latent embeddings before quantization
                - z_q: Quantized latent embeddings
                - indices: Codebook indices
                - total_loss: Total training loss (reconstruction + VQ + classification)
                - reconstruction_loss: MSE reconstruction loss
                - vq_loss: Vector quantization loss
                - perplexity: Codebook perplexity
                - classification_logits: Classification logits, or None
                - classification_loss: Classification loss, or None
        """
        # Encode and quantize
        encode_output = self.encode(x, y, mask=mask, time_tensor=time_tensor)
        z_q = encode_output["z_q"]
        x_patch = encode_output["x_patch"]
        
        # Decode
        reconstruction = self.decode(z_q, mask=mask)
        
        # Compute reconstruction loss at patch level
        # Both reconstruction and x_patch are shape (batch, num_patches, patch_embed_dim)
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x_patch)
        vq_loss = encode_output["loss"]
        
        # Compute total loss including classification if available
        total_loss = (
            self.config.reconstruction_loss_weight * reconstruction_loss
            + self.config.commitment_loss_weight * vq_loss
        )
        
        # Add classification loss if it exists
        classification_loss = encode_output["classification_loss"]
        if classification_loss is not None:
            # Default weight for classification loss (1.0)
            classification_weight = getattr(self.config, 'classification_loss_weight', 1.0)
            total_loss = total_loss + classification_weight * classification_loss
        
        return {
            "reconstruction": reconstruction,
            "z": encode_output["z"],
            "z_q": z_q,
            "indices": encode_output["indices"],
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "perplexity": encode_output["perplexity"],
            "encodings": encode_output["encodings"],
            "classification_logits": encode_output["classification_logits"],
            "classification_loss": classification_loss,
        }
    
    def reconstruct_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct output directly from codebook indices.
        Useful for inference and generation.
        
        Args:
            indices: Tensor of codebook indices
        
        Returns:
            Reconstructed output
        """
        # Decode indices to embeddings
        z_q = self.vector_quantizer.decode(indices)
        # Decode embeddings to output
        reconstruction = self.decode(z_q)
        return reconstruction
    
    def get_codebook(self) -> torch.Tensor:
        """Get the current codebook embeddings."""
        return self.vector_quantizer.embeddings.weight.data.clone()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def freeze_quantizer(self):
        """Freeze quantizer parameters."""
        for param in self.vector_quantizer.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters_count(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> str:
        """Get model summary."""
        total_params = self.get_total_parameters_count()
        trainable_params = self.get_trainable_parameters_count()
        
        lines = [
            "\n" + "="*60,
            "VQ-VAE Model Summary (Streamlined Architecture)",
            "="*60,
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            f"Non-trainable Parameters: {total_params - trainable_params:,}",
            "-"*60,
            "Input and Patch Embedding:",
            f"  Input Dimension: {self.config.input_dim}",
            f"  Patch Size: {self.config.patch_size}, Stride: {self.config.patch_stride}",
            f"  Patch Embed Dimension: {self.config.patch_embed_dim}",
            "-"*60,
            "Vector Quantization:",
            f"  Codebook Size: {self.config.num_embeddings}",
            f"  Embedding Dimension: {self.config.embedding_dim}",
            f"  Commitment Cost: {self.config.commitment_cost}",
            "-"*60,
            "Transformer (Unified for Encoder & Decoder):",
            f"  Hidden Dimension: {self.config.hidden_dim}",
            f"  Number of Layers: {self.config.num_layers}",
            f"  Number of Heads: {self.config.num_heads}",
            f"  FF Dimension: {self.config.ff_dim} (hidden_dim × {self.config.ff_multiplier})",
            f"  Dropout: {self.config.dropout}",
            "-"*60,
            "Classification Head:" if self.config.use_class_token else "Classification Head: Disabled",
        ]
        
        if self.config.use_class_token:
            lines.append(f"  Projection Dimension: {self.config.class_proj_dim}")
        
        lines.extend([
            "-"*60,
            f"Loss Weights:",
            f"  Reconstruction: {self.config.reconstruction_loss_weight}",
            f"  VQ Commitment: {self.config.commitment_loss_weight}",
            f"  Classification: {self.config.classification_loss_weight}",
            "="*60 + "\n",
        ])
        return "\n".join(lines)
