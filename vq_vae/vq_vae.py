"""
Main VQ-VAE module combining Encoder, Vector Quantizer, and Decoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .config import VQVAEConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .vector_quantizer import VectorQuantizer


class VQ_VAE(nn.Module):
    """
    Vector Quantized Variational AutoEncoder with Transformer backbone.
    
    Combines:
    - Transformer Encoder: compresses input to latent codes
    - Vector Quantizer: discretizes latent space
    - Transformer Decoder: reconstructs from quantized codes
    
    Args:
        config: VQVAEConfig instance containing all hyperparameters
    """
    
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = TransformerEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.encoder_hidden_dim,
            embedding_dim=config.embedding_dim,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            ff_dim=config.encoder_ff_dim,
            dropout=config.encoder_dropout,
            activation=config.encoder_activation,
            class_token=config.encoder_class_token,
            class_proj_dim=config.encoder_class_proj_dim,
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
            hidden_dim=config.decoder_hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            ff_dim=config.decoder_ff_dim,
            dropout=config.decoder_dropout,
            activation=config.decoder_activation,
        )
    
    def encode(self, x: torch.Tensor, y: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None) -> Dict:
        """
        Encode input to latent codes.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
        
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
        # Encode to latent space - always returns (z, cls_logits, cls_loss)
        z, cls_logits, cls_loss = self.encoder(x, y, mask=mask)
        
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
        }
    
    def decode(
        self,
        z_q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode quantized latent codes to reconstruction.
        
        Args:
            z_q: Quantized latent tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional attention mask
        
        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, output_dim)
        """
        return self.decoder(z_q, self_attn_mask=mask)
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass through VQ-VAE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
        
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
        encode_output = self.encode(x, y, mask=mask)
        z_q = encode_output["z_q"]
        
        # Decode
        reconstruction = self.decode(z_q, mask=mask)
        
        # Compute losses
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x)
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
            "\n" + "="*50,
            "VQ-VAE Model Summary",
            "="*50,
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            f"Non-trainable Parameters: {total_params - trainable_params:,}",
            "-"*50,
            f"Input Dimension: {self.config.input_dim}",
            f"Output Dimension: {self.config.output_dim}",
            f"Embedding Dimension: {self.config.embedding_dim}",
            f"Codebook Size: {self.config.num_embeddings}",
            "-"*50,
            "Encoder:",
            f"  Hidden Dim: {self.config.encoder_hidden_dim}",
            f"  Num Layers: {self.config.encoder_num_layers}",
            f"  Num Heads: {self.config.encoder_num_heads}",
            "-"*50,
            "Decoder:",
            f"  Hidden Dim: {self.config.decoder_hidden_dim}",
            f"  Num Layers: {self.config.decoder_num_layers}",
            f"  Num Heads: {self.config.decoder_num_heads}",
            "-"*50,
            "Vector Quantizer:",
            f"  Num Embeddings: {self.config.num_embeddings}",
            f"  Embedding Dim: {self.config.embedding_dim}",
            f"  Commitment Cost: {self.config.commitment_cost}",
            "="*50 + "\n",
        ]
        return "\n".join(lines)
