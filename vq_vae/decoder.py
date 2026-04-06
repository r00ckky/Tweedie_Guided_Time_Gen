"""
Transformer-based Decoder for VQ-VAE.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerDecoderBlock(nn.Module):
    """Single Transformer decoder block with self-attention, cross-attention and feed-forward."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Feed-forward network
        activation_fn = getattr(nn, activation.upper(), nn.ReLU)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            self_attn_mask: Optional self-attention mask
        
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(
            x, x, x, attn_mask=self_attn_mask
        )
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for VQ-VAE.
    Decodes latent representations back to output space.
    
    Args:
        embedding_dim: Input embedding dimension (from codebook)
        hidden_dim: Hidden dimension of transformer
        output_dim: Output feature dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function name
        layer_norm_eps: Layer norm epsilon
    """
    
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Project embedding to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Project to output dimension
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Decode latent representation to output.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            self_attn_mask: Optional self-attention mask
        
        Returns:
            Decoded tensor of shape (batch_size, seq_len, output_dim)
        """
        # Project embedding to hidden dimension
        x = self.input_projection(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, self_attn_mask=self_attn_mask)
        
        # Project to output dimension
        x = self.output_projection(x)
        
        return x
