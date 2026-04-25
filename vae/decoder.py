"""
Transformer-based Decoder for VAE.
"""

import torch
import torch.nn as nn
from typing import Optional
from torch.nn import functional as F

class TransformerDecoderBlock(nn.Module):
    """Single Transformer decoder block with self-attention and feed-forward."""
    
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
        self.attn_out_dropout = nn.Dropout(dropout)
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
        # 1. Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_output, _ = self.self_attention(
            norm_x, norm_x, norm_x, attn_mask=self_attn_mask
        )
        # Drop the update, add to the clean identity path
        x = x + self.attn_out_dropout(attn_output)
        
        # 2. Pre-LN Feed-Forward
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        # Drop is applied inside the FFN sequential block
        x = x + ffn_output
        
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer-based decoder for VAE.
    Decodes latent representations back to output space.
    
    Args:
        latent_dim: Input latent dimension
        hidden_dim: Hidden dimension of transformer
        output_dim: Output feature dimension
        seq_len: Sequence length to decode to
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function name
        layer_norm_eps: Layer norm epsilon
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        seq_len: int = 13,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.activation = activation
        
        # Project latent to sequence of embeddings
        # We'll use a learnable position encoding and replicate latent across sequence
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
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
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Output projection to reconstructed features
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        z: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = z.size(0)
        
        # Project latent to hidden dimension
        x = self.latent_proj(z)  # (batch_size, hidden_dim)
        
        # Expand to sequence length
        x = x.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Apply transformer blocks (CLEAN LOOP)
        for block in self.transformer_blocks:
            x = block(x, self_attn_mask=self_attn_mask)
        
        # Output layer norm (Crucial for Pre-LN networks before final projection)
        x = self.output_norm(x)
        
        # Project to output dimension
        reconstruction = self.output_proj(x)  # (batch_size, seq_len, output_dim)
        
        return reconstruction
