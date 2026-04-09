"""
Transformer-based Encoder for VQ-VAE.
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block with self-attention and feed-forward."""
    
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
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor of same shape as input
        """
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for VQ-VAE.
    Encodes input sequences to latent representations.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension of transformer
        embedding_dim: Output embedding dimension (for codebook)
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        dropout: Dropout rate
        activation: Activation function name
        layer_norm_eps: Layer norm epsilon
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        class_token: bool = True,
        class_proj_dim: Optional[int] = 1,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Project input to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=layer_norm_eps),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if class_token else None
        self.time_proj = nn.Linear(1, hidden_dim)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Project to embedding dimension
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        # Classification head projects from embedding_dim (not hidden_dim) after output_projection
        self.class_proj = nn.Linear(embedding_dim, class_proj_dim) if class_token and class_proj_dim is not None else None
    
    def forward(
            self, 
            x: Tensor,
            y: Tensor,
            time_tensor:Tensor, 
            mask: Optional[Tensor] = None,
            
        ):
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor for classification (batch_size,)
            mask: Optional attention mask
            time_tensor: Optional tensor of shape (batch_size, seq_len, 1) for time differences
        
        Returns:
            z: Encoded tensor of shape (batch_size, seq_len, embedding_dim)
            cls_logits: Classification logits of shape (batch_size, class_proj_dim), or None
            cls_loss: Classification loss (None if y not provided or no classification head)
        """
        x = self.input_projection(x)
        time = torch.sin(2 * self.time_proj(time_tensor))
        batch_size = x.size(0)
        x += time[:, ::2]

        if self.cls_token is not None:
            cls_token_expanded = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token_expanded, x], dim=1)  # Add CLS token
        
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        
        # Project to embedding dimension
        x = self.output_projection(x)
        
        # Extract classification tokens if needed
        if self.class_proj is not None:
            cls_token_hidden = x[:, 0]  # Extract CLS token: (batch_size, embedding_dim)
            z = x[:, 1:]  # Keep all sequence tokens: (batch_size, seq_len, embedding_dim)
            
            # Apply classification projection
            cls_logits = self.class_proj(cls_token_hidden)  # (batch_size, class_proj_dim)
            cls_loss = None
            
            # Compute loss if targets provided
            if y is not None:
                if cls_logits.shape[-1] == 1:
                    # Binary classification
                    cls_logits_squeezed = cls_logits.squeeze(-1)  # (batch_size,)
                    cls_logits_prob = torch.sigmoid(cls_logits_squeezed)  # Apply sigmoid
                    loss_fn = nn.BCELoss()
                    cls_loss = loss_fn(cls_logits_prob, y.float())
                else:
                    # Multi-class classification
                    cls_logits_prob = F.softmax(cls_logits, dim=-1)  # Apply softmax
                    loss_fn = nn.CrossEntropyLoss()
                    cls_loss = loss_fn(cls_logits, y.long())
            
            return z, cls_logits, cls_loss
        
        # No classification head - return full sequence
        return x, None, None