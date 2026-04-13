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
        class_func:str,
        class_token: bool = True,
        class_proj_dim: Optional[int] = 1,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-6,
        koleo_penalty_weight:Optional[float]=None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.class_func = class_func
        self.koleo_penalty_weight=koleo_penalty_weight
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
        if self.class_func=='entropy':
            self.class_proj = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, class_proj_dim)
            ) if class_token and class_proj_dim is not None else None

        elif self.class_func == 'knn':
            num_class = class_proj_dim + 1 if class_proj_dim == 1 else class_proj_dim
            self.class_proj = nn.Parameter(torch.empty(num_class, embedding_dim))
            nn.init.xavier_normal_(self.class_proj)
    
    def compute_koleo_loss(
            self,
            centers: torch.Tensor, 
            eps: float = 1e-8
        ) -> torch.Tensor:
        """
        Computes Kozachenko-Leonenko (KoLeo) loss to enforce uniformity among class centers.
        """
        # 1. L2 Normalize to prevent magnitude explosion
        norm_centers = F.normalize(centers, p=2, dim=-1)
        
        # 2. Compute pairwise Euclidean distances
        distances = torch.cdist(norm_centers, norm_centers, p=2.0)
        
        # 3. Clone the tensor before the in-place operation!
        # This saves the original 'distances' for the backward pass
        distances_cloned = distances.clone()
        
        # 4. Mask out self-distances
        distances_cloned.fill_diagonal_(float('inf'))
        
        # 5. Find the distance to the *nearest neighbor* for each center
        min_distances, _ = distances_cloned.min(dim=1)
        
        # 6. KoLeo is the negative mean log of nearest neighbor distances
        loss = -torch.mean(torch.log(min_distances + eps))
        
        return loss

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
        
        x = self.output_projection(x)
        
        if self.class_proj is not None and self.class_func=='entropy':
            cls_token_hidden = x[:, 0]
            z = x[:, 1:]

            cls_logits = self.class_proj(cls_token_hidden)  # (batch_size, class_proj_dim)
            cls_loss = None

            if y is not None:
                if cls_logits.shape[-1] == 1:
                    cls_logits_squeezed = cls_logits.squeeze(-1) 
                    loss_fn = nn.BCEWithLogitsLoss()
                    cls_loss = loss_fn(cls_logits_squeezed, y.float())
                else:
                    loss_fn = nn.CrossEntropyLoss()
                    cls_loss = loss_fn(cls_logits, y.long())
            
            return z, cls_logits, cls_loss
        
        elif self.class_func == 'knn':
            cls_tokens = x[:, 0]  
            z = x[:, 1:]
            dist_sq = torch.cdist(cls_tokens, self.class_proj, p=2.0) ** 2
            
            cls_logits = -dist_sq
            cls_loss = None
            
            if y is not None:
                loss_fn = nn.CrossEntropyLoss()
                classification_loss = loss_fn(cls_logits, y.long())
                koleo_penalty = 0.0
                if self.class_proj.size(0) > 1:
                    koleo_penalty = self.compute_koleo_loss(self.class_proj)
                
                koleo_weight = self.koleo_penalty_weight if self.koleo_penalty_weight is not None else 0
                cls_loss = classification_loss + (koleo_weight * koleo_penalty)
                    
            return z, cls_logits, cls_loss
        
        return x, None, None