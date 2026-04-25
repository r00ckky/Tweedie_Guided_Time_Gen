"""
Transformer-based Encoder for VAE.
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple


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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 1. Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_output, _ = self.attention(norm_x, norm_x, norm_x, attn_mask=mask)
        # Dropout applied to branch output, then added to clean identity path
        x = x + self.attn_out_dropout(attn_output)
        
        # 2. Pre-LN Feed-Forward
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x) 
        # Dropout is already handled by the last layer of self.ffn
        x = x + ffn_output
        
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for VAE.
    Encodes input sequences to latent representations (mu and log_var).
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension of transformer
        latent_dim: Output latent dimension
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
        latent_dim: int,
        class_func: str,
        class_token: bool = True,
        class_proj_dim: Optional[int] = 1,
        num_layers: int = 2,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-6,
        koleo_penalty_weight: Optional[float] = None,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.class_token = class_token
        self.class_proj_dim = class_proj_dim
        self.class_func = class_func
        self.koleo_penalty_weight = koleo_penalty_weight
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Optional classification token
        if self.class_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        
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
        
        # Output layer norm
        self.output_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        
        # Mean and log variance projections for VAE
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize mu and logvar projections
        nn.init.xavier_uniform_(self.mu_proj.weight)
        nn.init.xavier_uniform_(self.logvar_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.logvar_proj.bias)
        
        if self.class_func == 'entropy':
            self.class_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, class_proj_dim)
            ) if class_token and class_proj_dim is not None else None
        elif self.class_func == 'knn':
            num_class = class_proj_dim + 1 if class_proj_dim == 1 else class_proj_dim
            self.class_proj = nn.Parameter(torch.empty(num_class, hidden_dim))
            nn.init.xavier_normal_(self.class_proj)
        else:
            self.class_proj = None
            
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
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        time_tensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor for classification loss
            mask: Optional attention mask
            time_tensor: Optional temporal information
        
        Returns:
            A tuple of (mu, logvar, classification_logits, classification_loss)
            - mu: Mean of latent distribution (batch_size, latent_dim)
            - logvar: Log variance of latent distribution (batch_size, latent_dim)
            - classification_logits: Classification logits or None
            - classification_loss: Classification loss or None
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)
        
        # Add classification token if enabled
        if self.class_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer blocks with a residual connection around each block
        for block in self.transformer_blocks:
            x = F.layer_norm(x, self.output_norm.normalized_shape, eps=self.output_norm.eps)
            x = block(x, mask=mask)
        
        # Output layer norm
        x = self.output_norm(x)
        
        # Extract classification token representation if it exists
        if self.class_token:
            cls_rep = x[:, 0, :]  # (batch_size, hidden_dim)
            x = x[:, 1:, :]  # Remove cls token for VAE encoding
        else:
            cls_rep = x.mean(dim=1)  # Use average pooling
        
        # Project to mu and logvar
        mu = self.mu_proj(x)  # (batch_size, seq_len, latent_dim)
        logvar = self.logvar_proj(x)  # (batch_size, seq_len, latent_dim)
        
        # Aggregate to sequence level (mean over time dimension)
        mu = mu.mean(dim=1)  # (batch_size, latent_dim)
        logvar = logvar.mean(dim=1)  # (batch_size, latent_dim)
        
        # Classification logits and loss
        classification_logits = None
        classification_loss = None
        
        if hasattr(self, 'class_proj') and self.class_proj is not None:
            if self.class_func == 'entropy':
                classification_logits = self.class_proj(cls_rep)  # (batch_size, class_proj_dim)
                
                if y is not None:
                    if classification_logits.shape[-1] == 1:
                        cls_logits_squeezed = classification_logits.squeeze(-1) 
                        loss_fn = nn.BCEWithLogitsLoss()
                        classification_loss = loss_fn(cls_logits_squeezed, y.float())
                    else:
                        loss_fn = nn.CrossEntropyLoss()
                        classification_loss = loss_fn(classification_logits, y.long())
                        
            elif self.class_func == 'knn':
                dist_sq = torch.cdist(cls_rep, self.class_proj, p=2.0) ** 2
                classification_logits = -dist_sq
                
                if y is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    classification_loss_val = loss_fn(classification_logits, y.long())
                    koleo_penalty = 0.0
                    if self.class_proj.size(0) > 1:
                        koleo_penalty = self.compute_koleo_loss(self.class_proj)
                    
                    koleo_weight = self.koleo_penalty_weight if self.koleo_penalty_weight is not None else 0
                    classification_loss = classification_loss_val + (koleo_weight * koleo_penalty)
        
        return mu, logvar, classification_logits, classification_loss
