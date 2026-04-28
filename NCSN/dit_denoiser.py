import math
import torch
import torch.nn as nn

def modulate(x, shift, scale):
    """Applies Adaptive LayerNorm conditioning."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class SinusoidalPosEmb(nn.Module):
    """Standard Sinusoidal Time Embeddings."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiTBlock1D(nn.Module):
    """
    A Diffusion Transformer block with adaLN-Zero conditioning.
    """
    def __init__(self, hidden_dim, num_heads, time_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=dropout_rate
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # adaLN-Zero conditioning: Predicts shift, scale, and gate for both Attention and MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * hidden_dim)
        )
        
        # Zero-initialize the modulation layer so the block acts as an identity function at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_emb):
        # 1. Predict conditioning parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        
        # 2. Conditioned Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # 3. Conditioned MLP
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x

class TabularDenoiser(nn.Module):
    """
    Tabular Diffusion Transformer (DiT-1D).
    """
    def __init__(self, input_dim, hidden_dim=512, num_blocks=6, dropout_rate=0.1):
        super().__init__()
        
        # Time Embedding
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # DiT Blocks
        num_heads = hidden_dim // 64  # Standard 64-dim per head
        self.blocks = nn.ModuleList([
            DiTBlock1D(hidden_dim, num_heads, time_dim, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Final LayerNorm with scaling
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 2 * hidden_dim)
        )
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        """
        x: (Batch, Seq_Len, Input_Dim)
        t: (Batch,)
        """
        # Get time embeddings
        t_emb = self.time_mlp(t)
        
        # Project input
        x = self.input_proj(x)
        
        # Pass through DiT blocks
        for block in self.blocks:
            x = block(x, t_emb)
            
        # Final modulation and projection
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift, scale)
        x = self.output_proj(x)
        
        return x