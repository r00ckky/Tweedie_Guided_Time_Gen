import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    """Standard Sinusoidal Time Embeddings for Diffusion Models."""
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

class SeqResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.net(self.norm2(x).transpose(1, 2)).transpose(1, 2)
        return x

class TabularDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks=3):
        super().__init__()
        time_dim = hidden_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([SeqResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        t_emb = t_emb.unsqueeze(1)
        x = self.input_proj(x) + t_emb
        for block in self.res_blocks:
            x = block(x)
            
        return self.output_proj(x)