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

class ConvResBlock(nn.Module):
    """Fully Convolutional 1D Residual Block for Time-Series."""
    def __init__(self, dim, dropout_rate=0.1):
        super().__init__()
        self.block = nn.Sequential(
            # kernel_size=3 allows the model to look at adjacent time steps (e.g., previous and next month)
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # Prevents memorizing the noise
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim)
        )

    def forward(self, x):
        # x is expected to be (Batch, Channels, Seq_Len)
        return x + self.block(x)

class TabularDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks=3, dropout_rate=0.1):
        super().__init__()
        
        # Time embedding generation (MLP is standard here since t is a single scalar per batch)
        time_dim = hidden_dim // 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 1x1 Convolution acts as a Linear projection across the feature dimension
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # Stack of Convolutional Residual Blocks
        self.res_blocks = nn.ModuleList([
            ConvResBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)
        ])
        
        # 1x1 Convolution to project back to the raw feature dimension
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x, t):
        """
        x shape: (Batch, Seq_Len, Input_Dim)
        t shape: (Batch,)
        """
        # PyTorch Conv1d expects (Batch, Channels, Seq_Len)
        # So we transpose exactly once at the beginning
        x = x.transpose(1, 2)
        
        # Generate time embeddings
        t_emb = self.time_mlp(t)          # Shape: (Batch, Hidden_Dim)
        
        # Reshape time embeddings to broadcast across the sequence length
        t_emb = t_emb.unsqueeze(-1)       # Shape: (Batch, Hidden_Dim, 1)
        
        # Project input and add time embeddings
        x = self.input_proj(x) + t_emb
        
        # Pass through the convolutional blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Project back to original feature dimension
        x = self.output_proj(x)
        
        # Transpose back to match the expected (Batch, Seq_Len, Input_Dim) format
        return x.transpose(1, 2)