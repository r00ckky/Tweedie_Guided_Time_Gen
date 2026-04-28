import math
import torch
import torch.nn as nn

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

class ConvNeXt1DBlock(nn.Module):
    """
    ConvNeXt-inspired 1D block for time-series diffusion.
    Features large-kernel depthwise convs and FiLM time conditioning.
    """
    def __init__(self, dim, time_dim, drop_path=0.1):
        super().__init__()
        # Depthwise Conv: kernel=7 gives massive receptive field for seq_len=14
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # LayerNorm applied on the channels (requires sequence transposition)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Inverted Bottleneck (Expand by 4x)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # FiLM Time Conditioning: Predicts scale and shift parameters
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_emb):
        # x shape: (Batch, Channels, Seq)
        residual = x
        
        # 1. Depthwise Convolution
        x = self.dwconv(x)
        
        # 2. Transpose for LayerNorm and Linear layers -> (Batch, Seq, Channels)
        x = x.transpose(1, 2)
        x = self.norm(x)
        
        # 3. Apply FiLM Time Conditioning
        # Split the time MLP output into scale (gamma) and shift (beta)
        time_weight, time_bias = self.time_mlp(t_emb).chunk(2, dim=-1)
        x = x * (time_weight.unsqueeze(1) + 1.0) + time_bias.unsqueeze(1)
        
        # 4. Inverted Bottleneck
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # 5. Transpose back -> (Batch, Channels, Seq)
        x = x.transpose(1, 2)
        
        # 6. Residual connection with Dropout/DropPath
        x = residual + self.drop_path(x)
        return x

class TabularDenoiser(nn.Module):
    """
    Airawat-optimized Denoiser. 
    Replaces the standard ResNet with a ConvNeXt-1D backbone.
    """
    def __init__(self, input_dim, hidden_dim=512, num_blocks=6, dropout_rate=0.1):
        super().__init__()
        
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            ConvNeXt1DBlock(hidden_dim, time_dim, drop_path=dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Final norm before output projection
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x, t):
        """
        x: (Batch, Seq_Len, Input_Dim)
        t: (Batch,)
        """
        # Transpose to (Batch, Channels, Seq_Len)
        x = x.transpose(1, 2) 
        
        # Get Time Embeddings
        t_emb = self.time_mlp(t)
        
        # Initial Projection
        x = self.input_proj(x)
        
        # Pass through ConvNeXt blocks with FiLM conditioning
        for block in self.blocks:
            x = block(x, t_emb)
            
        # Final Norm and Projection
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        
        x = self.output_proj(x)
        
        # Transpose back to (Batch, Seq_Len, Input_Dim)
        return x.transpose(1, 2)