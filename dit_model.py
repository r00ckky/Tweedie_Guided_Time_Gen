import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Timestep Embedding ───────────────────────────────────────────────────────


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps:  (B,)  integer or float timestep indices
        dim:        Embedding dimension (should be even)
        max_period: Controls the minimum frequency

    Returns:
        emb: (B, dim)

    Shape trace:
        timesteps : (B,)
        freqs     : (dim//2,)
        args      : (B, dim//2)
        emb       : (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )  # (dim//2,)

    args = timesteps[:, None].float() * freqs[None, :]  # (B, dim//2)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)

    if dim % 2:  # pad if odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)  # (B, dim)

    return emb


class TimestepEmbedMLP(nn.Module):
    """
    Maps integer diffusion timesteps → conditioning vector via sinusoidal + MLP.

    Architecture:
        t (B,) → sinusoidal(fourier_dim) → Linear(fourier_dim, hidden) → SiLU
                                         → Linear(hidden, cond_dim)

    Args:
        fourier_dim: Dimension for sinusoidal embedding
        cond_dim:    Output conditioning dimension
    """

    def __init__(self, fourier_dim: int, cond_dim: int):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timestep tensor

        Returns:
            t_emb: (B, cond_dim)

        Shape trace:
            t      : (B,)
            sinusoid: (B, fourier_dim)
            t_emb  : (B, cond_dim)
        """
        sinusoid = sinusoidal_timestep_embedding(t, self.fourier_dim)  # (B, fourier_dim)
        t_emb = self.mlp(sinusoid)                                      # (B, cond_dim)
        return t_emb


# ─── AdaLN-Zero Conditioning ─────────────────────────────────────────────────


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization (AdaLN-Zero).

    Modulates LayerNorm shift/scale with a conditioning signal c.
    Also produces per-block residual gates initialized at zero,
    ensuring identity transformation at training start.

    From: "Scalable Diffusion Models with Transformers" (DiT, Peebles & Xie 2023)

    For each DiTBlock, produces 6 modulation params:
        shift_attn, scale_attn, gate_attn,   ← for attention branch
        shift_ffn,  scale_ffn,  gate_ffn     ← for FFN branch

    Args:
        dim:      Feature dimension of x
        cond_dim: Conditioning dimension (timestep + label embedding)
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        # Shared LayerNorm (no learnable affine — AdaLN replaces that)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Projects condition → 6 * dim modulation params
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        # Zero-init: all gates = 0 at start → identity block at init
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Compute 6 modulation parameters from condition c.

        Args:
            x: (B, S, dim)      — used only to determine shapes (norm done here)
            c: (B, cond_dim)    — conditioning vector

        Returns:
            6-tuple, each (B, 1, dim), ready to broadcast over seq dimension:
              shift_attn, scale_attn, gate_attn,
              shift_ffn,  scale_ffn,  gate_ffn

        Shape trace:
            c            : (B, cond_dim)
            raw params   : (B, 6*dim)
            unsqueezed   : (B, 1, 6*dim)
            6 × (B, 1, dim) after chunk
        """
        params = self.modulation(c)          # (B, 6*dim)
        params = params.unsqueeze(1)         # (B, 1, 6*dim)  ← broadcast over S
        return params.chunk(6, dim=-1)       # 6 × (B, 1, dim)


# ─── Single DiT Block ─────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    """
    Single Diffusion Transformer block with AdaLN-Zero conditioning.

    Block computation:
        shift1, scale1, gate1, shift2, scale2, gate2 = AdaLN(x, c)

        Attention branch:
          x_mod = norm(x) * (1 + scale1) + shift1     (B, S, H)
          attn  = MultiheadAttention(x_mod, x_mod, x_mod)
          x     = x + gate1 * attn                    (B, S, H)

        FFN branch:
          x_mod = norm(x) * (1 + scale2) + shift2     (B, S, H)
          ffn   = Linear → GELU → Dropout → Linear
          x     = x + gate2 * ffn                     (B, S, H)

    Args:
        hidden_dim:    Token feature dimension H
        cond_dim:      Conditioning dimension C
        num_heads:     Number of attention heads
        ff_multiplier: FFN hidden = hidden_dim * ff_multiplier
        dropout:       Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        num_heads: int,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # AdaLN-Zero module (holds the shared norm + modulation projection)
        self.adaln = AdaLNZero(dim=hidden_dim, cond_dim=cond_dim)

        # Multi-head self-attention (batch_first=True: inputs are B×S×H)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network
        ff_dim = hidden_dim * ff_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, S, H)    input sequence
            c:    (B, C)       conditioning vector
            mask: Optional attention mask

        Returns:
            x:    (B, S, H)    updated sequence

        Shape trace:
            x         : (B, S, H)
            shift/scale: (B, 1, H) each — broadcast over S
            x_mod1    : (B, S, H)   modulated for attention
            attn_out  : (B, S, H)
            x         : (B, S, H)   after gate1 residual
            x_mod2    : (B, S, H)   modulated for FFN
            ffn_out   : (B, S, H)
            x         : (B, S, H)   after gate2 residual
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaln(x, c)
        # Each shape: (B, 1, H) — broadcasts over S

        # ── Attention branch ──────────────────────────────────────────────────
        x_norm1 = self.adaln.norm(x)                       # (B, S, H)
        x_mod1  = x_norm1 * (1.0 + scale1) + shift1        # (B, S, H)
        attn_out, _ = self.attn(x_mod1, x_mod1, x_mod1, attn_mask=mask)  # (B, S, H)
        x = x + gate1 * attn_out                           # (B, S, H)

        # ── FFN branch ────────────────────────────────────────────────────────
        x_norm2 = self.adaln.norm(x)                       # (B, S, H)
        x_mod2  = x_norm2 * (1.0 + scale2) + shift2        # (B, S, H)
        ffn_out = self.ffn(x_mod2)                         # (B, S, H)
        x = x + gate2 * ffn_out                            # (B, S, H)

        return x


# ─── Full DiT Model ───────────────────────────────────────────────────────────


class DiT(nn.Module):
    """
    Diffusion Transformer for tabular time-series latent space.

    Processes sequences of latent vectors z ∈ R^(B×S×D), conditioned on:
      - Diffusion timestep t ∈ {0, …, T-1}
      - Optional class label y  (supports classifier-free guidance)

    Full forward pass:
    ┌──────────────────────────────────────────────────────────────────┐
    │  INPUTS                                                          │
    │  z_noisy (B, S, D)    — noisy encoder latent                    │
    │  t       (B,)          — diffusion timestep                     │
    │  y       (B,)          — class label (optional)                 │
    ├──────────────────────────────────────────────────────────────────┤
    │  CONDITIONING                                                    │
    │  t_emb = TimestepMLP(t)          → (B, C)                       │
    │  y_emb = LabelEmbed(y)           → (B, C)   (with CFG dropout)  │
    │  c = t_emb + y_emb               → (B, C)                       │
    ├──────────────────────────────────────────────────────────────────┤
    │  SEQUENCE PROCESSING                                             │
    │  z_h = input_proj(z_noisy)       → (B, S, H)                    │
    │  z_h = DiTBlock_1(z_h, c)        → (B, S, H)                    │
    │  z_h = DiTBlock_2(z_h, c)          ...                          │
    │  z_h = DiTBlock_N(z_h, c)        → (B, S, H)                    │
    ├──────────────────────────────────────────────────────────────────┤
    │  OUTPUT                                                          │
    │  z_h = final_norm(z_h)           → (B, S, H)                    │
    │  ε̂   = output_proj(z_h)          → (B, S, D)                    │
    └──────────────────────────────────────────────────────────────────┘

    Classifier-Free Guidance (CFG):
      During training, labels are randomly replaced with a null token
      with probability cfg_dropout_prob.
      During inference, run the model twice:
        ε̂_cond   = DiT(z_t, t, y=y)
        ε̂_uncond = DiT(z_t, t, y=null)
        ε̂        = ε̂_uncond + cfg_scale * (ε̂_cond - ε̂_uncond)

    Args:
        latent_dim:       D — must match TransformerEncoder.embedding_dim
        hidden_dim:       H — internal transformer width
        cond_dim:         C — timestep/label conditioning width
        num_layers:       Number of DiTBlock stacks
        num_heads:        Self-attention heads
        ff_multiplier:    FFN expansion (hidden_dim × ff_multiplier)
        dropout:          Dropout rate
        num_classes:      K — class count for label conditioning; None = unconditional
        cfg_dropout_prob: Probability to drop label to null during training
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_multiplier: int = 4,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        cfg_dropout_prob: float = 0.10,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.cond_dim   = cond_dim
        self.num_classes = num_classes
        self.cfg_dropout_prob = cfg_dropout_prob

        # ── Input projection: D → H ───────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ── Timestep conditioning ─────────────────────────────────────────────
        # fourier_dim matches cond_dim for simplicity
        self.time_embed = TimestepEmbedMLP(fourier_dim=cond_dim, cond_dim=cond_dim)

        # ── Label conditioning (optional, supports CFG) ───────────────────────
        if num_classes is not None:
            # Index [num_classes] is reserved for the null / unconditional token
            self.label_embed  = nn.Embedding(num_classes + 1, cond_dim)
            self.null_label_id = num_classes
        else:
            self.label_embed   = None
            self.null_label_id = None

        # ── Stack of DiT blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                num_heads=num_heads,
                ff_multiplier=ff_multiplier,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Output head ───────────────────────────────────────────────────────
        self.final_norm  = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

        # Zero-init output projection → model predicts ≈ 0 at start (stable)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_condition(
        self,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        B: int,
        device: torch.device,
        cfg_force_null: bool = False,
    ) -> torch.Tensor:
        """
        Build the conditioning vector c = t_emb + y_emb.

        Args:
            t:              (B,) integer timestep indices
            y:              (B,) class labels or None
            B:              batch size
            device:         target device
            cfg_force_null: If True, replace all labels with null (for CFG inference)

        Returns:
            c: (B, cond_dim)

        Shape trace:
            t_emb : (B, cond_dim)
            y_emb : (B, cond_dim) — zeros if no label embed
            c     : (B, cond_dim)
        """
        t_emb = self.time_embed(t)  # (B, cond_dim)
        c = t_emb

        if self.label_embed is not None:
            if cfg_force_null:
                # All-null for unconditional branch of CFG at inference
                y_idx = torch.full((B,), self.null_label_id, dtype=torch.long, device=device)
            elif y is not None and self.training:
                # Randomly drop labels → replace with null token (CFG training)
                drop_mask = torch.rand(B, device=device) < self.cfg_dropout_prob  # (B,)
                y_idx = torch.where(
                    drop_mask,
                    torch.full_like(y, self.null_label_id),
                    y,
                )  # (B,)
            elif y is not None:
                y_idx = y  # (B,) — inference with labels
            else:
                # No label provided at inference — use null
                y_idx = torch.full((B,), self.null_label_id, dtype=torch.long, device=device)

            y_emb = self.label_embed(y_idx)  # (B, cond_dim)
            c = c + y_emb                    # (B, cond_dim)

        return c  # (B, cond_dim)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        cfg_force_null: bool = False,
    ) -> torch.Tensor:
        """
        Predict noise for the given noisy latent z at timestep t.

        Args:
            z:              (B, S, D)   noisy latent input
            t:              (B,)        integer diffusion timesteps
            y:              (B,)        optional class labels
            mask:           Optional attention mask
            cfg_force_null: Force null label (for CFG unconditional pass)

        Returns:
            noise_pred: (B, S, D)

        Full shape trace:
            z           : (B, S, D)
            z_h         : (B, S, H)     after input_proj
            t_emb       : (B, C)        sinusoidal MLP
            y_emb       : (B, C)        label embedding (or 0)
            c           : (B, C)        combined condition
            per block:
              x_mod     : (B, S, H)     AdaLN modulated
              attn_out  : (B, S, H)
              ffn_out   : (B, S, H)
            z_h         : (B, S, H)     final_norm
            noise_pred  : (B, S, D)     output_proj
        """
        B, S, D = z.shape  # (B, seq_len, latent_dim)

        # ── 1. Project latent to hidden dim ───────────────────────────────────
        z_h = self.input_proj(z)  # (B, S, H)

        # ── 2. Build conditioning vector ──────────────────────────────────────
        c = self._build_condition(t, y, B, z.device, cfg_force_null)  # (B, C)

        # ── 3. Process through DiT blocks ─────────────────────────────────────
        for block in self.blocks:
            z_h = block(z_h, c, mask=mask)  # (B, S, H)

        # ── 4. Final norm + output projection ────────────────────────────────
        z_h        = self.final_norm(z_h)   # (B, S, H)
        noise_pred = self.output_proj(z_h)  # (B, S, D)

        return noise_pred  # (B, S, D) — same shape as z input

    def get_param_count(self) -> dict:
        """Return parameter counts by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "input_proj":   count(self.input_proj),
            "time_embed":   count(self.time_embed),
            "label_embed":  count(self.label_embed) if self.label_embed else 0,
            "dit_blocks":   count(self.blocks),
            "output_head":  count(self.final_norm) + count(self.output_proj),
            "total":        count(self),
        }
