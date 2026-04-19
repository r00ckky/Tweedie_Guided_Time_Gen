import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─── Noise Schedules ─────────────────────────────────────────────────────────


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal, 2021).

    Produces a smoother, more uniform noise profile than linear.
    Recommended for latent diffusion.

    Args:
        timesteps: Total diffusion steps T
        s:         Small offset to prevent β_0 from being too small

    Returns:
        betas: (T,) clamped to [1e-4, 0.9999]
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]          # Normalize: ᾱ_0 = 1
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])    # β_t = 1 - ᾱ_t/ᾱ_{t-1}
    return torch.clamp(betas, min=1e-4, max=0.9999)


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Linear noise schedule (Ho et al., 2020 original DDPM).

    Args:
        timesteps:  Total diffusion steps T
        beta_start: Starting β value
        beta_end:   Ending β value

    Returns:
        betas: (T,) linearly spaced
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# ─── Gaussian Diffusion ───────────────────────────────────────────────────────


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion process for latent sequence tensors.

    Implements the full DDPM forward / reverse process with pre-computed
    schedule quantities stored as registered buffers.

    Forward process (analytical, closed form):
        q(z_t | z_0) = N(√ᾱ_t · z_0, (1 - ᾱ_t) · I)

    Reverse process (learned, iterative):
        p_θ(z_{t-1} | z_t) — parameterized via noise predictor ε_θ

    Training objective:
        L = E_{z_0, t, ε} [ || ε - ε_θ(z_t, t) ||² ]

    Args:
        timesteps:   Total diffusion steps T (default 1000)
        schedule:    "cosine" (recommended) or "linear"
        beta_start:  For linear schedule only
        beta_end:    For linear schedule only
    """

    def __init__(
        self,
        timesteps: int = 1_000,
        schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.timesteps = timesteps

        # ── Noise schedule ────────────────────────────────────────────────────
        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule '{schedule}'. Use 'cosine' or 'linear'.")

        alphas            = 1.0 - betas                                  # (T,)
        alphas_cumprod    = torch.cumprod(alphas, dim=0)                 # ᾱ_t (T,)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # ᾱ_{t-1}

        # ── Register all schedule tensors as non-parameter buffers ────────────
        self.register_buffer("betas",            betas)
        self.register_buffer("alphas",           alphas)
        self.register_buffer("alphas_cumprod",   alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Forward process coefficients — q(z_t | z_0)
        self.register_buffer("sqrt_alphas_cumprod",           torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod",  torch.log(1.0 - alphas_cumprod + 1e-20))

        # Reverse process coefficients — for predicting z_0 from z_t + ε̂
        self.register_buffer("sqrt_recip_alphas_cumprod",  torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0 + 1e-20))

        # Posterior q(z_{t-1} | z_t, z_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-20)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-20),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod + 1e-20),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _gather(self, schedule: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Gather schedule values at timestep t and reshape for broadcasting.

        Args:
            schedule: (T,) pre-computed schedule buffer
            t:        (B,) integer timestep indices
            x_shape:  Shape of target tensor, e.g. (B, S, D)

        Returns:
            Gathered values reshaped to (B, 1, 1, …) for broadcasting

        Example:
            t = [3, 7, 1]  →  schedule[[3, 7, 1]]  →  reshape (3, 1, 1)
        """
        out = schedule.gather(-1, t)                              # (B,)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))  # (B, 1, …)

    # ── Forward Process ───────────────────────────────────────────────────────

    def q_sample(
        self,
        z_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: sample z_t from q(z_t | z_0).

        z_t = √ᾱ_t · z_0  +  √(1-ᾱ_t) · ε,   ε ~ N(0, I)

        Args:
            z_0:   (B, S, D)  clean encoder latent
            t:     (B,)       integer timestep indices
            noise: (B, S, D)  optional pre-sampled noise; sampled fresh if None

        Returns:
            z_t:   (B, S, D)  noisy latent at timestep t
            noise: (B, S, D)  noise that was added

        Shape trace:
            z_0                : (B, S, D)
            sqrt_alpha_bar     : (B, 1, 1)  ← gathered + broadcast
            sqrt_1_minus_alpha : (B, 1, 1)  ← gathered + broadcast
            z_t = coef1·z_0 + coef2·ε : (B, S, D)
        """
        if noise is None:
            noise = torch.randn_like(z_0)  # ε ~ N(0, I),  (B, S, D)

        sqrt_ab  = self._gather(self.sqrt_alphas_cumprod,           t, z_0.shape)  # (B, 1, 1)
        sqrt_1ab = self._gather(self.sqrt_one_minus_alphas_cumprod, t, z_0.shape)  # (B, 1, 1)

        z_t = sqrt_ab * z_0 + sqrt_1ab * noise  # (B, S, D)
        return z_t, noise

    # ── Loss ─────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Standard DDPM MSE loss: E[|| ε - ε̂ ||²]

        Args:
            noise_pred:   (B, S, D) DiT prediction
            noise_target: (B, S, D) actual noise ε

        Returns:
            loss: scalar tensor

        Shape: (B, S, D) → reduction to scalar
        """
        return F.mse_loss(noise_pred, noise_target)  # mean over all elements

    # ── Reverse Step ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample_step(
        self,
        model: nn.Module,
        z_t: torch.Tensor,
        t_idx: int,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: z_{t-1} ~ p_θ(z_{t-1} | z_t)

        Optionally applies Classifier-Free Guidance (CFG):
            ε̂ = ε̂_uncond + cfg_scale * (ε̂_cond - ε̂_uncond)

        Args:
            model:     DiT noise predictor (must be in eval mode)
            z_t:       (B, S, D) noisy latent at step t
            t_idx:     Integer timestep index (scalar)
            y:         (B,) optional class labels
            cfg_scale: CFG guidance scale (1.0 = no CFG)

        Returns:
            z_prev: (B, S, D) less noisy latent z_{t-1}

        Shape trace:
            z_t         : (B, S, D)
            t           : (B,)      all = t_idx
            ε̂           : (B, S, D) from DiT
            z_0_pred    : (B, S, D) predicted clean latent
            post_mean   : (B, S, D) posterior mean
            z_{t-1}     : (B, S, D) + optional noise σ_t·ε'
        """
        B      = z_t.shape[0]
        device = z_t.device
        t      = torch.full((B,), t_idx, dtype=torch.long, device=device)  # (B,)

        # ── Predict noise (with optional CFG) ─────────────────────────────────
        if cfg_scale > 1.0 and y is not None:
            # Two forward passes: conditioned and unconditioned
            eps_cond   = model(z_t, t, y=y)                       # (B, S, D)
            eps_uncond = model(z_t, t, y=None, cfg_force_null=True)  # (B, S, D)
            eps_pred   = eps_uncond + cfg_scale * (eps_cond - eps_uncond)  # (B, S, D)
        else:
            eps_pred = model(z_t, t, y=y)  # (B, S, D)

        # ── Predict z_0 from z_t and ε̂ ────────────────────────────────────────
        # z_0_pred = (z_t - √(1-ᾱ_t)·ε̂) / √ᾱ_t
        sqrt_recip  = self._gather(self.sqrt_recip_alphas_cumprod,   t, z_t.shape)  # (B,1,1)
        sqrt_recipm1 = self._gather(self.sqrt_recipm1_alphas_cumprod, t, z_t.shape)  # (B,1,1)

        z_0_pred = sqrt_recip * z_t - sqrt_recipm1 * eps_pred  # (B, S, D)
        z_0_pred = torch.clamp(z_0_pred, -5.0, 5.0)            # Stability clamp

        # ── Compute posterior mean  μ_θ(z_t, t) ──────────────────────────────
        # μ = coef1 · z_0_pred + coef2 · z_t
        coef1 = self._gather(self.posterior_mean_coef1, t, z_t.shape)  # (B,1,1)
        coef2 = self._gather(self.posterior_mean_coef2, t, z_t.shape)  # (B,1,1)
        post_mean = coef1 * z_0_pred + coef2 * z_t                     # (B, S, D)

        # ── Add stochastic noise (skip at t=0) ───────────────────────────────
        if t_idx > 0:
            post_log_var = self._gather(self.posterior_log_variance_clipped, t, z_t.shape)
            noise  = torch.randn_like(z_t)
            z_prev = post_mean + (0.5 * post_log_var).exp() * noise  # (B, S, D)
        else:
            z_prev = post_mean  # No noise at final step

        return z_prev  # (B, S, D)

    # ── Full Reverse Sampling ─────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples by iterating the full reverse chain z_T → z_0.

        Args:
            model:         DiT noise predictor (in eval mode)
            shape:         (B, S, D) — shape of output latents
            device:        Target device
            y:             (B,) optional class labels for conditioning
            cfg_scale:     CFG guidance scale (1.0 = no guidance)
            show_progress: Show tqdm progress bar

        Returns:
            z_0: (B, S, D) generated latent sequences

        Shape trace:
            z_T ~ N(0, I) : (B, S, D)
            loop t=T-1…0  : p_sample_step → (B, S, D)
            z_0           : (B, S, D)
        """
        model.eval()
        z = torch.randn(shape, device=device)  # z_T ~ N(0, I)   (B, S, D)

        timestep_iter = range(self.timesteps - 1, -1, -1)
        if show_progress:
            timestep_iter = tqdm(timestep_iter, desc="Denoising", total=self.timesteps)

        for t_idx in timestep_iter:
            z = self.p_sample_step(model, z, t_idx, y=y, cfg_scale=cfg_scale)
            # z: (B, S, D)

        return z  # z_0: (B, S, D)


# ─── Full Latent Diffusion Model ──────────────────────────────────────────────


class LatentDiffusionModel(nn.Module):
    """
    End-to-end Latent Diffusion pipeline for tabular time-series.

    Combines:
      ┌─────────────────────────────────────────────────────────────┐
      │  PatchEmbedding  ← from existing VQ-VAE (frozen)           │
      │  TransformerEncoder ← from existing VQ-VAE (frozen)        │
      │  GaussianDiffusion  ← forward + reverse process            │
      │  DiT                ← trainable noise predictor            │
      └─────────────────────────────────────────────────────────────┘

    During training:
      - Encoder and PatchEmbedding are frozen (or optionally fine-tuned)
      - Only DiT parameters are updated
      - Loss = DDPM MSE between predicted and actual noise

    Args:
        patch_embedding: PatchEmbedding module from VQ-VAE
        encoder:         TransformerEncoder from VQ-VAE
        dit:             DiT noise predictor
        diffusion:       GaussianDiffusion process
        freeze_encoder:  If True, freeze encoder + patch_embedding weights
    """

    def __init__(
        self,
        patch_embedding: nn.Module,
        encoder: nn.Module,
        dit: "DiT",
        diffusion: GaussianDiffusion,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.encoder         = encoder
        self.dit             = dit
        self.diffusion       = diffusion

        if freeze_encoder:
            self.freeze_encoder()

    # ── Encoder freeze / unfreeze ─────────────────────────────────────────────

    def freeze_encoder(self):
        """Freeze PatchEmbedding + Encoder (encoder provides fixed latent targets)."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze for optional joint fine-tuning."""
        for param in self.patch_embedding.parameters():
            param.requires_grad = True
        for param in self.encoder.parameters():
            param.requires_grad = True

    # ── Encode ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        time_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode raw input to clean latent z_e using the frozen encoder.

        Args:
            x:           (B, T, F)   raw tabular features
            y:           (B,)        class labels (passed to encoder for cls_token)
            time_tensor: (B, T, 1)   month delta features

        Returns:
            z_e: (B, T', D)   encoder latents  [T' may differ from T due to patching]

        Shape trace:
            x         : (B, T,  F)
            x_patch   : (B, T', P)  after PatchEmbedding (Conv1D)
            z_e, _, _ : (B, T', D)  after TransformerEncoder
        """
        x_patch = self.patch_embedding(x)                    # (B, T', P)
        z_e, _, _ = self.encoder(x_patch, y, time_tensor)   # (B, T', D)
        return z_e  # (B, T', D)

    # ── Training Step ─────────────────────────────────────────────────────────

    def training_step(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
        time_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full single training step for the DiT diffusion model.

        Steps:
          1. Encode x → z_e             (no gradient, encoder frozen)
          2. Sample timestep t
          3. Sample noise ε ~ N(0, I)
          4. Compute z_t = √ᾱ_t·z_e + √(1-ᾱ_t)·ε     [forward process]
          5. Predict ε̂ = DiT(z_t, t, y)
          6. Loss = MSE(ε̂, ε)

        Args:
            x:           (B, T, F)   raw input features
            y:           (B,)        class labels (optional)
            time_tensor: (B, T, 1)   time delta features

        Returns:
            dict:
              "loss"       : scalar   DDPM MSE loss
              "noise_pred" : (B,T',D) DiT's noise prediction
              "noise"      : (B,T',D) actual sampled noise ε
              "z_e"        : (B,T',D) clean encoder latent
              "z_t"        : (B,T',D) noisy latent
              "t"          : (B,)     sampled diffusion timesteps

        Shape trace:
            x            : (B, T,  F)
            x_patch      : (B, T', P)   PatchEmbedding
            z_e          : (B, T', D)   Encoder (no_grad)
            t            : (B,)         uniform sample in [0, T_diff)
            noise        : (B, T', D)   ε ~ N(0,I)
            z_t          : (B, T', D)   q_sample(z_e, t, noise)
            noise_pred   : (B, T', D)   DiT(z_t, t, y)
            loss         : scalar       F.mse_loss(noise_pred, noise)
        """
        B      = x.shape[0]
        device = x.device

        # ── Step 1: Encode (frozen, no grad) ──────────────────────────────────
        with torch.no_grad():
            x_patch = self.patch_embedding(x)                  # (B, T', P)
            z_e, _, _ = self.encoder(x_patch, y, time_tensor)  # (B, T', D)
        # z_e: (B, T', D) — clean latent targets

        # ── Step 2: Sample random diffusion timestep ──────────────────────────
        t = torch.randint(
            0, self.diffusion.timesteps, (B,),
            device=device, dtype=torch.long,
        )  # (B,)

        # ── Step 3: Sample noise ε ────────────────────────────────────────────
        noise = torch.randn_like(z_e)  # (B, T', D)

        # ── Step 4: Forward process — add noise to z_e ───────────────────────
        z_t, noise = self.diffusion.q_sample(z_e, t, noise=noise)  # (B, T', D)

        # ── Step 5: DiT predicts noise ────────────────────────────────────────
        noise_pred = self.dit(z_t, t, y=y)  # (B, T', D)

        # ── Step 6: Compute MSE loss ──────────────────────────────────────────
        loss = self.diffusion.compute_loss(noise_pred, noise)  # scalar

        return {
            "loss":       loss,        # scalar
            "noise_pred": noise_pred,  # (B, T', D)
            "noise":      noise,       # (B, T', D)
            "z_e":        z_e,         # (B, T', D)
            "z_t":        z_t,         # (B, T', D)
            "t":          t,           # (B,)
        }

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_latents(
        self,
        batch_size: int,
        seq_len: int,
        latent_dim: int,
        device: torch.device,
        y: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate synthetic latent sequences via iterative reverse diffusion.

        Args:
            batch_size:    Number of sequences to generate
            seq_len:       T' — encoder output sequence length (e.g. 13)
            latent_dim:    D  — encoder embedding_dim (e.g. 64)
            device:        Target device
            y:             (batch_size,) class labels for conditioning; None = unconditional
            cfg_scale:     CFG scale:
                             1.0 → no guidance (unconditional)
                             2.0 → moderate label guidance
                             4.0 → strong label guidance
            show_progress: Show tqdm denoising progress

        Returns:
            z_samples: (batch_size, seq_len, latent_dim)

        Shape trace:
            z_T ~ N(0,I) : (B, S, D)
            reverse loop : T-1 → 0, each step → (B, S, D)
            z_0          : (B, S, D)
        """
        self.dit.eval()
        shape = (batch_size, seq_len, latent_dim)  # (B, S, D)

        z_samples = self.diffusion.sample(
            model=self.dit,
            shape=shape,
            device=device,
            y=y,
            cfg_scale=cfg_scale,
            show_progress=show_progress,
        )  # (B, S, D)

        return z_samples

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print model component parameter counts."""
        def count(m):
            total     = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        pe_total, pe_train = count(self.patch_embedding)
        enc_total, enc_train = count(self.encoder)
        dit_counts = self.dit.get_param_count()
        diff_total, _ = count(self.diffusion)

        lines = [
            "\n" + "=" * 60,
            "Latent Diffusion Model Summary",
            "=" * 60,
            f"PatchEmbedding : {pe_total:>12,}  (trainable: {pe_train:,})",
            f"Encoder        : {enc_total:>12,}  (trainable: {enc_train:,})",
            f"GaussianDiffusion: {diff_total:>10,}  (buffers only, no params)",
            "-" * 60,
            "DiT breakdown:",
            f"  input_proj   : {dit_counts['input_proj']:>12,}",
            f"  time_embed   : {dit_counts['time_embed']:>12,}",
            f"  label_embed  : {dit_counts['label_embed']:>12,}",
            f"  dit_blocks   : {dit_counts['dit_blocks']:>12,}",
            f"  output_head  : {dit_counts['output_head']:>12,}",
            f"  DiT Total    : {dit_counts['total']:>12,}",
            "=" * 60,
            f"Total parameters       : {pe_total + enc_total + dit_counts['total']:,}",
            f"Trainable (DiT only)   : {dit_counts['total']:,}",
            "=" * 60 + "\n",
        ]
        return "\n".join(lines)
