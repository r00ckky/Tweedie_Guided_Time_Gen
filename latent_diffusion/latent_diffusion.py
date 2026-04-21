from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .diffusion import GaussianDiffusion
from .dit import DiT


class LatentDiffusionModel(nn.Module):
    """
    End-to-end Latent Diffusion pipeline for tabular time-series.

    Combines:
      ┌─────────────────────────────────────────────────────────────┐
      │  VQ_VAE             ← Full VQ-VAE instance                 │
      │  GaussianDiffusion  ← forward + reverse process            │
      │  DiT                ← trainable noise predictor            │
      └─────────────────────────────────────────────────────────────┘

    During training:
      - VQ-VAE is frozen (or optionally fine-tuned)
      - Only DiT parameters are updated
      - Loss = DDPM MSE between predicted and actual noise

    Args:
        vq_vae:          Full VQ_VAE model
        dit:             DiT noise predictor
        diffusion:       GaussianDiffusion process
        freeze_encoder:  If True, freeze VQ-VAE weights
    """

    def __init__(
        self,
        vq_vae: nn.Module,
        dit: DiT,
        diffusion: GaussianDiffusion,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.vq_vae          = vq_vae
        self.dit             = dit
        self.diffusion       = diffusion

        if freeze_encoder:
            self.freeze_encoder()

    # ── Encoder freeze / unfreeze ─────────────────────────────────────────────

    def freeze_encoder(self):
        """Freeze VQ-VAE (provides fixed latent targets)."""
        for param in self.vq_vae.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze VQ-VAE for optional joint fine-tuning."""
        for param in self.vq_vae.parameters():
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
        Encode raw input to clean latent z_e using the frozen VQ-VAE.

        Args:
            x:           (B, T, F)   raw tabular features
            y:           (B,)        class labels (passed to encoder for cls_token)
            time_tensor: (B, T, 1)   month delta features

        Returns:
            z: (B, T', D)   encoder latents  [T' may differ from T due to patching]
        """
        return self.vq_vae.encode(x, y, time_tensor=time_tensor)["z"]

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
        """
        B      = x.shape[0]
        device = x.device

        # ── Step 1: Encode (frozen, no grad) ──────────────────────────────────
        with torch.no_grad():
            z_e = self.vq_vae.encode(x, y, time_tensor=time_tensor)["z"]
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

    @torch.no_grad()
    def decode(self, z: torch.Tensor, force_quantize: bool = True) -> torch.Tensor:
        """
        Decode latent representation back to the original input space using VQ-VAE.
        
        If force_quantize is True, snaps the continuous diffusion output 
        to the nearest VQ codebook vectors before decoding (recommended).
        """
        if force_quantize:
            # Snap continuous latents to the nearest codebook vectors
            quantization = self.vq_vae.vector_quantizer(z)
            z = quantization["quantized"]
            
        return self.vq_vae.decode(z)

    @torch.no_grad()
    def generate(
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
        Generate synthetic tabular data by sampling latents and decoding them.
        """
        z_samples = self.sample_latents(
            batch_size, seq_len, latent_dim, device, y, cfg_scale, show_progress
        )
        # Decode continuous latents using VQ-VAE's decoder
        return self.decode(z_samples)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Print model component parameter counts."""
        def count(m):
            total     = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        vqvae_total, vqvae_train = count(self.vq_vae)
        dit_counts = self.dit.get_param_count()
        diff_total, _ = count(self.diffusion)

        lines = [
            "\n" + "=" * 60,
            "Latent Diffusion Model Summary",
            "=" * 60,
            f"VQ-VAE         : {vqvae_total:>12,}  (trainable: {vqvae_train:,})",
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
            f"Total parameters       : {vqvae_total + dit_counts['total']:,}",
            f"Trainable (DiT only)   : {dit_counts['total']:,}",
            "=" * 60 + "\n",
        ]
        return "\n".join(lines)