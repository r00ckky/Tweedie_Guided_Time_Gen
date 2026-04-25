import torch
import torch.nn as nn
from typing import Optional, Tuple

from .denoiser import TabularDenoiser
from vq_vae import VQ_VAE
from .config import NCSNConfig

class AmexGuidedGenerator(nn.Module):
    def __init__(self, vqvae_model: VQ_VAE, ncsn_config: NCSNConfig):
        super().__init__()
        self.config = ncsn_config

        self.vqvae = vqvae_model
        self.vqvae.eval()
        for param in self.vqvae.parameters():
            param.requires_grad = False
        
        flat_dim = self.vqvae.config.input_dim 
        self.denoiser = TabularDenoiser(
            input_dim=flat_dim, 
            hidden_dim=ncsn_config.denoiser_hidden_dim,
            num_blocks=ncsn_config.ncsn_num_blocks
        )

    def get_data_score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predicts the noise/score of the current feature vector."""
        return self.denoiser(x_t, t)

    def get_guidance_score(
        self, 
        x_t: torch.Tensor, 
        score_data: torch.Tensor, 
        time_tensor: torch.Tensor,
        sigma: torch.Tensor, 
        target_class: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates guidance using Tweedie's formula to estimate x_0 before classifying.
        """
        with torch.enable_grad():
            x_t = x_t.detach().requires_grad_(True)
            
            #Tweedie's Estimate for clean data (x_0)
            x_0_hat = x_t + (sigma ** 2) * score_data.detach()            
            x_0_hat = torch.clamp(x_0_hat, min=-5.2, max=5.2)
            
            encode_output = self.vqvae.encode(x_0_hat, y=target_class, time_tensor=time_tensor)
            cls_loss = encode_output["classification_loss"] 
            
            if cls_loss is None:
                raise ValueError("classification_loss returned None.")
                
            grad = torch.autograd.grad(cls_loss, x_t)[0]
            guidance_score = -grad
            
        return guidance_score

    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Continuous Variance Exploding (VE) noise schedule."""
        return self.config.sigma_min * ((self.config.sigma_max / self.config.sigma_min) ** t)

    @torch.no_grad()
    def generate(
        self, 
        batch_size: int, 
        seq_len: int,  
        target_class: torch.Tensor,
        time_tensor: torch.Tensor,
        steps: int = 50, 
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        """
        Langevin Dynamics sampling loop in feature space with Tweedie Guidance.
        """
        device = next(self.parameters()).device
        input_dim = self.vqvae.config.input_dim
        
        x = torch.randn((batch_size, seq_len, input_dim), device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t_val = 1.0 - (i / steps)
            t = torch.full((batch_size,), t_val, device=device)
            
            sigma = self.get_sigma(t).view(-1, 1, 1)
            
            score_data = self.get_data_score(x, t)
            
            if guidance_scale > 0:
                score_guide = self.get_guidance_score(
                    x_t=x, 
                    score_data=score_data, 
                    sigma=sigma, 
                    target_class=target_class,
                    time_tensor=time_tensor
                )
            else:
                score_guide = torch.zeros_like(x)
                
            noise = torch.randn_like(x) * torch.sqrt(torch.tensor(dt, device=device))    
            x = x - (0.5 * score_data + guidance_scale * score_guide) * dt + noise
            
        return x
    
