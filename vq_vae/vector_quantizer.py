"""
Vector Quantizer module for VQ-VAE.
Implements exponential moving average (EMA) version of VQ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .codebook_tracker import CodebookTracker


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA updates and usage tracking.
    
    Args:
        num_embeddings: Number of embedding vectors (codebook size)
        embedding_dim: Dimensionality of embedding vectors
        commitment_cost: Scalar which controls the weighting of the loss terms
        decay: Decay parameter for EMA updates
        epsilon: Small float constant for numerical stability
        use_tracker: Enable codebook usage tracking (default: True)
        tracker_config: Dictionary with tracker configuration:
            - threshold_strategy: 'count', 'ratio', or 'entropy'
            - threshold_value: Threshold for identifying unused vectors
            - reset_interval: How often to check for unused vectors
            - reinit_strategy: 'random', 'perturb', or 'kmeans'
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_tracker: bool = True,
        tracker_config: Optional[dict] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_tracker = use_tracker
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        
        # Register buffers for EMA
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("w", self.embeddings.weight.data.clone())
        
        # Initialize tracker
        if use_tracker:
            tracker_cfg = tracker_config or {}
            self.tracker = CodebookTracker(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                threshold_strategy=tracker_cfg.get("threshold_strategy", "count"),
                threshold_value=tracker_cfg.get("threshold_value", 1.0),
                reset_interval=tracker_cfg.get("reset_interval", 100),
                reinit_strategy=tracker_cfg.get("reinit_strategy", "perturb"),
            )
        else:
            self.tracker = None
    
    def forward(self, inputs: torch.Tensor):
        """
        Quantize input tensors.
        
        Args:
            inputs: Tensor of shape (batch_size, ..., embedding_dim)
        
        Returns:
            Dictionary containing:
                - quantized: Quantized tensor (same shape as input)
                - loss: VQ loss (scalar)
                - perplexity: Perplexity of the codebook usage
                - encodings: One-hot encodings of shape (batch_size, ..., num_embeddings)
                - encoding_indices: Indices of closest embeddings
                - tracker_info: Information from tracker (if enabled)
        """
        # Flatten input except last dimension
        input_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances to all embeddings
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight ** 2, dim=1, keepdim=True).t()
        )
        
        # Get the closest embedding indices
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # Quantize and unflatten
        quantized = F.embedding(encoding_indices, self.embeddings.weight)
        quantized = quantized.reshape(input_shape)
        
        # Check for numerical issues in quantized values
        if torch.isnan(quantized).any() or torch.isinf(quantized).any():
            print(f"⚠️  WARNING: NaN/Inf detected in quantized output!")
            print(f"   Quantized stats: min={quantized.min()}, max={quantized.max()}, mean={quantized.mean()}")
            # Clamp problematic values
            quantized = torch.clamp(quantized, -1e3, 1e3)
        
        # Track usage
        tracker_info = {}
        if self.use_tracker and self.tracker is not None:
            indices_reshaped = encoding_indices.view(input_shape[:-1])
            self.tracker.update_usage(indices_reshaped)
            
            # Check if reset is needed
            if self.tracker.should_reset():
                unused_indices, _ = self.tracker.get_unused_vectors()
                if len(unused_indices) > 0:
                    # Reinitialize unused vectors
                    self.embeddings.weight.data, reset_list = (
                        self.tracker.reinitialize_vectors(
                            self.embeddings.weight.data,
                            unused_indices=unused_indices,
                        )
                    )
                    # Update w buffer
                    self.w.data = self.embeddings.weight.data.clone()
                    tracker_info["reset_indices"] = reset_list
            
            tracker_info["stats"] = self.tracker.get_statistics()
        
        # Update embeddings with EMA
        if self.training:
            self._update_embeddings_ema(flat_inputs, encodings, encoding_indices)
        
        # Loss computation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.epsilon)))
        
        # Reshape encoding indices back to original shape (except last dim)
        encoding_indices = encoding_indices.reshape(input_shape[:-1])
        
        return {
            "quantized": quantized,
            "loss": loss,
            "perplexity": perplexity,
            "encodings": encodings,
            "encoding_indices": encoding_indices,
            "tracker_info": tracker_info,
        }
    
    def _update_embeddings_ema(
        self,
        flat_inputs: torch.Tensor,
        encodings: torch.Tensor,
        encoding_indices: torch.Tensor,
    ):
        """Update embeddings using exponential moving average."""
        # Numerically stable cluster size update
        updated_cluster_size = (
            self.decay * self.cluster_size
            + (1 - self.decay) * torch.sum(encodings, dim=0)
        )
        
        # Ensure cluster_size doesn't have NaN/Inf
        updated_cluster_size = torch.clamp(updated_cluster_size, min=self.epsilon)
        
        dw = torch.matmul(encodings.t(), flat_inputs)
        updated_w = self.decay * self.w + (1 - self.decay) * dw
        
        # Clamp updated_w to prevent explosion
        updated_w = torch.clamp(updated_w, -1e3, 1e3)
        
        n = torch.sum(updated_cluster_size)
        n = torch.clamp(n, min=self.epsilon)  # Prevent division by zero
        
        updated_cluster_size = (
            (updated_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )
        
        # Numerically stable normalization with safe division
        # Add epsilon before division to prevent inf
        normalised_updated_w = updated_w / (updated_cluster_size.unsqueeze(1) + self.epsilon)
        
        # Clamp result to prevent inf/nan
        normalised_updated_w = torch.clamp(normalised_updated_w, -1e3, 1e3)
        
        # Check for bad values and log warning
        if torch.isnan(normalised_updated_w).any() or torch.isinf(normalised_updated_w).any():
            print(f"⚠️  WARNING: NaN/Inf detected in EMA update!")
            print(f"   Before clamp: min={updated_w.min()}, max={updated_w.max()}")
            print(f"   Cluster sizes: min={updated_cluster_size.min()}, max={updated_cluster_size.max()}")
            # Reinitialize to safe values
            normalised_updated_w = torch.clamp(normalised_updated_w, -1e3, 1e3)
        
        self.cluster_size.data = updated_cluster_size
        self.w.data = normalised_updated_w
        self.embeddings.weight.data = normalised_updated_w
    
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Get encoding indices for inputs."""
        input_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t())
            + torch.sum(self.embeddings.weight ** 2, dim=1, keepdim=True).t()
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices.reshape(input_shape[:-1])
    
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices back to embedding vectors."""
        return F.embedding(indices, self.embeddings.weight)
    
    def get_tracker_stats(self):
        """Get tracker statistics if tracker is enabled."""
        if self.tracker is None:
            return None
        return self.tracker.get_statistics()
    
    def get_tracker_summary(self):
        """Get tracker summary if tracker is enabled."""
        if self.tracker is None:
            return None
        return str(self.tracker)
    
    def reset_tracker(self):
        """Reset tracker counts."""
        if self.tracker is not None:
            self.tracker.reset_counts()
