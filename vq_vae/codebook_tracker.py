"""
Codebook Tracker module for monitoring and managing vector usage in VQ-VAE.
Tracks usage frequency of each codebook vector and reinitializes underutilized vectors.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class CodebookTracker(nn.Module):
    """
    Tracks usage of codebook vectors and reinitializes underutilized ones.
    
    Prevents codebook collapse by monitoring which vectors are being used
    and reinitializing vectors that haven't been used recently.
    
    Args:
        num_embeddings: Number of codebook vectors
        embedding_dim: Dimension of each embedding
        threshold_strategy: Strategy for identifying unused vectors
            - 'count': vectors with usage count below threshold
            - 'ratio': vectors with usage ratio below threshold
            - 'entropy': based on usage entropy
        threshold_value: Threshold for the strategy (default: 1.0)
        reset_interval: How often to check and reset (default: 100 batches)
        reinit_strategy: How to reinitialize vectors
            - 'random': random initialization
            - 'perturb': perturb from used vectors
            - 'kmeans': from data centroids (requires data)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        threshold_strategy: str = "count",
        threshold_value: float = 1.0,
        reset_interval: int = 100,
        reinit_strategy: str = "perturb",
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.threshold_strategy = threshold_strategy
        self.threshold_value = threshold_value
        self.reset_interval = reset_interval
        self.reinit_strategy = reinit_strategy
        
        # Register buffers for tracking
        self.register_buffer(
            "usage_count",
            torch.zeros(num_embeddings, dtype=torch.long),
        )
        self.register_buffer(
            "usage_history",
            torch.zeros(num_embeddings, dtype=torch.float),
        )
        
        # Tracking state
        self.batch_counter = 0
        self.total_resets = 0
        self.reset_history: List[List[int]] = []
    
    def reset_counts(self):
        """Reset usage counters."""
        self.usage_count.zero_()
    
    def reset_history(self):
        """Clear reset history."""
        self.reset_history = []
    
    def update_usage(self, indices: torch.Tensor):
        """
        Update usage counts based on encoded indices.
        
        Args:
            indices: Tensor of shape (batch_size, seq_len) or (N,) containing
                    indices into the codebook
        """
        # Flatten indices
        flat_indices = indices.flatten()
        
        # Update counts
        for idx in flat_indices:
            self.usage_count[idx] += 1
        
        # Update history (exponential moving average)
        alpha = 0.1  # EMA coefficient
        unique_indices = torch.unique(flat_indices)
        for idx in unique_indices:
            count = (flat_indices == idx).sum().float()
            self.usage_history[idx] = (
                alpha * count + (1 - alpha) * self.usage_history[idx]
            )
        
        self.batch_counter += 1
    
    def get_unused_vectors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Identify unused or underutilized vectors.
        
        Returns:
            Tuple of (unused_indices, usage_values)
        """
        if self.threshold_strategy == "count":
            # Vectors with usage count below threshold
            unused_mask = self.usage_count < self.threshold_value
            unused_indices = torch.where(unused_mask)[0]
            usage_values = self.usage_count[unused_mask]
        
        elif self.threshold_strategy == "ratio":
            # Vectors with usage ratio below threshold
            mean_usage = self.usage_count.float().mean()
            threshold = mean_usage * self.threshold_value
            unused_mask = self.usage_count.float() < threshold
            unused_indices = torch.where(unused_mask)[0]
            usage_values = self.usage_count[unused_mask].float()
        
        elif self.threshold_strategy == "entropy":
            # Based on usage entropy (less used = higher entropy contribution)
            probs = self.usage_count.float() / (self.usage_count.sum() + 1e-8)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8))
            threshold = entropy * self.threshold_value
            # Vectors below mean usage
            mean_usage = probs.mean()
            unused_mask = probs < (mean_usage * self.threshold_value)
            unused_indices = torch.where(unused_mask)[0]
            usage_values = probs[unused_mask]
        
        else:
            raise ValueError(f"Unknown threshold strategy: {self.threshold_strategy}")
        
        return unused_indices, usage_values
    
    def should_reset(self) -> bool:
        """Check if reset interval has been reached."""
        return self.batch_counter % self.reset_interval == 0
    
    def get_statistics(self) -> Dict:
        """
        Get detailed usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        usage_count = self.usage_count.float()
        total_usage = usage_count.sum().item()
        
        # Handle edge case where no vectors have been used
        if total_usage == 0:
            return {
                "total_usage": 0,
                "mean_usage": 0,
                "std_usage": 0,
                "min_usage": 0,
                "max_usage": 0,
                "unused_count": self.num_embeddings,
                "unused_ratio": 1.0,
                "perplexity": 0,
                "batch_counter": self.batch_counter,
                "total_resets": self.total_resets,
            }
        
        mean_usage = usage_count.mean().item()
        std_usage = usage_count.std().item()
        min_usage = usage_count.min().item()
        max_usage = usage_count.max().item()
        
        # Count unused vectors
        unused_count = (usage_count == 0).sum().item()
        unused_ratio = unused_count / self.num_embeddings
        
        # Calculate perplexity (measure of codebook utilization)
        probs = usage_count / total_usage
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        perplexity = torch.exp(entropy).item()
        
        return {
            "total_usage": int(total_usage),
            "mean_usage": mean_usage,
            "std_usage": std_usage,
            "min_usage": min_usage,
            "max_usage": max_usage,
            "unused_count": unused_count,
            "unused_ratio": unused_ratio,
            "perplexity": perplexity,
            "batch_counter": self.batch_counter,
            "total_resets": self.total_resets,
        }
    
    def reinitialize_vectors(
        self,
        embeddings: torch.Tensor,
        unused_indices: Optional[torch.Tensor] = None,
        data_batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Reinitialize unused vectors in the codebook.
        
        Args:
            embeddings: The current codebook embeddings tensor
            unused_indices: Indices of vectors to reinitialize (if None, auto-detect)
            data_batch: Data for KMeans-based reinitialization
        
        Returns:
            Tuple of (updated embeddings, list of reset indices)
        """
        if unused_indices is None:
            unused_indices, _ = self.get_unused_vectors()
        
        if len(unused_indices) == 0:
            return embeddings, []
        
        unused_indices_list = unused_indices.tolist()
        
        if self.reinit_strategy == "random":
            # Random reinitialization
            for idx in unused_indices:
                embeddings[idx] = torch.randn_like(embeddings[idx])
        
        elif self.reinit_strategy == "perturb":
            # Perturb from used vectors
            used_mask = self.usage_count > 0
            used_indices = torch.where(used_mask)[0]
            
            if len(used_indices) > 0:
                for idx in unused_indices:
                    # Randomly select a used vector and perturb it
                    source_idx = used_indices[
                        torch.randint(len(used_indices), (1,)).item()
                    ]
                    noise = torch.randn_like(embeddings[idx]) * 0.1
                    embeddings[idx] = embeddings[source_idx] + noise
            else:
                # If no used vectors, use random
                for idx in unused_indices:
                    embeddings[idx] = torch.randn_like(embeddings[idx])
        
        elif self.reinit_strategy == "kmeans":
            # KMeans-based reinitialization (requires data)
            if data_batch is None:
                raise ValueError(
                    "data_batch is required for kmeans reinit_strategy"
                )
            # Get centroids of clusters for unused vectors
            flat_data = data_batch.reshape(-1, data_batch.shape[-1])
            
            for idx in unused_indices:
                # Random sample from data
                random_idx = torch.randint(len(flat_data), (1,)).item()
                embeddings[idx] = flat_data[random_idx].clone()
        
        else:
            raise ValueError(f"Unknown reinit strategy: {self.reinit_strategy}")
        
        # Update tracking
        self.total_resets += len(unused_indices)
        self.reset_history.append(unused_indices_list)
        # Reset usage for reinitialized vectors
        self.usage_count[unused_indices] = 0
        self.usage_history[unused_indices] = 0
        
        return embeddings, unused_indices_list
    
    def get_reset_history_summary(self) -> Dict:
        """Get summary of reset history."""
        if not self.reset_history:
            return {
                "total_reset_operations": 0,
                "total_vectors_reset": 0,
                "reset_operations": [],
            }
        
        total_vectors = sum(len(ops) for ops in self.reset_history)
        
        return {
            "total_reset_operations": len(self.reset_history),
            "total_vectors_reset": total_vectors,
            "average_vectors_per_reset": total_vectors / len(self.reset_history),
            "reset_operations": self.reset_history,
        }
    
    def get_usage_distribution(self) -> Dict:
        """Get distribution of vector usage."""
        usage = self.usage_count.float()
        
        # Bin usage counts
        bins = [0, 1, 5, 10, 50, 100, 500, 1000]
        bin_counts = []
        
        for i in range(len(bins) - 1):
            count = ((usage >= bins[i]) & (usage < bins[i + 1])).sum().item()
            bin_counts.append(count)
        
        # Last bin: >= last value
        count = (usage >= bins[-1]).sum().item()
        bin_counts.append(count)
        
        return {
            "bins": bins,
            "counts": bin_counts,
            "zero_usage": (usage == 0).sum().item(),
            "nonzero_usage": (usage > 0).sum().item(),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        lines = [
            "\nCodebookTracker Summary:",
            f"  Embeddings: {self.num_embeddings} x {self.embedding_dim}",
            f"  Strategy: {self.threshold_strategy} (threshold={self.threshold_value})",
            f"  Reinit Strategy: {self.reinit_strategy}",
            f"  Reset Interval: {self.reset_interval} batches",
            "-" * 50,
            f"  Total Usage: {stats['total_usage']:,}",
            f"  Mean Usage: {stats['mean_usage']:.2f}",
            f"  Unused Vectors: {stats['unused_count']}/{self.num_embeddings} "
            f"({stats['unused_ratio']*100:.1f}%)",
            f"  Perplexity: {stats['perplexity']:.2f}",
            f"  Total Resets: {stats['total_resets']}",
            "-" * 50,
        ]
        return "\n".join(lines)
