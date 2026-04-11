"""
Main VQ-VAE module combining Encoder, Vector Quantizer, and Decoder.
Also includes PatchEmbedding for time-series data preprocessing.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .config import VQVAEConfig
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .vector_quantizer import VectorQuantizer


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer using 1D Convolution for time-series data.
    
    Converts raw time-series features to patch embeddings.
    Example: input (batch, seq_len, 294) -> output (batch, new_seq_len, patch_embed_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        patch_stride: int,
        patch_embed_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_embed_dim = patch_embed_dim
        
        # Conv1d: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        # then apply Conv1d to get (batch, patch_embed_dim, new_seq_len)
        # then transpose back to (batch, new_seq_len, patch_embed_dim)
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=patch_embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=0,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply patch embedding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            Patch embeddings of shape (batch_size, new_seq_len, patch_embed_dim)
        """
        # x: (batch, seq_len, input_dim)
        # Permute to (batch, input_dim, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply convolution: (batch, input_dim, seq_len) -> (batch, patch_embed_dim, new_seq_len)
        x = self.conv(x)
        
        # Permute back to (batch, new_seq_len, patch_embed_dim)
        x = x.permute(0, 2, 1)
        
        return x


class VQ_VAE(nn.Module):
    """
    Vector Quantized Variational AutoEncoder with Transformer backbone.
    
    Combines:
    - Patch Embedding: converts raw input to patch embeddings
    - Transformer Encoder: compresses patches to latent codes
    - Vector Quantizer: discretizes latent space
    - Transformer Decoder: reconstructs from quantized codes
    
    Args:
        config: VQVAEConfig instance containing all hyperparameters
    """
    
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        
        # Patch Embedding Layer
        self.patch_embedding = PatchEmbedding(
            input_dim=config.input_dim,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            patch_embed_dim=config.patch_embed_dim,
        )
        
        # Encoder
        self.encoder = TransformerEncoder(
            input_dim=config.patch_embed_dim,  # Input is patch_embed_dim, not raw input_dim
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            activation="relu",
            class_token=config.use_class_token,
            class_proj_dim=config.class_proj_dim,
        )
        
        # Vector Quantizer
        self.vector_quantizer = VectorQuantizer(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            commitment_cost=config.commitment_cost,
            decay=config.decay,
            epsilon=config.epsilon,
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.patch_embed_dim,  # Decoder outputs patch_embed_dim
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            activation="relu",
        )
        
        # Reconstruction projection from patch embeddings back to input
        self.output_projection = nn.Linear(config.patch_embed_dim, config.input_dim)
        
        # NaN detection hook storage
        self._nan_hooks = []
        self._nan_detected = False
        self._nan_info = {}
    
    def encode(self, x: torch.Tensor, y: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None, time_tensor: Optional[torch.Tensor] = None) -> Dict:
        """
        Encode input to latent codes.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
            time_tensor: Optional tensor of shape (batch_size, seq_len) containing time differences

        Returns:
            Dictionary containing:
                - z: Latent embeddings before quantization
                - z_q: Quantized latent embeddings
                - indices: Codebook indices
                - loss: VQ loss
                - perplexity: Codebook perplexity
                - classification_logits: Classification logits, or None
                - classification_loss: Classification loss, or None
        """
        # Apply patch embedding first
        x_patch = self.patch_embedding(x)
        
        # Encode to latent space - always returns (z, cls_logits, cls_loss)
        z, cls_logits, cls_loss = self.encoder(x_patch, y, mask=mask, time_tensor=time_tensor)
        
        # Quantize
        quantization_output = self.vector_quantizer(z)
        z_q = quantization_output["quantized"]
        
        return {
            "z": z,
            "z_q": z_q,
            "indices": quantization_output["encoding_indices"],
            "loss": quantization_output["loss"],
            "perplexity": quantization_output["perplexity"],
            "encodings": quantization_output["encodings"],
            "classification_logits": cls_logits,
            "classification_loss": cls_loss,
            "x_patch": x_patch,
        }
    
    def decode(
        self,
        z_q: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode quantized latent codes to reconstruction at patch level.
        
        Args:
            z_q: Quantized latent tensor of shape (batch_size, seq_len, embedding_dim)
            mask: Optional attention mask
        
        Returns:
            Patch-level reconstructed tensor of shape (batch_size, num_patches, patch_embed_dim)
        """
        # Decode: (batch, num_patches, embedding_dim) -> (batch, num_patches, patch_embed_dim)
        patch_recon = self.decoder(z_q, self_attn_mask=mask)
        
        # Return patch-level reconstruction for loss computation
        return patch_recon
    
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        time_tensor: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass through VQ-VAE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            y: Optional target tensor of shape (batch_size,) for classification
            mask: Optional attention mask
            time_tensor: Optional tensor of shape (batch_size, seq_len) containing time differences
        
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed output
                - z: Latent embeddings before quantization
                - z_q: Quantized latent embeddings
                - indices: Codebook indices
                - total_loss: Total training loss (reconstruction + VQ + classification)
                - reconstruction_loss: MSE reconstruction loss
                - vq_loss: Vector quantization loss
                - perplexity: Codebook perplexity
                - classification_logits: Classification logits, or None
                - classification_loss: Classification loss, or None
        """
        # Encode and quantize
        encode_output = self.encode(x, y, mask=mask, time_tensor=time_tensor)
        z_q = encode_output["z_q"]
        x_patch = encode_output["x_patch"]
        
        # Decode
        reconstruction = self.decode(z_q, mask=mask)
        
        # Compute reconstruction loss at patch level
        # Both reconstruction and x_patch are shape (batch, num_patches, patch_embed_dim)
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x_patch)
        vq_loss = encode_output["loss"]
        
        # Compute total loss including classification if available
        total_loss = (
            self.config.reconstruction_loss_weight * reconstruction_loss
            + self.config.commitment_loss_weight * vq_loss
        )
        
        # Add classification loss if it exists
        classification_loss = encode_output["classification_loss"]
        if classification_loss is not None:
            # Default weight for classification loss (1.0)
            classification_weight = getattr(self.config, 'classification_loss_weight', 1.0)
            total_loss = total_loss + classification_weight * classification_loss
        
        return {
            "reconstruction": reconstruction,
            "z": encode_output["z"],
            "z_q": z_q,
            "indices": encode_output["indices"],
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "perplexity": encode_output["perplexity"],
            "encodings": encode_output["encodings"],
            "classification_logits": encode_output["classification_logits"],
            "classification_loss": classification_loss,
        }
    
    def reconstruct_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct output directly from codebook indices.
        Useful for inference and generation.
        
        Args:
            indices: Tensor of codebook indices
        
        Returns:
            Reconstructed output
        """
        # Decode indices to embeddings
        z_q = self.vector_quantizer.decode(indices)
        # Decode embeddings to output
        reconstruction = self.decode(z_q)
        return reconstruction
    
    def get_codebook(self) -> torch.Tensor:
        """Get the current codebook embeddings."""
        return self.vector_quantizer.embeddings.weight.data.clone()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def freeze_decoder(self):
        """Freeze decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = False
    
    def freeze_quantizer(self):
        """Freeze quantizer parameters."""
        for param in self.vector_quantizer.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters_count(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def _get_all_submodules(self, module: nn.Module, prefix: str = "") -> Dict[str, nn.Module]:
        """
        Recursively get all submodules with their full paths.
        
        Args:
            module: Root module to traverse
            prefix: Path prefix for current module
        
        Returns:
            Dictionary mapping full module paths to modules
        """
        submodules = {}
        
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            submodules[full_name] = child
            
            # Recursively get submodules of children
            submodules.update(self._get_all_submodules(child, full_name))
        
        return submodules
    
    def _create_nan_hook(self, layer_name: str, hook_type: str = "both"):
        """
        Create a hook function to detect NaN and Inf values.
        
        Args:
            layer_name: Full path name of the layer for logging
            hook_type: "pre" (inputs), "post" (outputs), or "both"
        
        Returns:
            Hook function
        """
        def pre_hook(module, inputs):
            """Check for NaN/Inf in inputs."""
            if hook_type in ["pre", "both"]:
                for i, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor):
                        nan_count = torch.isnan(inp).sum().item()
                        inf_count = torch.isinf(inp).sum().item()
                        
                        if nan_count > 0 or inf_count > 0:
                            self._nan_detected = True
                            key = f"{layer_name}::input_{i}"
                            self._nan_info[key] = {
                                "stage": "pre-forward",
                                "shape": tuple(inp.shape),
                                "nan_count": nan_count,
                                "inf_count": inf_count,
                                "total_elements": inp.numel(),
                                "dtype": str(inp.dtype),
                            }
                            print(f"🔴 NaN/Inf DETECTED in {layer_name} INPUT {i}")
                            print(f"   Shape: {inp.shape}, dtype: {inp.dtype}")
                            print(f"   NaN count: {nan_count}/{inp.numel()}, Inf count: {inf_count}/{inp.numel()}")
                            if nan_count > 0 or inf_count > 0:
                                print(f"   Min: {inp[~torch.isnan(inp) & ~torch.isinf(inp)].min() if (nan_count + inf_count) < inp.numel() else 'all NaN/Inf'}")
                                print(f"   Max: {inp[~torch.isnan(inp) & ~torch.isinf(inp)].max() if (nan_count + inf_count) < inp.numel() else 'all NaN/Inf'}")
        
        def post_hook(module, inputs, outputs):
            """Check for NaN/Inf in outputs."""
            if hook_type in ["post", "both"]:
                if isinstance(outputs, torch.Tensor):
                    nan_count = torch.isnan(outputs).sum().item()
                    inf_count = torch.isinf(outputs).sum().item()
                    
                    if nan_count > 0 or inf_count > 0:
                        self._nan_detected = True
                        key = f"{layer_name}::output"
                        self._nan_info[key] = {
                            "stage": "post-forward",
                            "shape": tuple(outputs.shape),
                            "nan_count": nan_count,
                            "inf_count": inf_count,
                            "total_elements": outputs.numel(),
                            "dtype": str(outputs.dtype),
                        }
                        print(f"🔴 NaN/Inf DETECTED in {layer_name} OUTPUT")
                        print(f"   Shape: {outputs.shape}, dtype: {outputs.dtype}")
                        print(f"   NaN count: {nan_count}/{outputs.numel()}, Inf count: {inf_count}/{outputs.numel()}")
                        if nan_count + inf_count < outputs.numel():
                            valid_vals = outputs[~torch.isnan(outputs) & ~torch.isinf(outputs)]
                            if valid_vals.numel() > 0:
                                print(f"   Min: {valid_vals.min():.6e}, Max: {valid_vals.max():.6e}, Mean: {valid_vals.mean():.6e}")
                
                elif isinstance(outputs, (tuple, list)):
                    for i, out in enumerate(outputs):
                        if isinstance(out, torch.Tensor):
                            nan_count = torch.isnan(out).sum().item()
                            inf_count = torch.isinf(out).sum().item()
                            
                            if nan_count > 0 or inf_count > 0:
                                self._nan_detected = True
                                key = f"{layer_name}::output_{i}"
                                self._nan_info[key] = {
                                    "stage": "post-forward",
                                    "shape": tuple(out.shape),
                                    "nan_count": nan_count,
                                    "inf_count": inf_count,
                                    "total_elements": out.numel(),
                                    "dtype": str(out.dtype),
                                }
                                print(f"🔴 NaN/Inf DETECTED in {layer_name} OUTPUT {i}")
                                print(f"   Shape: {out.shape}, dtype: {out.dtype}")
                                print(f"   NaN count: {nan_count}/{out.numel()}, Inf count: {inf_count}/{out.numel()}")
        
        if hook_type == "pre":
            return pre_hook
        elif hook_type == "post":
            return post_hook
        else:  # "both"
            def combined_hook(module, inputs, outputs=None):
                if outputs is None:
                    pre_hook(module, inputs)
                else:
                    post_hook(module, inputs, outputs)
            return combined_hook
    
    def register_nan_detection_hooks(self, layer_names: Optional[list] = None, deep: bool = True):
        """
        Register NaN detection hooks on specified layers and their submodules.
        
        Args:
            layer_names: List of top-level layer names to monitor. If None, monitors:
                        ["patch_embedding", "encoder", "vector_quantizer", "decoder", "output_projection"]
            deep: If True (default), recursively register hooks on all submodules.
                 If False, only register on top-level layers.
        
        Example:
            >>> model.register_nan_detection_hooks()  # Deep monitoring of all modules
            >>> model.register_nan_detection_hooks(["encoder", "decoder"])  # Deep monitoring of specific modules
            >>> model.register_nan_detection_hooks(deep=False)  # Shallow monitoring (top-level only)
        """
        if layer_names is None:
            layer_names = ["patch_embedding", "encoder", "vector_quantizer", "decoder", "output_projection"]
        
        # Clear existing hooks
        self.remove_nan_detection_hooks()
        
        for layer_name in layer_names:
            if hasattr(self, layer_name):
                layer = getattr(self, layer_name)
                
                if deep:
                    # Get all submodules recursively with their full paths
                    all_modules = {layer_name: layer}
                    all_modules.update(self._get_all_submodules(layer, layer_name))
                    
                    # Register hooks on all modules
                    for full_path, module in all_modules.items():
                        pre_hook = module.register_forward_pre_hook(
                            lambda m, inp, name=full_path: self._create_nan_hook(name, "pre")(m, inp)
                        )
                        post_hook = module.register_forward_hook(
                            lambda m, inp, out, name=full_path: self._create_nan_hook(name, "post")(m, inp, out)
                        )
                        self._nan_hooks.append(pre_hook)
                        self._nan_hooks.append(post_hook)
                    
                    num_modules = len(all_modules)
                    print(f"✓ Deep NaN detection enabled for {layer_name}: {num_modules} module(s) monitored")
                else:
                    # Register only on the top-level layer
                    pre_hook = layer.register_forward_pre_hook(
                        lambda m, inp, name=layer_name: self._create_nan_hook(name, "pre")(m, inp)
                    )
                    post_hook = layer.register_forward_hook(
                        lambda m, inp, out, name=layer_name: self._create_nan_hook(name, "post")(m, inp, out)
                    )
                    self._nan_hooks.append(pre_hook)
                    self._nan_hooks.append(post_hook)
                    print(f"✓ Shallow NaN detection enabled for {layer_name}")
        
        hook_count = len(self._nan_hooks)
        print(f"✓ Total hooks registered: {hook_count}")
    
    def remove_nan_detection_hooks(self):
        """Remove all registered NaN detection hooks."""
        for hook in self._nan_hooks:
            hook.remove()
        self._nan_hooks.clear()
        print("✓ NaN detection hooks removed")
    
    def reset_nan_detection(self):
        """Reset NaN detection state."""
        self._nan_detected = False
        self._nan_info = {}
    
    def get_nan_detection_report(self) -> Dict:
        """
        Get a report of any NaN values detected.
        
        Returns:
            Dictionary with detection status and details
        """
        return {
            "nan_detected": self._nan_detected,
            "nan_count": len(self._nan_info),
            "details": self._nan_info,
        }
    
    def check_numerical_stability(self) -> Dict:
        """
        Check model parameters and buffers for NaN/Inf values.
        
        Returns:
            Dictionary with stability status and issues found
        """
        issues = {
            "has_nan_params": False,
            "has_inf_params": False,
            "has_nan_grads": False,
            "has_inf_grads": False,
            "nan_param_details": [],
            "inf_param_details": [],
            "nan_grad_details": [],
            "inf_grad_details": [],
        }
        
        for name, param in self.named_parameters():
            # Check weights
            if torch.isnan(param).any():
                issues["has_nan_params"] = True
                issues["nan_param_details"].append(name)
                print(f"⚠️  NaN found in parameter: {name}")
            if torch.isinf(param).any():
                issues["has_inf_params"] = True
                issues["inf_param_details"].append(name)
                print(f"⚠️  Inf found in parameter: {name}")
            
            # Check gradients
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    issues["has_nan_grads"] = True
                    issues["nan_grad_details"].append(name)
                    print(f"⚠️  NaN found in gradient: {name}")
                if torch.isinf(param.grad).any():
                    issues["has_inf_grads"] = True
                    issues["inf_grad_details"].append(name)
                    print(f"⚠️  Inf found in gradient: {name}")
        
        return issues
    
    def clamp_tensor_values(self, max_val: float = 1e4):
        """
        Clamp all tensor values to prevent numerical explosion.
        Useful as a safety measure during training.
        
        Args:
            max_val: Maximum absolute value to clamp to
        """
        for param in self.parameters():
            if param.requires_grad:
                param.data = torch.clamp(param.data, -max_val, max_val)
    
    def apply_activation_clamping(self, model_or_tensor, min_val: float = -1e3, max_val: float = 1e3):
        """
        Clamp activation values to prevent explosions.
        Can be used as a debugging tool.
        
        Args:
            model_or_tensor: Model or tensor to clamp
            min_val: Minimum value
            max_val: Maximum value
        """
        if isinstance(model_or_tensor, torch.Tensor):
            return torch.clamp(model_or_tensor, min_val, max_val)
        else:
            for param in model_or_tensor.parameters():
                param.data = torch.clamp(param.data, min_val, max_val)
    
    def summary(self) -> str:
        """Get model summary."""
        total_params = self.get_total_parameters_count()
        trainable_params = self.get_trainable_parameters_count()
        
        lines = [
            "\n" + "="*60,
            "VQ-VAE Model Summary (Streamlined Architecture)",
            "="*60,
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            f"Non-trainable Parameters: {total_params - trainable_params:,}",
            "-"*60,
            "Input and Patch Embedding:",
            f"  Input Dimension: {self.config.input_dim}",
            f"  Patch Size: {self.config.patch_size}, Stride: {self.config.patch_stride}",
            f"  Patch Embed Dimension: {self.config.patch_embed_dim}",
            "-"*60,
            "Vector Quantization:",
            f"  Codebook Size: {self.config.num_embeddings}",
            f"  Embedding Dimension: {self.config.embedding_dim}",
            f"  Commitment Cost: {self.config.commitment_cost}",
            "-"*60,
            "Transformer (Unified for Encoder & Decoder):",
            f"  Hidden Dimension: {self.config.hidden_dim}",
            f"  Number of Layers: {self.config.num_layers}",
            f"  Number of Heads: {self.config.num_heads}",
            f"  FF Dimension: {self.config.ff_dim} (hidden_dim × {self.config.ff_multiplier})",
            f"  Dropout: {self.config.dropout}",
            "-"*60,
            "Classification Head:" if self.config.use_class_token else "Classification Head: Disabled",
        ]
        
        if self.config.use_class_token:
            lines.append(f"  Projection Dimension: {self.config.class_proj_dim}")
        
        lines.extend([
            "-"*60,
            f"Loss Weights:",
            f"  Reconstruction: {self.config.reconstruction_loss_weight}",
            f"  VQ Commitment: {self.config.commitment_loss_weight}",
            f"  Classification: {self.config.classification_loss_weight}",
            "="*60 + "\n",
        ])
        return "\n".join(lines)
