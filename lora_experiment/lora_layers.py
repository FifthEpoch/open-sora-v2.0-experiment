"""
LoRA (Low-Rank Adaptation) layers for Open-Sora v2.0 MMDiT model.

This module provides LoRA injection for the MMDiT (Multimodal Diffusion Transformer)
architecture used in Open-Sora v2.0. 

The MMDiT has:
- DoubleStreamBlock: Separate attention for image and text with img_attn, txt_attn, img_mlp, txt_mlp
- SingleStreamBlock: Combined attention with fused QKV projection

LoRA adds trainable low-rank matrices to frozen model weights:
    h = W0*x + (B @ A) * x * scaling
    
Where:
    - W0: Original frozen weights
    - A: Low-rank down-projection (d x r)
    - B: Low-rank up-projection (r x d)  
    - r: Rank (much smaller than d)
    - scaling: alpha / r

Reference: https://arxiv.org/abs/2106.09685
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA layer that wraps a linear layer with low-rank adaptation.
    
    Args:
        original_layer: The original nn.Linear layer to wrap
        rank: Rank of the low-rank matrices (default: 8)
        alpha: Scaling factor (default: 16)
        dropout: Dropout probability for LoRA layers (default: 0.0)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Get device and dtype from original layer
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
        
        # Freeze the original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Create LoRA matrices ON THE SAME DEVICE as the original layer
        # A: down projection (in_features -> rank)
        # B: up projection (rank -> out_features)
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA weights using Kaiming uniform for A and zeros for B."""
        # Store device and dtype to preserve them
        device = self.lora_A.device
        dtype = self.lora_A.dtype
        
        # Initialize in-place while preserving device/dtype
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            
            # Ensure parameters stay on correct device (safety check)
            if self.lora_A.device != device:
                self.lora_A.data = self.lora_A.data.to(device, dtype)
            if self.lora_B.device != device:
                self.lora_B.data = self.lora_B.data.to(device, dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        result = self.original_layer(x)
        
        # Ensure LoRA weights are on the same device as input
        if self.lora_A.device != x.device:
            self.lora_A.data = self.lora_A.data.to(x.device, x.dtype)
            self.lora_B.data = self.lora_B.data.to(x.device, x.dtype)
        
        # LoRA forward pass
        # x @ A^T @ B^T * scaling
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # (batch, seq, rank)
        lora_out = F.linear(lora_out, self.lora_B)  # (batch, seq, out_features)
        lora_out = lora_out * self.scaling
        
        return result + lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into the original layer for inference."""
        with torch.no_grad():
            # W_merged = W0 + B @ A * scaling
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_w
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from the original layer."""
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data -= delta_w


class LoRAFusedQKV(nn.Module):
    """
    LoRA wrapper for fused QKV projection layer in MMDiT.
    
    Applies separate LoRA adapters to Q, K, V projections within a fused QKV layer.
    """
    
    def __init__(
        self,
        original_qkv: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        enable_lora: List[bool] = [True, True, True],  # [Q, K, V]
    ):
        super().__init__()
        
        self.original_qkv = original_qkv
        self.in_features = original_qkv.in_features
        self.out_features = original_qkv.out_features
        
        # For fused QKV, out_features should be 3x hidden_size
        self.hidden_size = self.out_features // 3
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.enable_lora = enable_lora
        
        # Get device and dtype from original layer
        device = original_qkv.weight.device
        dtype = original_qkv.weight.dtype
        
        # Freeze the original layer
        for param in self.original_qkv.parameters():
            param.requires_grad = False
        
        # Create LoRA matrices for Q, K, V separately ON THE SAME DEVICE
        self.lora_A_q = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype)) if enable_lora[0] else None
        self.lora_B_q = nn.Parameter(torch.zeros(self.hidden_size, rank, device=device, dtype=dtype)) if enable_lora[0] else None
        
        self.lora_A_k = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype)) if enable_lora[1] else None
        self.lora_B_k = nn.Parameter(torch.zeros(self.hidden_size, rank, device=device, dtype=dtype)) if enable_lora[1] else None
        
        self.lora_A_v = nn.Parameter(torch.zeros(rank, self.in_features, device=device, dtype=dtype)) if enable_lora[2] else None
        self.lora_B_v = nn.Parameter(torch.zeros(self.hidden_size, rank, device=device, dtype=dtype)) if enable_lora[2] else None
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA weights."""
        with torch.no_grad():
            for A, B in [(self.lora_A_q, self.lora_B_q), 
                         (self.lora_A_k, self.lora_B_k),
                         (self.lora_A_v, self.lora_B_v)]:
                if A is not None:
                    device = A.device
                    dtype = A.dtype
                    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
                    nn.init.zeros_(B)
                    # Ensure parameters stay on correct device (safety check)
                    if A.device != device:
                        A.data = A.data.to(device, dtype)
                    if B.device != device:
                        B.data = B.data.to(device, dtype)
    
    def _ensure_device(self, x: torch.Tensor):
        """Ensure all LoRA weights are on the same device as input."""
        for A, B in [(self.lora_A_q, self.lora_B_q), 
                     (self.lora_A_k, self.lora_B_k),
                     (self.lora_A_v, self.lora_B_v)]:
            if A is not None and A.device != x.device:
                A.data = A.data.to(x.device, x.dtype)
                B.data = B.data.to(x.device, x.dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure LoRA weights are on the same device as input
        self._ensure_device(x)
        
        # Original QKV projection
        qkv = self.original_qkv(x)
        
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Apply LoRA to each
        x_dropped = self.dropout(x)
        
        if self.lora_A_q is not None:
            q = q + F.linear(F.linear(x_dropped, self.lora_A_q), self.lora_B_q) * self.scaling
        
        if self.lora_A_k is not None:
            k = k + F.linear(F.linear(x_dropped, self.lora_A_k), self.lora_B_k) * self.scaling
        
        if self.lora_A_v is not None:
            v = v + F.linear(F.linear(x_dropped, self.lora_A_v), self.lora_B_v) * self.scaling
        
        # Concatenate back
        return torch.cat([q, k, v], dim=-1)


def inject_lora_into_self_attention(
    attn_module: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: List[str] = ["qkv", "proj"],
) -> Dict[str, nn.Module]:
    """
    Inject LoRA layers into a SelfAttention module.
    
    Args:
        attn_module: The SelfAttention module
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout for LoRA layers
        target_modules: Which modules to apply LoRA to
    
    Returns:
        Dict of original modules replaced by LoRA versions
    """
    lora_modules = {}
    
    # Handle fused QKV
    if "qkv" in target_modules and hasattr(attn_module, "qkv") and attn_module.fused_qkv:
        original_qkv = attn_module.qkv
        lora_qkv = LoRAFusedQKV(original_qkv, rank=rank, alpha=alpha, dropout=dropout)
        attn_module.qkv = lora_qkv
        lora_modules["qkv"] = lora_qkv
    
    # Handle separate Q, K, V projections
    if not getattr(attn_module, "fused_qkv", True):
        for name in ["q_proj", "k_proj", "v_proj"]:
            if name in target_modules and hasattr(attn_module, name):
                original = getattr(attn_module, name)
                lora = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                setattr(attn_module, name, lora)
                lora_modules[name] = lora
    
    # Handle output projection
    if "proj" in target_modules and hasattr(attn_module, "proj"):
        original_proj = attn_module.proj
        lora_proj = LoRALinear(original_proj, rank=rank, alpha=alpha, dropout=dropout)
        attn_module.proj = lora_proj
        lora_modules["proj"] = lora_proj
    
    return lora_modules


def inject_lora_into_mlp(
    mlp_module: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
) -> Dict[str, nn.Module]:
    """
    Inject LoRA layers into an MLP module (nn.Sequential with Linear layers).
    
    Args:
        mlp_module: The MLP module (nn.Sequential)
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout for LoRA layers
    
    Returns:
        Dict of original modules replaced by LoRA versions
    """
    lora_modules = {}
    
    # MLP in MMDiT is nn.Sequential with [Linear, GELU, Linear]
    for idx, layer in enumerate(mlp_module):
        if isinstance(layer, nn.Linear):
            lora = LoRALinear(layer, rank=rank, alpha=alpha, dropout=dropout)
            mlp_module[idx] = lora
            lora_modules[f"linear_{idx}"] = lora
    
    return lora_modules


def inject_lora_into_mmdit(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: List[str] = ["qkv", "proj"],
    target_blocks: str = "all",  # "all", "double", "single"
    target_mlp: bool = False,
) -> Dict[str, Dict[str, nn.Module]]:
    """
    Inject LoRA layers into MMDiT model.
    
    Args:
        model: MMDiT model
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout for LoRA layers
        target_modules: Which attention modules to apply LoRA to
        target_blocks: Which blocks to apply LoRA to
        target_mlp: Whether to also target MLP layers
    
    Returns:
        Dict mapping block names to their LoRA modules
    """
    all_lora_modules = {}
    
    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradient computation for input to blocks (needed for LoRA)
    if hasattr(model, "_input_requires_grad"):
        model._input_requires_grad = True
    
    # Inject LoRA into double stream blocks
    if target_blocks in ["all", "double"]:
        if hasattr(model, "double_blocks"):
            for i, block in enumerate(model.double_blocks):
                block_name = f"double_block_{i}"
                all_lora_modules[block_name] = {}
                
                # Image attention
                if hasattr(block, "img_attn"):
                    lora_mods = inject_lora_into_self_attention(
                        block.img_attn, rank=rank, alpha=alpha, 
                        dropout=dropout, target_modules=target_modules
                    )
                    all_lora_modules[block_name]["img_attn"] = lora_mods
                
                # Text attention
                if hasattr(block, "txt_attn"):
                    lora_mods = inject_lora_into_self_attention(
                        block.txt_attn, rank=rank, alpha=alpha,
                        dropout=dropout, target_modules=target_modules
                    )
                    all_lora_modules[block_name]["txt_attn"] = lora_mods
                
                # Image MLP
                if target_mlp and hasattr(block, "img_mlp"):
                    lora_mlp_mods = inject_lora_into_mlp(
                        block.img_mlp, rank=rank, alpha=alpha, dropout=dropout
                    )
                    all_lora_modules[block_name]["img_mlp"] = lora_mlp_mods
                
                # Text MLP
                if target_mlp and hasattr(block, "txt_mlp"):
                    lora_mlp_mods = inject_lora_into_mlp(
                        block.txt_mlp, rank=rank, alpha=alpha, dropout=dropout
                    )
                    all_lora_modules[block_name]["txt_mlp"] = lora_mlp_mods
    
    # Inject LoRA into single stream blocks
    if target_blocks in ["all", "single"]:
        if hasattr(model, "single_blocks"):
            for i, block in enumerate(model.single_blocks):
                block_name = f"single_block_{i}"
                all_lora_modules[block_name] = {}
                
                # Handle fused linear1 (QKV + MLP input)
                # SingleStreamBlock has linear1 that outputs [QKV, MLP_in]
                if hasattr(block, "linear1"):
                    original = block.linear1
                    lora = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                    block.linear1 = lora
                    all_lora_modules[block_name]["linear1"] = {"linear1": lora}
                
                # Handle separate projections if not fused
                if not getattr(block, "fused_qkv", True):
                    for name in ["q_proj", "k_proj"]:
                        if hasattr(block, name):
                            original = getattr(block, name)
                            lora = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                            setattr(block, name, lora)
                            if "attn" not in all_lora_modules[block_name]:
                                all_lora_modules[block_name]["attn"] = {}
                            all_lora_modules[block_name]["attn"][name] = lora
                    
                    if hasattr(block, "v_mlp"):
                        original = block.v_mlp
                        lora = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                        block.v_mlp = lora
                        all_lora_modules[block_name]["v_mlp"] = {"v_mlp": lora}
                
                # Output linear2
                if hasattr(block, "linear2"):
                    original = block.linear2
                    lora = LoRALinear(original, rank=rank, alpha=alpha, dropout=dropout)
                    block.linear2 = lora
                    all_lora_modules[block_name]["linear2"] = {"linear2": lora}
    
    return all_lora_modules


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters from a model."""
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> Dict[str, int]:
    """Count LoRA and total parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "lora": lora_params,
        "frozen": total_params - trainable_params,
        "trainable_pct": 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def save_lora_weights(model: nn.Module, path: str):
    """Save only the LoRA weights to a file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_state_dict[name] = param.data.clone().cpu()
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str, strict: bool = True):
    """Load LoRA weights from a file."""
    lora_state_dict = torch.load(path, map_location="cpu")
    
    model_state = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state:
            model_state[name].copy_(param)
        elif strict:
            raise KeyError(f"LoRA weight {name} not found in model")
    
    model.load_state_dict(model_state)


def reset_lora_weights(model: nn.Module, device: torch.device = None):
    """
    Reset all LoRA weights to initial values.
    
    Args:
        model: The model containing LoRA modules
        device: Device to move LoRA weights to (if None, inferred from model)
    """
    # Infer device from model if not specified
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Move to device before reset to ensure proper initialization
            module.lora_A.data = module.lora_A.data.to(device)
            module.lora_B.data = module.lora_B.data.to(device)
            module.reset_lora_parameters()
        elif isinstance(module, LoRAFusedQKV):
            # Move to device before reset
            for A, B in [(module.lora_A_q, module.lora_B_q), 
                         (module.lora_A_k, module.lora_B_k),
                         (module.lora_A_v, module.lora_B_v)]:
                if A is not None:
                    A.data = A.data.to(device)
                    B.data = B.data.to(device)
            module.reset_lora_parameters()


def merge_lora_weights(model: nn.Module):
    """Merge all LoRA weights into base model for faster inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()


def unmerge_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights from base model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge_weights()

