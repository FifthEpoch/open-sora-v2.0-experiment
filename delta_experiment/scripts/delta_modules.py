#!/usr/bin/env python3
"""
δ modules for SLOT-inspired test-time adaptation in Open-Sora v2.0.

These are designed to be sample-specific (reset per video) and lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from opensora.acceleration.checkpoint import auto_grad_checkpoint


@dataclass
class DeltaState:
    """
    Container for δ parameters for the three options.
    """

    # Option A: single vector in vec-space (hidden_size)
    delta_global: Optional[nn.Parameter] = None

    # Option B: grouped per-layer vectors in vec-space
    delta_double_groups: Optional[nn.ParameterList] = None
    delta_single_groups: Optional[nn.ParameterList] = None
    delta_final: Optional[nn.Parameter] = None
    n_groups_double: int = 0
    n_groups_single: int = 0

    # Option C: output correction vector in output space (out_channels)
    delta_out: Optional[nn.Parameter] = None


def make_delta_global(hidden_size: int, device: torch.device, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))


def make_delta_groups(n_groups: int, dim: int, device: torch.device, dtype: torch.dtype) -> nn.ParameterList:
    return nn.ParameterList([nn.Parameter(torch.zeros(dim, device=device, dtype=dtype)) for _ in range(n_groups)])


def make_delta_out(out_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.zeros(out_dim, device=device, dtype=dtype))


def group_index(i: int, n_blocks: int, n_groups: int) -> int:
    if n_groups <= 1:
        return 0
    # map i in [0, n_blocks-1] to [0, n_groups-1]
    return min(n_groups - 1, (i * n_groups) // max(1, n_blocks))


def forward_with_delta_groups(
    model,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    timesteps: Tensor,
    y_vec: Tensor,
    cond: Tensor,
    guidance: Tensor | None,
    delta_double_groups: nn.ParameterList,
    delta_single_groups: nn.ParameterList,
    delta_final: nn.Parameter,
    n_groups_double: int,
    n_groups_single: int,
    **kwargs,
) -> Tensor:
    """
    A custom forward that injects grouped per-layer δ vectors into the modulation vec.
    This mirrors MMDiTModel.forward_selective_ckpt/forward_ckpt but with per-block vec offsets.
    """

    img, txt, vec, pe = model.prepare_block_inputs(img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance)

    n_double = len(model.double_blocks)
    for i, block in enumerate(model.double_blocks):
        g = group_index(i, n_double, n_groups_double)
        vec_i = vec + delta_double_groups[g][None, :]
        img, txt = auto_grad_checkpoint(block, img, txt, vec_i, pe)

    img = torch.cat((txt, img), 1)
    n_single = len(model.single_blocks)
    for j, block in enumerate(model.single_blocks):
        g = group_index(j, n_single, n_groups_single)
        vec_j = vec + delta_single_groups[g][None, :]
        img = auto_grad_checkpoint(block, img, vec_j, pe)

    img = img[:, txt.shape[1] :, ...]
    img = model.final_layer(img, vec + delta_final[None, :])
    return img


class DeltaAWrapper:
    """
    Option A: patch model.prepare_block_inputs so vec is replaced with vec + δ.

    This is the most SLOT-like: a single additive vector in the global conditioning pathway.
    """

    def __init__(self, model):
        self.model = model
        self._orig_prepare = None

    def enable(self, delta_global: nn.Parameter):
        if self._orig_prepare is not None:
            return

        self._orig_prepare = self.model.prepare_block_inputs

        def wrapped_prepare(*args, **kwargs):
            img, txt, vec, pe = self._orig_prepare(*args, **kwargs)
            vec = vec + delta_global[None, :].to(vec.dtype)
            return img, txt, vec, pe

        self.model.prepare_block_inputs = wrapped_prepare

    def disable(self):
        if self._orig_prepare is None:
            return
        self.model.prepare_block_inputs = self._orig_prepare
        self._orig_prepare = None


def apply_output_delta(v_pred: Tensor, delta_out: nn.Parameter) -> Tensor:
    """
    Option C: output correction. Add a constant vector to every token prediction.
    v_pred: [B, L, out_dim]
    delta_out: [out_dim]
    """
    return v_pred + delta_out[None, None, :].to(v_pred.dtype)


