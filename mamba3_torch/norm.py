# Copyright (c) 2026, Dao AI Lab, Goombalab.
# Pure-PyTorch adaptation; Apache-2.0, see _official/LICENSE.
"""Gated RMSNorm with the official reference's float32 normalization."""

import torch
from torch import nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    def __init__(
        self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True,
        device=None, dtype=None,
    ):
        super().__init__()
        if group_size is not None and (group_size <= 0 or hidden_size % group_size):
            raise ValueError("group_size must be a positive divisor of hidden_size")
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x, z=None):
        dtype = x.dtype
        # The official upcast=True oracle uses float32 even for float64 inputs.
        x = x.float()
        z = z.float() if z is not None else None
        weight = self.weight.float()
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z)
        if self.group_size is None:
            rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
            out = x * rstd * weight
        else:
            grouped = x.reshape(*x.shape[:-1], -1, self.group_size)
            rstd = 1 / torch.sqrt(grouped.square().mean(dim=-1, keepdim=True) + self.eps)
            out = (grouped * rstd).reshape_as(x) * weight
        if z is not None and self.norm_before_gate:
            out = out * F.silu(z)
        return out.to(dtype)
