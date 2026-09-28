# Copyright (c) 2026, Dao AI Lab, Goombalab.
# SISO port of state-spaces/mamba e9594ce; Apache-2.0, see _official/LICENSE.
"""Mamba-3 modules with explicit, differentiable recurrent states."""

import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .norm import RMSNormGated
from .ops import _compute_dtype, _initial_states, mamba3_siso_chunked, mamba3_siso_step


def heavy_tail_activation(x):
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


class Mamba3(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        is_outproj_norm=False,
        is_mimo=False,
        mimo_rank=4,
        fuse_pregate_headwise_norm=True,
        chunk_size=64,
        dropout=0.0,
        layer_idx=None,
        n_layer=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        if is_mimo:
            raise NotImplementedError("MIMO not ported yet")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.is_mimo = is_mimo
        self.mimo_rank = 1
        self.fuse_pregate_headwise_norm = False
        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        if ngroups <= 0 or self.nheads % ngroups:
            raise ValueError("ngroups must divide the number of heads")
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2 / rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        d_in_proj = (
            2 * self.d_inner + 2 * self.d_state * self.num_bc_heads * self.mimo_rank
            + 3 * self.nheads + self.num_rope_angles
        )
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True
        self.B_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device),
        )
        self.C_bias = nn.Parameter(
            1 + torch.zeros((self.nheads, self.mimo_rank, self.d_state), dtype=torch.float32, device=device),
        )
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True
        if self.is_outproj_norm:
            self.norm = RMSNormGated(
                self.d_inner, eps=1e-5, norm_before_gate=True,
                group_size=self.headdim, **factory_kwargs,
            )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def _project(self, u):
        """The official forward preprocessing, shared with single-token decoding."""
        zxBCdtAtrap = self.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            zxBCdtAtrap,
            [
                self.d_inner, self.d_inner,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.d_state * self.num_bc_heads * self.mimo_rank,
                self.nheads, self.nheads, self.nheads, self.num_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")
        _A = -heavy_tail_activation(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)
        B = self.B_norm(B)
        C = self.C_norm(C)
        return z, x, B.squeeze(2), C.squeeze(2), ADT, DT, trap, angles

    def _output(self, y, z, dtype):
        y = y.flatten(-2)
        if self.is_outproj_norm:
            y = self.norm(y, z.flatten(-2))
        return self.out_proj(y.to(dtype))

    def forward(self, u, initial_states=None, return_final_states=False):
        """Map (batch, length, d_model) to outputs and optionally final states."""
        z, x, B, C, ADT, DT, trap, angles = self._project(u)
        result = mamba3_siso_chunked(
            Q=C, K=B, V=x, ADT=ADT, DT=DT, Trap=trap,
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles, D=self.D, Z=z if not self.is_outproj_norm else None,
            initial_states=initial_states, chunk_size=self.chunk_size,
            return_final_states=return_final_states,
        )
        if return_final_states:
            y, states = result
            return self._output(y, z, x.dtype), states
        return self._output(result, z, x.dtype)

    def step(self, u_t, states):
        """Decode (batch, d_model), returning outputs and new states."""
        z, x, B, C, ADT, DT, trap, angles = self._project(u_t.unsqueeze(1))
        y, states = mamba3_siso_step(
            q=C[:, 0], k=B[:, 0], v=x[:, 0], adt=ADT[..., 0], dt=DT[..., 0],
            trap=trap[..., 0], q_bias=self.C_bias.squeeze(1), k_bias=self.B_bias.squeeze(1),
            angles=angles[:, 0], D=self.D, z=z[:, 0] if not self.is_outproj_norm else None,
            states=states,
        )
        return self._output(y, z[:, 0], x.dtype), states

    def allocate_states(self, batch, device=None, dtype=None):
        """Allocate zero states, promoting half/bfloat16 storage to float32."""
        device = self.in_proj.weight.device if device is None else device
        dtype = self.in_proj.weight.dtype if dtype is None else dtype
        return _initial_states(
            None, batch, self.nheads, self.headdim, self.d_state,
            self.num_rope_angles, device, _compute_dtype(dtype),
        )


class Mamba3Block(nn.Module):
    """Pre-normalized Mamba-3 residual block."""

    def __init__(self, d_model, **mamba_kwargs):
        super().__init__()
        self.norm = RMSNormGated(
            d_model, device=mamba_kwargs.get("device"), dtype=mamba_kwargs.get("dtype"),
        )
        self.mamba = Mamba3(d_model, **mamba_kwargs)

    def forward(self, u, initial_states=None, return_final_states=False):
        result = self.mamba(self.norm(u), initial_states, return_final_states)
        if return_final_states:
            y, states = result
            return u + y, states
        return u + result

    def step(self, u_t, states):
        y, states = self.mamba.step(self.norm(u_t), states)
        return u_t + y, states

    def allocate_states(self, batch, device=None, dtype=None):
        return self.mamba.allocate_states(batch, device, dtype)


class Mamba3Stack(nn.Module):
    """A sequence of residual Mamba-3 blocks with a final RMSNorm."""

    def __init__(self, d_model, n_layer, **mamba_kwargs):
        super().__init__()
        if n_layer < 1:
            raise ValueError("n_layer must be positive")
        self.layers = nn.ModuleList(Mamba3Block(d_model, **mamba_kwargs) for _ in range(n_layer))
        self.norm = RMSNormGated(
            d_model, device=mamba_kwargs.get("device"), dtype=mamba_kwargs.get("dtype"),
        )

    def _states(self, states):
        if states is None:
            return [None] * len(self.layers)
        if len(states) != len(self.layers):
            raise ValueError("Expected one state tuple per layer")
        return states

    def forward(self, u, initial_states=None, return_final_states=False):
        states = self._states(initial_states)
        final_states = []
        for layer, state in zip(self.layers, states):
            result = layer(u, state, return_final_states)
            if return_final_states:
                u, state = result
                final_states.append(state)
            else:
                u = result
        out = self.norm(u)
        return (out, final_states) if return_final_states else out

    def step(self, u_t, states):
        final_states = []
        for layer, state in zip(self.layers, self._states(states)):
            u_t, state = layer.step(u_t, state)
            final_states.append(state)
        return self.norm(u_t), final_states

    def allocate_states(self, batch, device=None, dtype=None):
        return [layer.allocate_states(batch, device, dtype) for layer in self.layers]
