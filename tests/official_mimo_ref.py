"""Pure-torch reference oracles for Mamba-3 MIMO, copied VERBATIM from state-spaces/mamba (commit e9594ce,
2026-07-23, Apache-2.0): mamba3_MIMO_step_ref, apply_angle_dt_reference, mamba3_MIMO_chunk_ref, _pad_zeros from
tests/ops/tilelang/test_mamba3_mimo.py and compute_dacs_segsum_ref from mamba_ssm/ops/triton/mamba3/mamba3_mimo_utils.py.
DO NOT EDIT: this file is the parity oracle."""
import math
from typing import Optional, Tuple
import torch
from torch import Tensor
from einops import rearrange, repeat
F = torch.nn.functional

def _pad_zeros(t: Optional[Tensor], pad_len: int, dim: int) -> Optional[Tensor]:
    """Append ``pad_len`` zero-slices along ``dim``.  Returns ``t`` if pad_len==0."""
    if t is None or pad_len == 0:
        return t
    shape = list(t.shape)
    shape[dim] = pad_len
    return torch.cat([t, torch.zeros(shape, device=t.device, dtype=t.dtype)], dim=dim)


def mamba3_MIMO_step_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    MIMO_V: torch.Tensor,
    MIMO_O: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    MIMO_Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    fused_norm: bool = False,
    outproj_norm_weight: Optional[torch.Tensor] = None,
    outproj_norm_eps: float = 1e-5,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference implementation of Mamba-3 MIMO in recurrent (step) mode.

    Args:
        Input_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)

    Returns:
        out: Output tensor (batch, seqlen, nheads, headdim_v)
        Final_States: Tuple of (Angle_State, SSM_State, K_State, V_State)
    """
    batch, seqlen, mimo_rank, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0

    # Expand Q/K for GQA
    if Q.shape[3] != V.shape[2]:
        Q = repeat(Q, "b s r h_bc d -> b s r (h_bc g) d", g=V.shape[2] // Q.shape[3])
    if K.shape[3] != V.shape[2]:
        K = repeat(K, "b s r h_bc d -> b s r (h_bc g) d", g=V.shape[2] // K.shape[3])

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
        return rotated

    q_bias = rearrange(Q_bias, "h r d -> r h d")
    k_bias = rearrange(K_bias, "h r d -> r h d")

    # Initialize states
    if Input_States is not None:
        Angle_State, SSM_State, K_State, V_State = Input_States
        Angle_State = Angle_State.clone()
        SSM_State = SSM_State.clone().to(torch.float32)
        K_State = K_State.clone()
        V_State = V_State.clone()
    else:
        Angle_State = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=device)
        SSM_State = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device)
        K_State = torch.zeros((batch, nheads, mimo_rank, headdim_qk), dtype=Q.dtype, device=device)
        V_State = torch.zeros((batch, nheads, mimo_rank, headdim_v), dtype=V.dtype, device=device)

    # MIMO up project x and z:
    v_proj = torch.einsum("bthd,hrd->btrhd", V, MIMO_V)
    if Z is not None:
        z_proj = torch.einsum("bthd,hrd->btrhd", Z, MIMO_Z)
    else:
        z_proj = None

    TWO_PI = 2 * math.pi
    out_arr = []

    # Main SSM recurrence
    for idx in range(seqlen):
        q = Q[:, idx, :, :, :] + q_bias.unsqueeze(0)
        k = K[:, idx, :, :, :] + k_bias.unsqueeze(0)
        v = v_proj[:, idx, :, :, :] # (B R H P)
        adt = ADT[:, :, idx]
        dt = DT[:, :, idx]
        trap = torch.nn.functional.sigmoid(Trap[:, :, idx])
        z = z_proj[:, idx, :, :, :] if z_proj is not None else None
        angles = Angles[:, idx, :, :] # (B H N)

        q = q.permute(0, 2, 1, 3) # (B H R N)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        if z is not None:
            z = z.permute(0, 2, 1, 3)

        # Update angle state with cumsum: Angle_State = (Angle_State + Angles * DT) mod 2π
        # Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        # Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)
        Angle_State = Angle_State + torch.tanh(angles) * dt.unsqueeze(-1) * math.pi


        # Apply rotary embeddings to Q and K using cumulative angles
        cos_angles = torch.cos(Angle_State).unsqueeze(2) # (B H 1 N)
        sin_angles = torch.sin(Angle_State).unsqueeze(2)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        # Update SSM state using previous K_State and V_State
        prev_kv = torch.einsum("bhrd,bhrp->bhpd", K_State, V_State)
        curr_kv = torch.einsum("bhrd,bhrp->bhpd", k_rot, v)
        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * prev_kv
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * curr_kv

        # Compute output
        out = torch.einsum("bhpd,bhrd->bhrp", SSM_State, q_rot.to(SSM_State.dtype))

        if D is not None:
            out = out + D[None, :, None, None] * v

        if fused_norm:
            if z is None:
                raise ValueError("fused_norm=True requires Z and MIMO_Z.")
            out = out.float()
            out = out * torch.rsqrt(
                out.square().mean(dim=-1, keepdim=True) + outproj_norm_eps
            )
            if outproj_norm_weight is not None:
                out = out * outproj_norm_weight[None, :, None, :].float()
            out = out * torch.nn.functional.silu(z.float())
        elif z is not None:
            out = out * z * torch.sigmoid(z)

        out = torch.einsum("bhrp,hrp->bhp", out, MIMO_O)
        out_arr.append(out)

        # Update K and V states for next step
        K_State = k_rot
        V_State = v

    out = torch.stack(out_arr, dim=1)
    Final_States = (Angle_State, SSM_State, K_State, V_State)
    return out, Final_States


def apply_angle_dt_reference(
    angle: Tensor,  # (batch, seqlen, nheads, dim)
    dt: Tensor,     # (batch, seqlen, nheads)
) -> Tensor:
    # Match debug_mimo_step.py preprocessing for chunk reference path.
    base_vals = angle.to(torch.float32)
    base_vals = torch.tanh(base_vals) * dt[..., None].to(torch.float32) * torch.pi
    return torch.cumsum(base_vals, dim=1)


def mamba3_MIMO_chunk_ref(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_bias: Tensor,
    k_bias: Tensor,
    mimo_v: Tensor,
    mimo_o: Optional[Tensor],
    z: Optional[Tensor],
    mimo_z: Optional[Tensor],
    angles: Tensor,
    dA_cs: Tensor,
    dA_cs_rev: Tensor,
    dt: Tensor,
    trap: Tensor,
    D: Optional[Tensor],
    chunk_size: int = 64,
    rotary_dim_divisor: int = 4,
    return_final_state: bool = False,
    dtype: torch.dtype = torch.float32,
    rotate_pairwise: bool = False,
    contract_mimo_out: bool = True,
    cu_seqlens: Optional[Tensor] = None,
    fused_norm: bool = False,
    outproj_norm_weight: Optional[Tensor] = None,
    outproj_norm_eps: float = 1e-5,
) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    # Local copy of the reference program so tests remain valid even if module-level
    # debug/reference helpers are removed from shipped kernels.
    from einops import rearrange, repeat

    # --- Varlen path: loop per-sequence, delegate to the single-sequence path ---
    if cu_seqlens is not None:
        NS = cu_seqlens.shape[0] - 1
        out_parts = []
        for i in range(NS):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            out_i, _, _ = mamba3_MIMO_chunk_ref(
                q[:, start:end], k[:, start:end], v[:, start:end],
                q_bias, k_bias, mimo_v, mimo_o,
                z[:, start:end] if z is not None else None, mimo_z,
                angles[:, start:end],
                dA_cs[:, :, start:end], dA_cs_rev[:, :, start:end],
                dt[:, :, start:end], trap[:, :, start:end],
                D,
                chunk_size=chunk_size,
                rotary_dim_divisor=rotary_dim_divisor,
                return_final_state=False, dtype=dtype,
                rotate_pairwise=rotate_pairwise,
                contract_mimo_out=contract_mimo_out,
                cu_seqlens=None,
                fused_norm=fused_norm,
                outproj_norm_weight=outproj_norm_weight,
                outproj_norm_eps=outproj_norm_eps,
            )
            out_parts.append(out_i[:, :seq_len])
        return torch.cat(out_parts, dim=1), None, None

    if fused_norm:
        if not contract_mimo_out:
            raise ValueError("fused_norm=True requires contract_mimo_out=True.")
        if z is None or mimo_z is None:
            raise ValueError("fused_norm=True requires z and mimo_z.")

    # --- Single-sequence path ---
    # Pad to the next multiple of chunk_size so the chunked rearranges are valid
    # for sequences whose length is not a multiple of chunk_size.
    orig_seqlen = q.shape[1]
    pad_len = (chunk_size - orig_seqlen % chunk_size) % chunk_size
    if pad_len > 0:
        q         = _pad_zeros(q,         pad_len, dim=1)
        k         = _pad_zeros(k,         pad_len, dim=1)
        v         = _pad_zeros(v,         pad_len, dim=1)
        angles    = _pad_zeros(angles,    pad_len, dim=1)
        z         = _pad_zeros(z,         pad_len, dim=1)
        dA_cs     = _pad_zeros(dA_cs,     pad_len, dim=2)
        dA_cs_rev = _pad_zeros(dA_cs_rev, pad_len, dim=2)
        dt        = _pad_zeros(dt,        pad_len, dim=2)
        trap      = _pad_zeros(trap,      pad_len, dim=2)

    nchunks = q.shape[1] // chunk_size
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if z is not None:
        z = z.to(dtype)
        mimo_z = mimo_z.to(dtype)
    if D is not None:
        D = D.to(dtype)
    q_bias, k_bias = q_bias.to(dtype), k_bias.to(dtype)
    mimo_v = mimo_v.to(dtype)
    if contract_mimo_out:
        assert mimo_o is not None
        mimo_o = mimo_o.to(dtype)
    outproj_norm_weight = (
        outproj_norm_weight.float() if outproj_norm_weight is not None else None
    )
    if dA_cs is not None:
        dA_cs, dA_cs_rev = dA_cs.to(dtype), dA_cs_rev.to(dtype)
        dA_cs = rearrange(dA_cs, "b h (n c) -> b h n c", c=chunk_size)
        dA_cs_rev = rearrange(dA_cs_rev, "b h (n c) -> b h n c", c=chunk_size)

    batch, seqlen, mimo_rank, nheads_qk, dstate = q.shape
    nheads = v.shape[-2]
    if nheads_qk != nheads:
        q = repeat(q, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)
        k = repeat(k, "b s r h_qk d -> b s r (h_qk g) d", g=nheads // nheads_qk)

    angles = angles.to(dtype) if angles is not None else None
    trap = trap.to(dtype) if trap is not None else None
    dt = dt.to(dtype) if dt is not None else None

    q_bias = rearrange(q_bias, "h r d -> r h d")
    k_bias = rearrange(k_bias, "h r d -> r h d")
    q = q + q_bias[None, None, :, :, :]
    k = k + k_bias[None, None, :, :, :]

    qk_dot = torch.einsum("bsRhd,bsrhd->bsRrh", q, k)

    if angles is not None:
        angles = angles.unsqueeze(2)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        def apply_rotary_emb(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            if rotate_pairwise:
                # Pairwise convention used by mamba3_MIMO_step_ref / debug_mimo_step.py.
                tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
                tensor_0 = tensor_reshaped[..., 0]
                tensor_1 = tensor_reshaped[..., 1]
                rotated_0 = tensor_0 * cos - tensor_1 * sin
                rotated_1 = tensor_0 * sin + tensor_1 * cos
                return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
            # Kernel-aligned convention (kept as default for existing tests).
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 2, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 1, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack([rotated_0, rotated_1], dim=-2).view_as(tensor)

        def apply_rotary_emb_rotate_half(tensor: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            tensor_reshaped = tensor.view(*tensor.shape[:-1], 4, -1)
            tensor_0 = tensor_reshaped[..., 0, :]
            tensor_1 = tensor_reshaped[..., 2, :]
            rotated_0 = tensor_0 * cos - tensor_1 * sin
            rotated_1 = tensor_0 * sin + tensor_1 * cos
            return torch.stack(
                [
                    rotated_0,
                    tensor_reshaped[..., 1, :],
                    rotated_1,
                    tensor_reshaped[..., 3, :],
                ],
                dim=-2,
            ).view_as(tensor)

        if rotary_dim_divisor == 4:
            q = apply_rotary_emb_rotate_half(q, cos_angles, sin_angles)
            k = apply_rotary_emb_rotate_half(k, cos_angles, sin_angles)
        elif rotary_dim_divisor == 2:
            q = apply_rotary_emb(q, cos_angles, sin_angles)
            k = apply_rotary_emb(k, cos_angles, sin_angles)
        else:
            raise ValueError(f"Invalid rotary_dim_divisor: {rotary_dim_divisor}")

    if return_final_state:
        final_k = k[:, -1].contiguous().clone()
    else:
        final_k = None

    trap = torch.nn.functional.sigmoid(trap)
    gamma = dt * trap
    dt_shifted = torch.nn.functional.pad(dt[:, :, 1:], (0, 1), value=0.0)
    trap_shifted = torch.nn.functional.pad(trap[:, :, 1:], (0, 1), value=0.0)
    shifted_gamma = dt_shifted * (1 - trap_shifted)
    factor = gamma + shifted_gamma
    k = torch.einsum("bsrhn,bhs->bsrhn", k, factor)
    qk_dot = torch.einsum("bsrRh,bhs->bsrRh", qk_dot, shifted_gamma)

    v = torch.einsum("bthd,hrd->btrhd", v, mimo_v)

    def segsum_unstable(x: Tensor) -> Tensor:
        x_segsum = x[..., :, None] - x[..., None, :]
        mask = torch.tril(torch.ones(x.size(-1), x.size(-1), device=x.device, dtype=torch.bool), diagonal=0)
        return x_segsum.masked_fill(~mask, -torch.inf)

    mimo_mask_outer = segsum_unstable(dA_cs)
    mimo_mask_inner = torch.ones(mimo_rank, mimo_rank, dtype=torch.bool, device=q.device)
    mimo_mask = torch.kron(mimo_mask_outer, mimo_mask_inner[None, None, None, :, :])

    q = rearrange(q, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    k_scaled = rearrange(k, "b (n c) r h d -> b h n c r d", c=chunk_size)
    k_scaled = torch.einsum("bhncrd,bhnc->bhncrd", k_scaled, torch.exp(dA_cs_rev))
    k_scaled = rearrange(k_scaled, "b h n c r d -> b h n (c r) d", c=chunk_size)
    k = rearrange(k, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    v = rearrange(v, "b (n c) r h d -> b h n (c r) d", c=chunk_size)
    kv = k_scaled.transpose(-1, -2) @ v

    curr_state = torch.zeros_like(kv[:, :, 0, :, :])
    for n in range(nchunks):
        curr_dA_sum = dA_cs[:, :, n, -1]
        next_state = (torch.exp(curr_dA_sum[:, :, None, None]) * curr_state) + kv[:, :, n, :, :]
        kv[:, :, n, :, :] = curr_state
        curr_state = next_state

    if return_final_state:
        final_state = next_state.float()
    else:
        final_state = None

    q_inter = q * torch.exp(repeat(dA_cs, "b h n c -> b h n (c r)", r=mimo_rank).unsqueeze(-1))
    inter = q_inter @ kv
    intra = ((q @ k.transpose(-1, -2)) * torch.exp(mimo_mask)) @ v
    o = inter + intra
    o = rearrange(o, "b h n (c r) d -> b h n c r d", r=mimo_rank)

    v = rearrange(v, "b h n (c r) d -> b h (n c) r d", r=mimo_rank)
    qk_dot = rearrange(qk_dot, "b t R r h -> b h t R r")
    qkv = torch.einsum("bhtRr,bhtrp->bhtRp", qk_dot, v)
    qkv = rearrange(qkv, "b h (n c) r d -> b h n c r d", c=chunk_size)
    o -= qkv

    if D is not None:
        vd = torch.einsum("bhtrp,h->bhtrp", v, D)
        vd = rearrange(vd, "b h (n c) r d -> b h n c r d", c=chunk_size)
        o += vd

    if fused_norm:
        raw_y = rearrange(o, "b h n c r p -> b (n c) r h p").float()
        z_gate = torch.einsum(
            "b l h p,h r p->b l r h p",
            z.float(),
            mimo_z.float(),
        )
        raw_y = raw_y * torch.rsqrt(
            raw_y.square().mean(dim=-1, keepdim=True) + outproj_norm_eps
        )
        if outproj_norm_weight is not None:
            if outproj_norm_weight.ndim == 1:
                if outproj_norm_weight.numel() != nheads * raw_y.shape[-1]:
                    raise ValueError(
                        f"Expected flattened outproj_norm_weight to have "
                        f"{nheads * raw_y.shape[-1]} elements, got "
                        f"{outproj_norm_weight.numel()}."
                    )
                outproj_norm_weight = rearrange(
                    outproj_norm_weight, "(h p) -> h p", h=nheads
                )
            elif outproj_norm_weight.shape != (nheads, raw_y.shape[-1]):
                raise ValueError(
                    f"Expected outproj_norm_weight to have shape "
                    f"({nheads}, {raw_y.shape[-1]}) or "
                    f"({nheads * raw_y.shape[-1]},), got "
                    f"{tuple(outproj_norm_weight.shape)}."
                )
            raw_y = raw_y * outproj_norm_weight[None, None, None, :, :]
        raw_y = raw_y * torch.nn.functional.silu(z_gate)
        out = torch.einsum("b l r h p,h r p->b l h p", raw_y, mimo_o.float())
        return out[:, :orig_seqlen], final_state, final_k

    if z is not None:
        z = torch.einsum("bthd,hrd->btrhd", z, mimo_z)
        z = rearrange(z, "b (n c) r h d -> b h n c r d", c=chunk_size)
        o = o * torch.nn.functional.silu(z)

    if contract_mimo_out:
        assert mimo_o is not None
        o = torch.einsum("bhncrd,hrd->bhncd", o, mimo_o)
        out = rearrange(o, "b h n c d -> b (n c) h d")
        return out[:, :orig_seqlen], final_state, final_k

    out = rearrange(o, "b h n c r d -> b (n c) r h d")
    return out[:, :orig_seqlen], final_state, final_k



def compute_dacs_segsum_ref(
    da: torch.Tensor,  # [B, H, S]
    chunk_size: int,
):
    """Dense reference for compute_dacs_segsum_triton.

    Requires S to be a multiple of chunk_size.  Returns (da_cs, da_cs_rev, segsum).
    """
    from einops import repeat
    B, H, S = da.shape
    nchunks = S // chunk_size

    da_reshaped = da.view(B, H, nchunks, chunk_size)
    da_cs = torch.cumsum(da_reshaped, dim=-1)
    da_cs_sum = torch.sum(da_reshaped, dim=-1)
    da_cs_rev = da_cs_sum[..., None] - da_cs

    segsum = repeat(da_reshaped, "... d -> ... d e", e=chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=da_cs.device, dtype=bool), diagonal=-1)
    segsum = segsum.masked_fill(~mask, 0)
    segsum = torch.cumsum(segsum, dim=-2)

    return da_cs.view(B, H, S), da_cs_rev.view(B, H, S), segsum
