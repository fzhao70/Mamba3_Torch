"""Differentiable, pure-PyTorch Mamba-3 SISO and MIMO scans.

The trapezoidal recurrence and pairwise rotary convention follow
state-spaces/mamba, commit e9594ce (Apache-2.0; see _official/LICENSE).
The chunked SSD implementation here uses only ordinary PyTorch operations.
"""

import math

import torch
import torch.nn.functional as F


def _compute_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


def _expand_heads(x, nheads):
    if nheads % x.shape[-2]:
        raise ValueError("Q/K head counts must divide the V head count")
    return x.repeat_interleave(nheads // x.shape[-2], dim=-2)


def _rotary(x, theta):
    """Rotate adjacent pairs in the first 2 * na dimensions."""
    na = theta.shape[-1]
    if 2 * na > x.shape[-1]:
        raise ValueError("Angles cannot rotate more than the state dimension")
    pairs = x[..., : 2 * na].reshape(*x.shape[:-1], na, 2)
    cos, sin = theta.cos(), theta.sin()
    even, odd = pairs.unbind(-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
    return torch.cat((rotated.flatten(-2), x[..., 2 * na :]), dim=-1)


def _wrap_angles(theta):
    two_pi = 2 * math.pi
    return theta - two_pi * torch.floor(theta / two_pi)


def _initial_states(states, batch, heads, width, state_dim, na, device, dtype):
    shapes = (
        (batch, heads, na),
        (batch, heads, width, state_dim),
        (batch, heads, state_dim),
        (batch, heads, width),
    )
    if states is None:
        return tuple(torch.zeros(shape, device=device, dtype=dtype) for shape in shapes)
    if len(states) != 4 or any(tuple(s.shape) != shape for s, shape in zip(states, shapes)):
        raise ValueError(f"Expected (angle, SSM, K, V) states with shapes {shapes}")
    return tuple(s.to(device=device, dtype=dtype) for s in states)


def _segsum(x):
    """Within-chunk log decays, without subtracting large cumulative sums."""
    size = x.shape[-1]
    lower = torch.ones(size, size, dtype=torch.bool, device=x.device).tril(-1)
    terms = x.unsqueeze(-1).expand(*x.shape, size).masked_fill(~lower, 0)
    sums = terms.cumsum(dim=-2)
    causal = torch.ones_like(lower).tril()
    return sums.masked_fill(~causal, -torch.inf)


def mamba3_siso_chunked(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
    D=None, Z=None, initial_states=None, chunk_size=64, return_final_states=False,
):
    """Run a batched SISO scan with O(L * chunk_size) attention memory.

    Q/K: (b, L, hq, n); V/Z: (b, L, h, p); ADT/DT/Trap: (b, h, L);
    biases: (h, n); Angles: (b, L, h, na); D: (h,). Q/K heads are
    repeated consecutively to h heads. Trap and Angles are raw projections.

    States are (angle, SSM, rotated K, V), shaped (b,h,na), (b,h,p,n),
    (b,h,n), (b,h,p). The SSM state excludes the last token's pending
    trapezoid contribution. Input states are never modified or detached.
    All computation and returned states use float64 for float64 V and
    float32 otherwise, including under autocast. Output uses V.dtype.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    batch, length, heads, width = V.shape
    if length == 0:
        raise ValueError("The sequence must contain at least one token")
    state_dim, na = Q.shape[-1], Angles.shape[-1]
    dtype = _compute_dtype(V.dtype)

    # Explicitly disable autocast: einsum must accumulate in the chosen dtype.
    with torch.autocast(device_type=V.device.type, enabled=False):
        q = _expand_heads(Q.to(dtype), heads) + Q_bias.to(dtype)
        k = _expand_heads(K.to(dtype), heads) + K_bias.to(dtype)
        v = V.to(dtype)
        a, dt, trap = ADT.to(dtype), DT.to(dtype), Trap.to(dtype).sigmoid()
        angle0, state, k0, v0 = _initial_states(
            initial_states, batch, heads, width, state_dim, na, V.device, dtype,
        )
        increments = Angles.to(dtype).tanh() * math.pi * dt.transpose(1, 2).unsqueeze(-1)
        theta = _wrap_angles(increments.cumsum(1) + angle0.unsqueeze(1))

        # B_s = k_s * (gamma_s + DT_{s+1}(1-trap_{s+1})). The second
        # term belongs only to future outputs, so subtract it on the diagonal.
        shifted = F.pad((dt * (1 - trap))[..., 1:], (0, 1))
        scale = dt * trap + shifted
        diagonal = (q * k).sum(-1) * shifted.transpose(1, 2)
        q, k = _rotary(q, theta), _rotary(k, theta)
        final_k, final_v = k[:, -1], v[:, -1]

        # The carried K/V pair is pending at the start of this sequence.
        pending = dt[..., 0] * (1 - trap[..., 0])
        state = state + pending[..., None, None] * v0.unsqueeze(-1) * k0.unsqueeze(-2)

        size = min(chunk_size, length)
        nchunks = (length + size - 1) // size
        padding = nchunks * size - length

        def chunks(x):
            x = F.pad(x, (0, 0, 0, 0, 0, padding))
            return x.reshape(batch, nchunks, size, heads, x.shape[-1]).permute(0, 3, 1, 2, 4)

        qc = chunks(q)
        kc = chunks(k * scale.transpose(1, 2).unsqueeze(-1))
        vc = chunks(v)
        ac = F.pad(a, (0, padding)).reshape(batch, heads, nchunks, size)
        decay = _segsum(ac).exp()
        scores = torch.einsum("bhctn,bhcsn->bhcts", qc, kc) * decay
        local = torch.einsum("bhcts,bhcsp->bhctp", scores, vc)

        # Each chunk contributes a (p,n) state at its end. Reverse cumsum
        # avoids cancellation in exp(sum_{r=s+1..end} ADT_r).
        suffix = F.pad(ac[..., 1:].flip(-1).cumsum(-1).flip(-1), (0, 1))
        summaries = torch.einsum("bhcsn,bhcsp,bhcs->bhcpn", kc, vc, suffix.exp())
        chunk_decay = ac.sum(-1).exp()
        incoming = []
        for idx in range(nchunks):
            incoming.append(state)
            state = chunk_decay[:, :, idx, None, None] * state + summaries[:, :, idx]
        incoming = torch.stack(incoming, dim=2)
        carried = torch.einsum("bhctn,bhcpn->bhctp", qc, incoming)
        out = local + carried * ac.cumsum(-1).exp().unsqueeze(-1)
        out = out.permute(0, 2, 3, 1, 4).reshape(batch, nchunks * size, heads, width)
        out = out[:, :length]
        if D is not None:
            out = out + D.to(dtype)[None, None, :, None] * v
        out = out - diagonal.unsqueeze(-1) * v
        if Z is not None:
            z = Z.to(dtype)
            out = out * z * z.sigmoid()
        out = out.to(V.dtype)
        final_states = (theta[:, -1], state, final_k, final_v)
    return (out, final_states) if return_final_states else out


def mamba3_siso_step(
    q, k, v, adt, dt, trap, q_bias, k_bias, angles,
    D=None, z=None, states=None,
):
    """One recurrent step; inputs omit L and states follow the chunked API."""
    batch, heads, width = v.shape
    dtype = _compute_dtype(v.dtype)
    with torch.autocast(device_type=v.device.type, enabled=False):
        angle0, state, k0, v0 = _initial_states(
            states, batch, heads, width, q.shape[-1], angles.shape[-1], v.device, dtype,
        )
        q = _expand_heads(q.to(dtype), heads) + q_bias.to(dtype)
        k = _expand_heads(k.to(dtype), heads) + k_bias.to(dtype)
        value = v.to(dtype)
        dt, trap = dt.to(dtype), trap.to(dtype).sigmoid()
        theta = _wrap_angles(angle0 + angles.to(dtype).tanh() * math.pi * dt.unsqueeze(-1))
        q, k = _rotary(q, theta), _rotary(k, theta)
        alpha = adt.to(dtype).exp()
        beta, gamma = (1 - trap) * dt * alpha, trap * dt
        state = (
            alpha[..., None, None] * state
            + beta[..., None, None] * k0.unsqueeze(-2) * v0.unsqueeze(-1)
            + gamma[..., None, None] * k.unsqueeze(-2) * value.unsqueeze(-1)
        )
        out = torch.einsum("bhpn,bhn->bhp", state, q)
        if D is not None:
            out = out + D.to(dtype)[None, :, None] * value
        if z is not None:
            z = z.to(dtype)
            out = out * z * z.sigmoid()
        return out.to(v.dtype), (theta, state, k, value)


def _mimo_initial_states(states, batch, rank, heads, width, state_dim, na, device, dtype):
    shapes = (
        (batch, heads, na),
        (batch, heads, width, state_dim),
        (batch, rank, heads, state_dim),
        (batch, heads, width),
    )
    if states is None:
        return tuple(torch.zeros(shape, device=device, dtype=dtype) for shape in shapes)
    if len(states) != 4 or any(tuple(s.shape) != shape for s, shape in zip(states, shapes)):
        raise ValueError(f"Expected (angle, SSM, K, V) states with shapes {shapes}")
    return tuple(s.to(device=device, dtype=dtype) for s in states)


def _mimo_rotary(x, theta, rotate_pairwise):
    theta = theta.unsqueeze(-3)  # Share the angles across the rank dimension.
    if rotate_pairwise:
        return _rotary(x, theta)
    half, na = x.shape[-1] // 2, theta.shape[-1]
    if x.shape[-1] % 2 or na > half:
        raise ValueError("Half rotation requires an even state dimension and 2 * na <= n")
    first, second = x[..., :na], x[..., half : half + na]
    cos, sin = theta.cos(), theta.sin()
    return torch.cat((
        first * cos - second * sin, x[..., na:half],
        first * sin + second * cos, x[..., half + na :],
    ), dim=-1)


def _mimo_output(out, projected_v, mimo_o, D, z, mimo_z, fused_norm, weight, eps):
    """Skip, optional per-rank normalization/gate, then rank contraction."""
    dtype = out.dtype
    if D is not None:
        out = out + D.to(dtype).unsqueeze(-1) * projected_v
    if fused_norm and (z is None or mimo_z is None):
        raise ValueError("fused_norm=True requires Z and MIMO_Z")
    if fused_norm:
        out = out * torch.rsqrt(out.square().mean(dim=-1, keepdim=True) + eps)
        if weight is not None:
            heads, width = out.shape[-2:]
            if tuple(weight.shape) not in {(heads, width), (heads * width,)}:
                raise ValueError("outproj_norm_weight must have shape (h, p) or (h * p,)")
            out = out * weight.to(dtype).reshape(heads, width)
    if z is not None:
        if mimo_z is None:
            raise ValueError("Z requires MIMO_Z")
        gate = z.to(dtype).unsqueeze(-3) * mimo_z.to(dtype).transpose(0, 1)
        out = out * F.silu(gate)
    if mimo_o is not None:
        out = (out * mimo_o.to(dtype).transpose(0, 1)).sum(dim=-3)
    return out


def mamba3_mimo_chunked(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, MIMO_V,
    MIMO_O=None, D=None, Z=None, MIMO_Z=None, initial_states=None, chunk_size=16,
    rotate_pairwise=False, fused_norm=False, outproj_norm_weight=None,
    outproj_norm_eps=1e-5, return_final_states=False,
):
    """Run a batched MIMO scan with O(L * chunk_size) attention memory.

    Q/K: (b,L,R,hq,n); V/Z: (b,L,h,p); ADT/DT/Trap: (b,h,L);
    biases: (h,R,n); Angles: (b,L,h,na); MIMO projections: (h,R,p).
    Q/K heads repeat consecutively for GQA. Trap and Angles are raw.
    Output is (b,L,h,p), or (b,L,R,h,p) when MIMO_O is None.

    States are (angle, SSM, rotated K, raw V), shaped (b,h,na),
    (b,h,p,n), (b,R,h,n), (b,h,p). The last K/V contribution is pending
    until the next token. States are neither modified nor detached.
    By default rotation pairs i with i+n/2 for i<na, as in the official
    module; rotate_pairwise=True pairs 2i with 2i+1 instead.
    Compute/states use float64 for float64 V, otherwise float32, even
    under autocast. The output uses V.dtype, including with fused_norm.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    batch, length, heads, width = V.shape
    if length == 0:
        raise ValueError("The sequence must contain at least one token")
    rank, state_dim, na = Q.shape[2], Q.shape[-1], Angles.shape[-1]
    dtype = _compute_dtype(V.dtype)

    with torch.autocast(device_type=V.device.type, enabled=False):
        q = _expand_heads(Q.to(dtype), heads) + Q_bias.to(dtype).transpose(0, 1)
        k = _expand_heads(K.to(dtype), heads) + K_bias.to(dtype).transpose(0, 1)
        value = V.to(dtype)
        projection = MIMO_V.to(dtype).transpose(0, 1)
        v = value.unsqueeze(2) * projection
        a, dt, trap = ADT.to(dtype), DT.to(dtype), Trap.to(dtype).sigmoid()
        angle0, state, k0, v0 = _mimo_initial_states(
            initial_states, batch, rank, heads, width, state_dim, na, V.device, dtype,
        )
        increments = Angles.to(dtype).tanh() * math.pi * dt.transpose(1, 2).unsqueeze(-1)
        theta = _wrap_angles(increments.cumsum(1) + angle0.unsqueeze(1))

        # Every query rank interacts with every key rank at the same time.
        # Remove the next token's trapezoid weight from these diagonal blocks.
        shifted = F.pad((dt * (1 - trap))[..., 1:], (0, 1))
        scale = dt * trap + shifted
        diagonal = torch.einsum("btRhn,btrhn->btRrh", q, k)
        diagonal = diagonal * shifted.transpose(1, 2)[:, :, None, None, :]
        correction = torch.einsum("btRrh,btrhp->btRhp", diagonal, v)
        q, k = _mimo_rotary(q, theta, rotate_pairwise), _mimo_rotary(k, theta, rotate_pairwise)
        final_k, final_v = k[:, -1], value[:, -1]

        pending = dt[..., 0] * (1 - trap[..., 0])
        pending_kv = torch.einsum("brhn,brhp->bhpn", k0, v0.unsqueeze(1) * projection)
        state = state + pending[..., None, None] * pending_kv

        size = min(chunk_size, length)
        nchunks = (length + size - 1) // size
        padding = nchunks * size - length

        def chunks(x):
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, padding))
            x = x.reshape(batch, nchunks, size, rank, heads, x.shape[-1])
            return x.permute(0, 4, 1, 2, 3, 5).flatten(3, 4)

        qc = chunks(q)
        kc = chunks(k * scale.transpose(1, 2)[:, :, None, :, None])
        vc = chunks(v)
        ac = F.pad(a, (0, padding)).reshape(batch, heads, nchunks, size)
        decay = _segsum(ac).exp().repeat_interleave(rank, -2).repeat_interleave(rank, -1)
        scores = torch.einsum("bhctn,bhcsn->bhcts", qc, kc) * decay
        local = torch.einsum("bhcts,bhcsp->bhctp", scores, vc)

        suffix = F.pad(ac[..., 1:].flip(-1).cumsum(-1).flip(-1), (0, 1))
        summaries = torch.einsum(
            "bhcsn,bhcsp,bhcs->bhcpn", kc, vc, suffix.exp().repeat_interleave(rank, -1),
        )
        chunk_decay = ac.sum(-1).exp()
        incoming = []
        for idx in range(nchunks):
            incoming.append(state)
            state = chunk_decay[:, :, idx, None, None] * state + summaries[:, :, idx]
        incoming = torch.stack(incoming, dim=2)
        carried = torch.einsum("bhctn,bhcpn->bhctp", qc, incoming)
        out = local + carried * ac.cumsum(-1).exp().repeat_interleave(rank, -1).unsqueeze(-1)
        out = out.reshape(batch, heads, nchunks, size, rank, width)
        out = out.permute(0, 2, 3, 4, 1, 5).reshape(batch, nchunks * size, rank, heads, width)
        out = out[:, :length] - correction
        out = _mimo_output(
            out, v, MIMO_O, D, Z, MIMO_Z, fused_norm, outproj_norm_weight, outproj_norm_eps,
        ).to(V.dtype)
        final_states = (theta[:, -1], state, final_k, final_v)
    return (out, final_states) if return_final_states else out


def mamba3_mimo_step(
    q, k, v, adt, dt, trap, q_bias, k_bias, angles, mimo_v,
    mimo_o=None, D=None, z=None, mimo_z=None, states=None, rotate_pairwise=False,
    fused_norm=False, outproj_norm_weight=None, outproj_norm_eps=1e-5,
):
    """One MIMO recurrent step; inputs omit L and states follow the chunked API."""
    batch, heads, width = v.shape
    dtype = _compute_dtype(v.dtype)
    with torch.autocast(device_type=v.device.type, enabled=False):
        angle0, state, k0, v0 = _mimo_initial_states(
            states, batch, q.shape[1], heads, width, q.shape[-1], angles.shape[-1], v.device, dtype,
        )
        q = _expand_heads(q.to(dtype), heads) + q_bias.to(dtype).transpose(0, 1)
        k = _expand_heads(k.to(dtype), heads) + k_bias.to(dtype).transpose(0, 1)
        value = v.to(dtype)
        projection = mimo_v.to(dtype).transpose(0, 1)
        projected_v = value.unsqueeze(1) * projection
        dt, trap = dt.to(dtype), trap.to(dtype).sigmoid()
        theta = _wrap_angles(angle0 + angles.to(dtype).tanh() * math.pi * dt.unsqueeze(-1))
        q, k = _mimo_rotary(q, theta, rotate_pairwise), _mimo_rotary(k, theta, rotate_pairwise)
        alpha = adt.to(dtype).exp()
        beta, gamma = (1 - trap) * dt * alpha, trap * dt
        prev_kv = torch.einsum("brhn,brhp->bhpn", k0, v0.unsqueeze(1) * projection)
        curr_kv = torch.einsum("brhn,brhp->bhpn", k, projected_v)
        state = (
            alpha[..., None, None] * state
            + beta[..., None, None] * prev_kv
            + gamma[..., None, None] * curr_kv
        )
        out = torch.einsum("bhpn,brhn->brhp", state, q)
        out = _mimo_output(
            out, projected_v, mimo_o, D, z, mimo_z, fused_norm, outproj_norm_weight, outproj_norm_eps,
        )
        return out.to(v.dtype), (theta, state, k, value)
