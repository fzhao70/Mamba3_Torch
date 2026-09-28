"""CPU MIMO parity against the unmodified official e9594ce oracles."""

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mamba3_torch import (  # noqa: E402
    Mamba3, Mamba3Stack, mamba3_mimo_chunked, mamba3_mimo_step,
)
from official_mimo_ref import (  # noqa: E402
    apply_angle_dt_reference, compute_dacs_segsum_ref,
    mamba3_MIMO_chunk_ref, mamba3_MIMO_step_ref,
)
from official_norm_ref import rms_norm_ref  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(260315569)


def make_inputs(b=2, length=37, rank=2, h=4, hq=2, p=8, n=16, na=4, seed=1729):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float64)

    return dict(
        Q=randn(b, length, rank, hq, n), K=randn(b, length, rank, hq, n),
        V=randn(b, length, h, p), ADT=-0.1 - 0.4 * rand(b, h, length),
        DT=0.05 + 0.2 * rand(b, h, length), Trap=randn(b, h, length),
        Q_bias=randn(h, rank, n), K_bias=randn(h, rank, n),
        Angles=randn(b, length, h, na), MIMO_V=randn(h, rank, p),
        MIMO_O=randn(h, rank, p), MIMO_Z=randn(h, rank, p),
        D=randn(h), Z=randn(b, length, h, p),
    )


def slice_inputs(inputs, start, stop):
    return {
        name: (value[:, start:stop] if name in {"Q", "K", "V", "Angles", "Z"}
               else value[..., start:stop] if name in {"ADT", "DT", "Trap"} else value)
        if value is not None else None
        for name, value in inputs.items()
    }


def step_loop(inputs, states=None, **options):
    outputs = []
    for t in range(inputs["V"].shape[1]):
        y, states = mamba3_mimo_step(
            q=inputs["Q"][:, t], k=inputs["K"][:, t], v=inputs["V"][:, t],
            adt=inputs["ADT"][..., t], dt=inputs["DT"][..., t], trap=inputs["Trap"][..., t],
            q_bias=inputs["Q_bias"], k_bias=inputs["K_bias"], angles=inputs["Angles"][:, t],
            mimo_v=inputs["MIMO_V"], mimo_o=inputs["MIMO_O"], mimo_z=inputs["MIMO_Z"],
            D=inputs["D"], z=inputs["Z"][:, t] if inputs["Z"] is not None else None,
            states=states, **options,
        )
        outputs.append(y)
    return torch.stack(outputs, 1), states


def assert_result_close(actual, expected, atol=1e-10, rtol=0):
    torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol, check_dtype=False)
    assert len(actual[1]) == len(expected[1]) == 4
    for a, e in zip(actual[1], expected[1]):
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol, check_dtype=False)


def official_mimo_combined(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, MIMO_V, MIMO_Z, MIMO_Out,
    Angles, D=None, Z=None, chunk_size=16, rotary_dim_divisor=4,
    dtype=torch.float64, return_state=False, cu_seqlens=None,
    fuse_pregate_headwise_rms_norm=False, outproj_norm_weight=None,
    outproj_norm_eps=1e-5, rotate_pairwise=False,
):
    """Official wrapper preprocessing and chunk oracle, with no port helpers."""
    angles_cumsum = apply_angle_dt_reference(Angles, DT.permute(0, 2, 1))
    da_cs, da_cs_rev = compute_dacs_segsum_ref(ADT, chunk_size)[:2]
    y, state, last_k = mamba3_MIMO_chunk_ref(
        q=Q, k=K, v=V, q_bias=Q_bias, k_bias=K_bias, mimo_v=MIMO_V, mimo_o=MIMO_Out,
        z=Z, mimo_z=MIMO_Z, angles=angles_cumsum, dA_cs=da_cs, dA_cs_rev=da_cs_rev,
        dt=DT, trap=Trap, D=D, chunk_size=chunk_size, rotary_dim_divisor=rotary_dim_divisor,
        return_final_state=return_state, dtype=dtype, rotate_pairwise=rotate_pairwise,
        contract_mimo_out=MIMO_Out is not None, cu_seqlens=cu_seqlens,
        fused_norm=fuse_pregate_headwise_rms_norm, outproj_norm_weight=outproj_norm_weight,
        outproj_norm_eps=outproj_norm_eps,
    )
    if return_state:
        last_angle = angles_cumsum[:, -1] % (2 * math.pi)
        return y, last_angle, state.permute(0, 1, 3, 2), last_k, V[:, -1]
    return y


@pytest.mark.parametrize("rotate_pairwise,na", [(False, 8), (False, 4), (True, 8)])
@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize("h,hq", [(4, 4), (4, 2)])
@pytest.mark.parametrize("length", [32, 64])
@pytest.mark.parametrize("chunk_size", [8, 16])
@pytest.mark.parametrize("with_d", [False, True])
@pytest.mark.parametrize("with_z,contract,fused_norm", [
    (False, False, False), (False, True, False),
    (True, False, False), (True, True, False), (True, True, True),
])
def test_chunked_official_parity(
    rotate_pairwise, na, rank, h, hq, length, chunk_size, with_d, with_z, contract, fused_norm,
):
    inputs = make_inputs(length=length, rank=rank, h=h, hq=hq, na=na)
    inputs["D"] = inputs["D"] if with_d else None
    inputs["Z"] = inputs["Z"] if with_z else None
    inputs["MIMO_Z"] = inputs["MIMO_Z"] if with_z else None
    inputs["MIMO_O"] = inputs["MIMO_O"] if contract else None
    weight = torch.randn(h, 8, dtype=torch.float64) if fused_norm else None
    actual = mamba3_mimo_chunked(
        **inputs, chunk_size=chunk_size, rotate_pairwise=rotate_pairwise,
        fused_norm=fused_norm, outproj_norm_weight=weight, return_final_states=True,
    )
    ref_inputs = dict(inputs)
    ref_inputs["MIMO_Out"] = ref_inputs.pop("MIMO_O")
    expected = official_mimo_combined(
        **ref_inputs, chunk_size=chunk_size, rotary_dim_divisor=16 // na,
        rotate_pairwise=rotate_pairwise, fuse_pregate_headwise_rms_norm=fused_norm,
        outproj_norm_weight=weight, dtype=torch.float64, return_state=True,
    )
    # The oracle computes the angle cumsum in float32 (apply_angle_dt_reference), so parity is
    # float32-limited (observed max abs diff ~8e-6); float64 exactness is tested separately.
    assert_result_close(actual, (expected[0], expected[1:]), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize("hq", [4, 2])
@pytest.mark.parametrize("na", [4, 8])
@pytest.mark.parametrize("with_states", [False, True])
@pytest.mark.parametrize("with_d", [False, True])
@pytest.mark.parametrize("with_z,fused_norm", [(False, False), (True, False), (True, True)])
def test_recurrent_official_parity(rank, hq, na, with_states, with_d, with_z, fused_norm):
    # Prefix and suffix share the MIMO projections: the oracle carries projected V.
    inputs = make_inputs(length=43, rank=rank, hq=hq, na=na)
    inputs["D"] = inputs["D"] if with_d else None
    inputs["Z"] = inputs["Z"] if with_z else None
    inputs["MIMO_Z"] = inputs["MIMO_Z"] if with_z else None
    if fused_norm:
        # The verbatim step oracle contracts a forced-float32 output without
        # casting MIMO_O; float32 weights are required to execute that branch.
        inputs["MIMO_O"] = inputs["MIMO_O"].float()
    options = dict(
        fused_norm=fused_norm,
        outproj_norm_weight=torch.randn(4, 8, dtype=torch.float64) if fused_norm else None,
    )
    initial, ref_initial = None, None
    if with_states:
        prefix = slice_inputs(inputs, 0, 11)
        _, initial = mamba3_mimo_chunked(
            **prefix, **options, rotate_pairwise=True, chunk_size=8, return_final_states=True,
        )
        _, ref_initial = mamba3_MIMO_step_ref(**prefix, **options)
    suffix = slice_inputs(inputs, 11, 43)
    saved = tuple(s.clone() for s in initial) if initial is not None else None
    actual = mamba3_mimo_chunked(
        **suffix, **options, initial_states=initial, rotate_pairwise=True,
        chunk_size=8, return_final_states=True,
    )
    expected, ref_states = mamba3_MIMO_step_ref(**suffix, **options, Input_States=ref_initial)
    angle, state, k, raw_v = actual[1]
    projected_v = torch.einsum("bhp,hrp->bhrp", raw_v, inputs["MIMO_V"])
    ref_angle, ref_state, ref_k, ref_v = ref_states
    assert_result_close(
        (actual[0], (angle, state, k.transpose(1, 2), projected_v)),
        (expected, (ref_angle % (2 * math.pi), ref_state, ref_k, ref_v)),
        atol=1e-4, rtol=1e-4,
    )
    if saved is not None:
        for value, snapshot in zip(initial, saved):
            torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("rotate_pairwise", [False, True])
@pytest.mark.parametrize("fused_norm", [False, True])
@pytest.mark.parametrize("rank", [1, 2, 4])
@pytest.mark.parametrize("hq", [4, 2])
@pytest.mark.parametrize("na", [4, 8])
@pytest.mark.parametrize("contract", [False, True])
@pytest.mark.parametrize("with_states", [False, True])
def test_float64_exactness(rotate_pairwise, fused_norm, rank, hq, na, contract, with_states):
    inputs = make_inputs(length=48, rank=rank, hq=hq, na=na)
    inputs["MIMO_O"] = inputs["MIMO_O"] if contract else None
    options = dict(
        rotate_pairwise=rotate_pairwise, fused_norm=fused_norm,
        outproj_norm_weight=torch.randn(4, 8, dtype=torch.float64) if fused_norm else None,
    )
    initial = None
    if with_states:
        _, initial = mamba3_mimo_chunked(
            **slice_inputs(inputs, 0, 11), **options, chunk_size=8, return_final_states=True,
        )
    inputs = slice_inputs(inputs, 11, 48)  # Nonmultiple lengths exercise padding.
    actual = mamba3_mimo_chunked(
        **inputs, **options, initial_states=initial, chunk_size=8, return_final_states=True,
    )
    assert_result_close(actual, step_loop(inputs, initial, **options))
    for chunk_size in (1, 16, 64):
        other = mamba3_mimo_chunked(
            **inputs, **options, initial_states=initial, chunk_size=chunk_size, return_final_states=True,
        )
        assert_result_close(actual, other)
    first, state = mamba3_mimo_chunked(
        **slice_inputs(inputs, 0, 20), **options, initial_states=initial,
        chunk_size=8, return_final_states=True,
    )
    second, state = mamba3_mimo_chunked(
        **slice_inputs(inputs, 20, 37), **options, initial_states=state,
        chunk_size=8, return_final_states=True,
    )
    assert_result_close(actual, (torch.cat((first, second), 1), state))


@pytest.mark.parametrize("rotate_pairwise", [False, True])
def test_gradcheck(rotate_pairwise):
    inputs = make_inputs(b=1, length=8, rank=2, h=2, hq=1, p=3, n=4, na=1)
    names = (
        "Q", "K", "V", "ADT", "DT", "Trap", "Angles", "MIMO_V", "MIMO_O", "MIMO_Z", "Z", "D",
    )
    variables = tuple(inputs[name].requires_grad_() for name in names)

    def fn(*args):
        values = dict(inputs)
        values.update(zip(names, args))
        return mamba3_mimo_chunked(**values, chunk_size=4, rotate_pairwise=rotate_pairwise)

    assert torch.autograd.gradcheck(fn, variables)


@pytest.mark.parametrize("rotate_pairwise", [False, True])
@pytest.mark.parametrize("fused_norm", [False, True])
def test_gradients_through_carried_states(rotate_pairwise, fused_norm):
    inputs = make_inputs(b=1, length=8, rank=2, h=2, hq=1, p=3, n=4, na=1)
    inputs = {name: value.requires_grad_() for name, value in inputs.items()}
    options = dict(rotate_pairwise=rotate_pairwise, fused_norm=fused_norm)
    full, states = mamba3_mimo_chunked(**inputs, **options, chunk_size=4, return_final_states=True)
    loss = full.square().sum() + sum(s.square().sum() for s in states)
    full_grads = torch.autograd.grad(loss, tuple(inputs.values()))
    prefix, initial = mamba3_mimo_chunked(
        **slice_inputs(inputs, 0, 3), **options, chunk_size=4, return_final_states=True,
    )
    suffix, states = mamba3_mimo_chunked(
        **slice_inputs(inputs, 3, 8), **options, initial_states=initial,
        chunk_size=4, return_final_states=True,
    )
    loss = torch.cat((prefix, suffix), 1).square().sum() + sum(s.square().sum() for s in states)
    carried_grads = torch.autograd.grad(loss, tuple(inputs.values()))
    for actual, expected in zip(carried_grads, full_grads):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


def make_module(rank, rope_fraction, is_outproj_norm, fuse):
    return Mamba3(
        d_model=32, d_state=16, headdim=8, is_mimo=True, mimo_rank=rank,
        rope_fraction=rope_fraction, is_outproj_norm=is_outproj_norm,
        fuse_pregate_headwise_norm=fuse, chunk_size=8,
    ).double()


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("rope_fraction", [0.5, 1.0])
@pytest.mark.parametrize("is_outproj_norm", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_module_step_and_carry(rank, rope_fraction, is_outproj_norm, fuse):
    mod = make_module(rank, rope_fraction, is_outproj_norm, fuse)
    u = torch.randn(2, 37, 32, dtype=torch.float64)
    with torch.no_grad():
        actual = mod(u, return_final_states=True)
        state = mod.allocate_states(2)
        assert [s.shape for s in state] == [
            (2, 8, mod.num_rope_angles), (2, 8, 8, 16), (2, rank, 8, 16), (2, 8, 8),
        ]
        assert all(s.dtype == torch.float64 and torch.count_nonzero(s) == 0 for s in state)
        outputs = []
        for t in range(u.shape[1]):
            y, state = mod.step(u[:, t], state)
            outputs.append(y)
        assert_result_close(actual, (torch.stack(outputs, 1), state))
        first, state = mod(u[:, :20], return_final_states=True)
        second, state = mod(u[:, 20:], initial_states=state, return_final_states=True)
        assert_result_close(actual, (torch.cat((first, second), 1), state))
        torch.testing.assert_close(actual[0], mod(u), atol=0, rtol=0)


def heavy_tail_activation(x):
    # Copied independently from the supplied official module.
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def official_forward_mimo(mod, u):
    """Official forward MIMO branch with only torch oracles and dtype promotion.

    The inference-cache/SISO branches are inapplicable. Explicit promotion of
    the last norm contraction permits float64 module weights; the official
    code otherwise attempts an unsupported float32/float64 einsum.
    """
    self = mod
    batch, seqlen, dim = u.shape

    # Apply in_proj
    zxBCdtAtrap = self.in_proj(u)
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        zxBCdtAtrap,
        [
            self.d_inner, self.d_inner,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.nheads, self.nheads, self.nheads,
            self.num_rope_angles
        ],
        dim=-1)
    z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
    x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
    B = rearrange(B, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
    C = rearrange(C, "b l (r g n) -> b l r g n", r=self.mimo_rank, g=self.num_bc_heads)
    trap = rearrange(trap, "b l h -> b h l")

    # Compute ADT, DT
    _A = -heavy_tail_activation(dd_A.to(torch.float32))
    _A = torch.clamp(_A, max=-self.A_floor)
    DT = F.softplus(dd_dt + self.dt_bias)
    ADT = _A * DT
    DT = rearrange(DT, "b l n -> b n l")
    ADT = rearrange(ADT, "b l n -> b n l")

    # Compute angle — cast to float32 as required by the MIMO/SISO kernels
    angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

    # Apply RMS Norm on B and C
    B = rms_norm_ref(B, self.B_norm.weight, None, eps=self.B_norm.eps)
    C = rms_norm_ref(C, self.C_norm.weight, None, eps=self.C_norm.eps)

    y = official_mimo_combined(
        Q=C,
        K=B,
        V=x,
        ADT=ADT,
        DT=DT,
        Trap=trap,
        Q_bias=self.C_bias,
        K_bias=self.B_bias,
        MIMO_V=self.mimo_x,
        MIMO_Z=self.mimo_z,
        MIMO_Out=self.mimo_o if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
        Angles=angles,
        D=self.D,
        Z=z if (self.fuse_pregate_headwise_norm or not self.is_outproj_norm) else None,
        chunk_size=self.chunk_size,
        rotary_dim_divisor=self.rotary_dim_divisor,
        dtype=x.dtype,
        return_state=False,
        cu_seqlens=None,
        fuse_pregate_headwise_rms_norm=self.fuse_pregate_headwise_norm,
        outproj_norm_weight=self.norm.weight if self.fuse_pregate_headwise_norm else None,
        outproj_norm_eps=self.norm.eps if self.fuse_pregate_headwise_norm else 1e-5,
    )
    if self.is_outproj_norm and not self.fuse_pregate_headwise_norm:
        z = torch.einsum("blhp,hrp->blrhp", z.float(), self.mimo_z)
        z = rearrange(z, "b l r h p -> b l r (h p)")
        y = rearrange(y, "b l r h p -> b l r (h p)").float()
        y = rms_norm_ref(
            y, self.norm.weight, None, z, self.norm.eps, self.headdim, True, upcast=True,
        )
        y = rearrange(y, "b l r (h p) -> b l r h p", p=self.headdim)
        y = torch.einsum("blrhp,hrp->blhp", y.to(self.mimo_o.dtype), self.mimo_o)
    y = rearrange(y, "b l h p -> b l (h p)")

    out = self.out_proj(y.to(x.dtype))
    return out


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("rope_fraction", [0.5, 1.0])
@pytest.mark.parametrize("is_outproj_norm", [False, True])
@pytest.mark.parametrize("fuse", [False, True])
def test_module_official_and_state_dict(rank, rope_fraction, is_outproj_norm, fuse):
    mod = make_module(rank, rope_fraction, is_outproj_norm, fuse)
    u = torch.randn(2, 32, 32, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(mod(u), official_forward_mimo(mod, u), atol=1e-4, rtol=1e-4)
    # Exact MIMO parameter list from _official/mamba3_module_official.py.
    expected_shapes = {
        "in_proj.weight": (2 * 64 + 2 * 16 * rank + 3 * 8 + mod.num_rope_angles, 32),
        "dt_bias": (8,), "B_bias": (8, rank, 16), "C_bias": (8, rank, 16),
        "B_norm.weight": (16,), "C_norm.weight": (16,), "D": (8,),
        "mimo_x": (8, rank, 8), "mimo_z": (8, rank, 8), "mimo_o": (8, rank, 8),
        "out_proj.weight": (32, 64),
    }
    if is_outproj_norm:
        expected_shapes["norm.weight"] = (64,)
    assert set(mod.state_dict()) == set(expected_shapes)
    assert dict(mod.named_buffers()) == {}
    assert {k: tuple(v.shape) for k, v in mod.named_parameters()} == expected_shapes
    assert mod.fuse_pregate_headwise_norm == bool(fuse and is_outproj_norm)
    for name, value in [("mimo_x", 1 / rank), ("mimo_z", 1), ("mimo_o", 1 / rank),
                        ("B_bias", 1), ("C_bias", 1), ("D", 1)]:
        parameter = getattr(mod, name)
        torch.testing.assert_close(parameter, torch.full_like(parameter, value), atol=0, rtol=0)
    assert mod.dt_bias._no_weight_decay and mod.D._no_weight_decay
    fresh = make_module(rank, rope_fraction, is_outproj_norm, fuse)
    fresh.load_state_dict(mod.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(fresh(u), mod(u), atol=0, rtol=0)


@pytest.mark.parametrize("rank", [2, 4])
@pytest.mark.parametrize("rope_fraction", [0.5, 1.0])
@pytest.mark.parametrize("is_outproj_norm,fuse", [(False, False), (True, False), (True, True)])
def test_stack_step_and_carry(rank, rope_fraction, is_outproj_norm, fuse):
    model = Mamba3Stack(
        32, 2, d_state=16, headdim=8, chunk_size=8, is_mimo=True, mimo_rank=rank,
        rope_fraction=rope_fraction, is_outproj_norm=is_outproj_norm,
        fuse_pregate_headwise_norm=fuse,
    ).double()
    u = torch.randn(2, 37, 32, dtype=torch.float64)
    with torch.no_grad():
        out, final_states = model(u, return_final_states=True)
        outputs, states = [], model.allocate_states(2)
        for t in range(u.shape[1]):
            y, states = model.step(u[:, t], states)
            outputs.append(y)
        recurrent = torch.stack(outputs, 1)
        for actual, expected in zip(final_states, states):
            assert_result_close((out, actual), (recurrent, expected))
        first, states = model(u[:, :20], return_final_states=True)
        second, states = model(u[:, 20:], initial_states=states, return_final_states=True)
        for actual, expected in zip(final_states, states):
            assert_result_close((out, actual), (torch.cat((first, second), 1), expected))


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("fused_norm", [False, True])
def test_precision_and_autocast(device, fused_norm):
    inputs64 = make_inputs()
    options = dict(fused_norm=fused_norm)
    out64 = mamba3_mimo_chunked(**inputs64, **options, chunk_size=8)
    inputs32 = {k: v.to(device=device, dtype=torch.float32) for k, v in inputs64.items()}
    out32 = mamba3_mimo_chunked(**inputs32, **options, chunk_size=8)
    assert out32.dtype == torch.float32
    relative_error = (out32.cpu().double() - out64).norm() / out64.norm()
    assert relative_error < 1e-4
    with torch.autocast(device_type=device, dtype=torch.float16):
        autocast_out = mamba3_mimo_chunked(**inputs32, **options, chunk_size=8)
    torch.testing.assert_close(autocast_out, out32, atol=0, rtol=0)
    for dtype in (torch.float16, torch.bfloat16):
        low_inputs = {k: v.to(dtype) for k, v in inputs32.items()}
        promoted = {k: v.float() for k, v in low_inputs.items()}
        with torch.autocast(device_type=device, dtype=torch.float16):
            low_out, states = mamba3_mimo_chunked(
                **low_inputs, **options, chunk_size=8, return_final_states=True,
            )
        expected = mamba3_mimo_chunked(**promoted, **options, chunk_size=8)
        assert low_out.dtype == dtype and all(s.dtype == torch.float32 for s in states)
        torch.testing.assert_close(low_out, expected.to(dtype), atol=0, rtol=0)

    model = Mamba3Stack(
        32, 2, d_state=16, headdim=8, chunk_size=8, is_mimo=True, mimo_rank=2,
        is_outproj_norm=True, fuse_pregate_headwise_norm=fused_norm, device=device,
    )
    u = torch.randn(2, 13, 32, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        result = model(u)
        loss = result.float().square().mean()
    loss.backward()
    assert torch.isfinite(result).all() and torch.isfinite(u.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("rotate_pairwise", [False, True])
def test_single_token_and_norm_weight_layout(rotate_pairwise):
    inputs = make_inputs(length=1)
    weight = torch.randn(4, 8, dtype=torch.float64)
    options = dict(rotate_pairwise=rotate_pairwise, fused_norm=True, outproj_norm_weight=weight)
    actual = mamba3_mimo_chunked(**inputs, **options, chunk_size=16, return_final_states=True)
    assert_result_close(actual, step_loop(inputs, **options))
    options["outproj_norm_weight"] = weight.flatten()
    other = mamba3_mimo_chunked(**inputs, **options, chunk_size=16, return_final_states=True)
    assert_result_close(actual, other, atol=0, rtol=0)
