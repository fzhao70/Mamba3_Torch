"""CPU parity against the unmodified official e9594ce reference functions."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

# Repo root on sys.path so the package imports without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mamba3_torch import (  # noqa: E402
    Mamba3, Mamba3Block, Mamba3Stack, RMSNormGated,
    mamba3_siso_chunked, mamba3_siso_step,
)
from official_ref import mamba3_siso_fwd_ref, mamba3_siso_step_ref  # noqa: E402
from official_norm_ref import rms_norm_ref  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(260315569)


def make_inputs(b=2, length=37, h=4, hq=2, p=8, n=16, na=4, seed=1729):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    def rand(*shape):
        return torch.rand(*shape, generator=generator, dtype=torch.float64)

    return dict(
        Q=randn(b, length, hq, n), K=randn(b, length, hq, n), V=randn(b, length, h, p),
        ADT=-0.1 - 0.4 * rand(b, h, length), DT=0.05 + 0.2 * rand(b, h, length),
        Trap=randn(b, h, length), Q_bias=randn(h, n), K_bias=randn(h, n),
        Angles=randn(b, length, h, na), D=randn(h), Z=randn(b, length, h, p),
    )


def slice_inputs(inputs, start, stop):
    return {
        name: (value[:, start:stop] if name in {"Q", "K", "V", "Angles", "Z"}
               else value[..., start:stop] if name in {"ADT", "DT", "Trap"} else value)
        if value is not None else None
        for name, value in inputs.items()
    }


def step_loop(inputs, states=None):
    outputs = []
    for t in range(inputs["V"].shape[1]):
        y, states = mamba3_siso_step(
            q=inputs["Q"][:, t], k=inputs["K"][:, t], v=inputs["V"][:, t],
            adt=inputs["ADT"][..., t], dt=inputs["DT"][..., t], trap=inputs["Trap"][..., t],
            q_bias=inputs["Q_bias"], k_bias=inputs["K_bias"], angles=inputs["Angles"][:, t],
            D=inputs["D"], z=inputs["Z"][:, t] if inputs["Z"] is not None else None, states=states,
        )
        outputs.append(y)
    return torch.stack(outputs, 1), states


def assert_result_close(actual, expected, atol, rtol):
    torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol, check_dtype=False)
    for a, e in zip(actual[1], expected[1]):
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol, check_dtype=False)


@pytest.mark.parametrize("h,hq", [(4, 4), (4, 2)])
@pytest.mark.parametrize("na", [4, 8])
@pytest.mark.parametrize("length", [1, 37, 64])
@pytest.mark.parametrize("chunk_size", [8, 16, 64])
@pytest.mark.parametrize("with_d", [False, True])
@pytest.mark.parametrize("with_z", [False, True])
@pytest.mark.parametrize("with_states", [False, True])
def test_chunked_official_parity(h, hq, na, length, chunk_size, with_d, with_z, with_states):
    inputs = make_inputs(length=length, h=h, hq=hq, na=na)
    inputs["D"] = inputs["D"] if with_d else None
    inputs["Z"] = inputs["Z"] if with_z else None
    initial = None
    if with_states:
        prefix = make_inputs(length=11, h=h, hq=hq, na=na, seed=31415)
        _, initial = mamba3_siso_fwd_ref(**prefix, dtype=torch.float64)
    saved = tuple(s.clone() for s in initial) if initial is not None else None
    actual = mamba3_siso_chunked(
        **inputs, initial_states=initial, chunk_size=chunk_size, return_final_states=True,
    )
    quadratic = mamba3_siso_fwd_ref(
        **inputs, Initial_States=initial, dtype=torch.float64, chunk_size=chunk_size,
    )
    recurrent = mamba3_siso_step_ref(**inputs, Input_States=initial)
    assert_result_close(actual, quadratic, atol=1e-4, rtol=1e-4)
    assert_result_close(actual, recurrent, atol=1e-4, rtol=1e-4)
    if saved is not None:
        for a, e in zip(initial, saved):
            torch.testing.assert_close(a, e, atol=0, rtol=0)


@pytest.mark.parametrize("hq", [4, 2])
@pytest.mark.parametrize("na", [4, 8])
@pytest.mark.parametrize("with_states", [False, True])
def test_float64_exactness(hq, na, with_states):
    inputs = make_inputs(hq=hq, na=na)
    initial = None
    if with_states:
        _, initial = mamba3_siso_chunked(
            **make_inputs(length=11, hq=hq, na=na, seed=2718), return_final_states=True,
        )
    actual = mamba3_siso_chunked(**inputs, initial_states=initial, chunk_size=8, return_final_states=True)
    assert_result_close(actual, step_loop(inputs, initial), atol=1e-10, rtol=0)
    other_chunk = mamba3_siso_chunked(
        **inputs, initial_states=initial, chunk_size=64, return_final_states=True,
    )
    assert_result_close(actual, other_chunk, atol=1e-10, rtol=0)
    first, state = mamba3_siso_chunked(
        **slice_inputs(inputs, 0, 20), initial_states=initial, chunk_size=8, return_final_states=True,
    )
    second, state = mamba3_siso_chunked(
        **slice_inputs(inputs, 20, 37), initial_states=state, chunk_size=8, return_final_states=True,
    )
    assert_result_close(actual, (torch.cat((first, second), 1), state), atol=1e-10, rtol=0)


def test_gradcheck():
    inputs = make_inputs(b=1, length=10, h=2, hq=1, p=3, n=4, na=2)
    names = ("Q", "K", "V", "ADT", "DT", "Trap", "Angles", "D", "Z")
    variables = tuple(inputs[name].requires_grad_() for name in names)

    def fn(*args):
        values = dict(inputs)
        values.update(zip(names, args))
        return mamba3_siso_chunked(**values, chunk_size=4)

    assert torch.autograd.gradcheck(fn, variables)


def test_gradients_through_carried_states():
    inputs = make_inputs(b=1, length=10, h=2, hq=1, p=3, n=4, na=2)
    inputs = {name: value.requires_grad_() for name, value in inputs.items()}
    full, states = mamba3_siso_chunked(**inputs, chunk_size=4, return_final_states=True)
    loss = full.square().sum() + sum(s.square().sum() for s in states)
    full_grads = torch.autograd.grad(loss, tuple(inputs.values()))
    prefix, initial = mamba3_siso_chunked(
        **slice_inputs(inputs, 0, 6), chunk_size=4, return_final_states=True,
    )
    suffix, states = mamba3_siso_chunked(
        **slice_inputs(inputs, 6, 10), initial_states=initial, chunk_size=4, return_final_states=True,
    )
    loss = torch.cat((prefix, suffix), 1).square().sum() + sum(s.square().sum() for s in states)
    carried_grads = torch.autograd.grad(loss, tuple(inputs.values()))
    for actual, expected in zip(carried_grads, full_grads):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [None, 4])
@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("with_z", [False, True])
def test_norm_official(dtype, group_size, norm_before_gate, with_z):
    norm = RMSNormGated(16, group_size=group_size, norm_before_gate=norm_before_gate, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn_like(norm.weight))
    x = torch.randn(2, 3, 16, dtype=dtype)
    z = torch.randn_like(x) if with_z else None
    expected = rms_norm_ref(
        x, norm.weight, None, z, norm.eps, group_size, norm_before_gate, upcast=True,
    )
    torch.testing.assert_close(norm(x, z), expected, atol=0, rtol=0)


def make_module(is_outproj_norm, rope_fraction):
    return Mamba3(
        d_model=32, d_state=16, headdim=8, ngroups=1,
        is_outproj_norm=is_outproj_norm, rope_fraction=rope_fraction, chunk_size=8,
    ).double()


@pytest.mark.parametrize("is_outproj_norm", [False, True])
@pytest.mark.parametrize("rope_fraction", [0.5, 1.0])
def test_module_step_and_carry(is_outproj_norm, rope_fraction):
    mod = make_module(is_outproj_norm, rope_fraction)
    u = torch.randn(2, 37, 32, dtype=torch.float64)
    with torch.no_grad():
        actual = mod(u, return_final_states=True)
        state = mod.allocate_states(2)
        assert [s.shape for s in state] == [
            (2, 8, mod.num_rope_angles), (2, 8, 8, 16), (2, 8, 16), (2, 8, 8),
        ]
        assert all(s.dtype == torch.float64 and torch.count_nonzero(s) == 0 for s in state)
        outputs = []
        for t in range(u.shape[1]):
            y, state = mod.step(u[:, t], state)
            outputs.append(y)
        assert_result_close(actual, (torch.stack(outputs, 1), state), atol=1e-10, rtol=0)
        first, state = mod(u[:, :20], return_final_states=True)
        second, state = mod(u[:, 20:], initial_states=state, return_final_states=True)
        assert_result_close(actual, (torch.cat((first, second), 1), state), atol=1e-10, rtol=0)
        torch.testing.assert_close(actual[0], mod(u), atol=0, rtol=0)


def heavy_tail_activation(x):
    # Copied independently from _official/mamba3_module_official.py.
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


def official_forward(mod, u):
    """Official forward's SISO branch, substituting only the torch oracles.

    No official module is imported, so this never imports mamba_ssm/Triton.
    The inference-cache/MIMO branches are inapplicable to this full forward.
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

    # Apply RMS Norm on B and C (official torch oracle replaces Triton).
    B = rms_norm_ref(B, self.B_norm.weight, None, None, self.B_norm.eps, None, True, upcast=True)
    C = rms_norm_ref(C, self.C_norm.weight, None, None, self.C_norm.eps, None, True, upcast=True)

    y, _ = mamba3_siso_fwd_ref(
        Q=C.squeeze(2),
        K=B.squeeze(2),
        V=x,
        ADT=ADT,
        DT=DT,
        Trap=trap,
        Q_bias=self.C_bias.squeeze(1),
        K_bias=self.B_bias.squeeze(1),
        Angles=angles,
        D=self.D,
        Z=z if not self.is_outproj_norm else None,
        chunk_size=self.chunk_size,
        Initial_States=None,
        dtype=torch.float64,
    )
    y = rearrange(y, "b l h p -> b l (h p)")
    if self.is_outproj_norm:
        z = rearrange(z, "b l h p -> b l (h p)")
        y = rms_norm_ref(y, self.norm.weight, None, z, self.norm.eps, self.headdim, True, upcast=True)

    out = self.out_proj(y.to(x.dtype))
    return out


@pytest.mark.parametrize("is_outproj_norm", [False, True])
@pytest.mark.parametrize("rope_fraction", [0.5, 1.0])
def test_module_official_and_state_dict(is_outproj_norm, rope_fraction):
    mod = make_module(is_outproj_norm, rope_fraction)
    u = torch.randn(2, 37, 32, dtype=torch.float64)
    with torch.no_grad():
        torch.testing.assert_close(mod(u), official_forward(mod, u), atol=1e-4, rtol=1e-4)
    # Exact SISO parameter names and shapes read from the supplied official file.
    expected_shapes = {
        "in_proj.weight": (2 * 64 + 2 * 16 + 3 * 8 + mod.num_rope_angles, 32),
        "dt_bias": (8,), "B_bias": (8, 1, 16), "C_bias": (8, 1, 16),
        "B_norm.weight": (16,), "C_norm.weight": (16,), "D": (8,),
        "out_proj.weight": (32, 64),
    }
    if is_outproj_norm:
        expected_shapes["norm.weight"] = (64,)
    assert set(mod.state_dict()) == set(expected_shapes)
    assert dict(mod.named_buffers()) == {}
    assert {k: tuple(v.shape) for k, v in mod.named_parameters()} == expected_shapes
    fresh = make_module(is_outproj_norm, rope_fraction)
    fresh.load_state_dict(mod.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(fresh(u), mod(u), atol=0, rtol=0)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
def test_float32_and_fp16_autocast(device):
    inputs64 = {k: v.to(device) for k, v in make_inputs().items()}
    inputs32 = {k: v.float() for k, v in inputs64.items()}
    out64 = mamba3_siso_chunked(**inputs64, chunk_size=8)
    out32 = mamba3_siso_chunked(**inputs32, chunk_size=8)
    assert out32.dtype == torch.float32
    relative_error = (out32.double() - out64).norm() / out64.norm()
    assert relative_error < 1e-4
    with torch.autocast(device_type=device, dtype=torch.float16):
        autocast_out = mamba3_siso_chunked(**inputs32, chunk_size=8)
    torch.testing.assert_close(autocast_out, out32, atol=0, rtol=0)
    for dtype in (torch.float16, torch.bfloat16):
        low_inputs = {k: v.to(dtype) for k, v in inputs32.items()}
        promoted = {k: v.float() for k, v in low_inputs.items()}
        with torch.autocast(device_type=device, dtype=torch.float16):
            low_out, states = mamba3_siso_chunked(**low_inputs, chunk_size=8, return_final_states=True)
        expected = mamba3_siso_chunked(**promoted, chunk_size=8)
        assert low_out.dtype == dtype and all(s.dtype == torch.float32 for s in states)
        torch.testing.assert_close(low_out, expected.to(dtype), atol=0, rtol=0)

    model = Mamba3Stack(32, 2, d_state=16, headdim=8, chunk_size=8, device=device)
    u = torch.randn(2, 13, 32, device=device, requires_grad=True)
    with torch.autocast(device_type=device, dtype=torch.float16):
        result = model(u)
        loss = result.float().square().mean()
    loss.backward()
    assert torch.isfinite(result).all() and torch.isfinite(u.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("is_outproj_norm", [False, True])
@pytest.mark.parametrize("rope_fraction", [0.5, 1.0])
def test_stack_step_and_carry(is_outproj_norm, rope_fraction):
    model = Mamba3Stack(
        32, 3, d_state=16, headdim=8, chunk_size=8,
        is_outproj_norm=is_outproj_norm, rope_fraction=rope_fraction,
    ).double()
    u = torch.randn(2, 37, 32, dtype=torch.float64)
    with torch.no_grad():
        out, final_states = model(u, return_final_states=True)
        states = model.allocate_states(2)
        outputs = []
        for t in range(u.shape[1]):
            y, states = model.step(u[:, t], states)
            outputs.append(y)
        torch.testing.assert_close(out, torch.stack(outputs, 1), atol=1e-9, rtol=0)
        for actual, expected in zip(final_states, states):
            assert_result_close((out, actual), (out, expected), atol=1e-9, rtol=0)
        first, states = model(u[:, :20], return_final_states=True)
        second, states = model(u[:, 20:], initial_states=states, return_final_states=True)
        torch.testing.assert_close(out, torch.cat((first, second), 1), atol=1e-9, rtol=0)
        for actual, expected in zip(final_states, states):
            assert_result_close((out, actual), (out, expected), atol=1e-9, rtol=0)
        torch.testing.assert_close(out, model(u), atol=0, rtol=0)


def test_block_residual():
    block = Mamba3Block(32, d_state=16, headdim=8, chunk_size=8).double()
    u = torch.randn(2, 7, 32, dtype=torch.float64)
    with torch.no_grad():
        expected = u + block.mamba(block.norm(u))
        torch.testing.assert_close(block(u), expected, atol=0, rtol=0)


def test_mimo_not_ported():
    with pytest.raises(NotImplementedError, match="^MIMO not ported yet$"):
        Mamba3(32, is_mimo=True)
