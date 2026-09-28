# Pure-PyTorch Mamba-3 (SISO)

A differentiable SISO port of [state-spaces/mamba](https://github.com/state-spaces/mamba/tree/e9594ce),
commit `e9594ce`, implementing Mamba-3 (Lahoti et al., 2026, arXiv:2603.15569).
Licensed under Apache-2.0; see [_official/LICENSE](_official/LICENSE).
The original module and unmodified PyTorch test oracles are included for provenance.
This port uses PyTorch and einops, with no Triton, mamba_ssm, custom CUDA, or additional dependencies.
CPU and ordinary PyTorch CUDA backends are supported; MIMO is not ported.

Install with `pip install -e .` from the repo root, or put the repo root on `PYTHONPATH`:

```python
import torch
from mamba3_torch import Mamba3, Mamba3Block, Mamba3Stack

model = Mamba3(32, d_state=16, headdim=8, chunk_size=64)
u = torch.randn(2, 37, 32)
y, states = model(u, return_final_states=True)
y_next, states = model.step(torch.randn(2, 32), states)

# Carry states between sequence segments without detaching the graph.
y_more, states = model(u, initial_states=states, return_final_states=True)
states = model.allocate_states(batch=2)  # Zero states for a fresh rollout.

stack = Mamba3Stack(32, n_layer=4, d_state=16, headdim=8)
y, layer_states = stack(u, return_final_states=True)
y_next, layer_states = stack.step(u[:, 0], layer_states)
```

`Mamba3` retains the official SISO constructor and state_dict names/shapes;
official SISO weights load with `strict=True`. `Mamba3Block` implements
`x + Mamba3(RMSNormGated(x))`; the stack adds a final RMSNorm. Blocks and stacks
share the forward/step interface, with one state tuple per stack layer.

The exported `mamba3_siso_chunked` and `mamba3_siso_step` accept the raw `Trap`
and `Angles` projections. Q/K have shape `(b,L,hq,n)`, V/Z `(b,L,h,p)`,
ADT/DT/Trap `(b,h,L)`, biases `(h,n)`, Angles `(b,L,h,na)`, and D `(h,)`.
Q/K heads are repeated for GQA; `hq` must divide `h`. Step inputs omit L.
Chunked returns the output, or `(output, states)` with `return_final_states=True`;
step always returns both. `D`, `Z`/`z`, and input states are optional.

States are `(Angle_State, SSM_State, K_State, V_State)`, with shapes
`(b,h,na)`, `(b,h,p,n)`, `(b,h,n)`, `(b,h,p)`. K is the last rotated key;
its pending trapezoid contribution is kept out of the SSM state until the next
token. States are returned without in-place mutation or detaching.
The scan uses float64 for float64 V, otherwise float32, and returns output in
V's dtype. States and `allocate_states` use the scan's compute dtype.
The official module's float32 A/angle projection casts and RMSNorm's float32
normalization are preserved, including for double modules.

Chunked SSD uses batched operations within chunks and a loop over chunk
summaries. Its attention memory is O(L × chunk_size); sequence length need
not divide chunk_size. Nonempty, fixed-length batches are supported.

From the repo root, run the test suite (the CUDA smoke test skips on a CPU-only node):

```bash
python -m pytest -q tests
```

Verified 2026-09-27 (torch 2.5.1, CUDA 12.4): 346 passed on NVIDIA V100 and A100,
345 passed + 1 CUDA skip on CPU.

Benchmark forward plus backward separately from tests:

```bash
python bench.py --device cuda --batch 32 --seqlen 128 --d_model 256 --n_layer 4 --dtype float32
```

The benchmark prints milliseconds per iteration and, on CUDA, peak allocated
memory. Use `--device cpu` on a CPU node and `--warmup`/`--steps` to change repeats.
