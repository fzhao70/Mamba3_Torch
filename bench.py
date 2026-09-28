"""Time complete forward/backward passes; run directly from the checkout."""

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mamba3_torch import Mamba3Stack  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--mimo_rank", type=int, default=0, help="MIMO rank; 0 selects SISO")
    parser.add_argument("--dtype", choices=["float32", "float64", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch, args.seqlen, args.d_model, args.n_layer, args.steps) < 1 or args.warmup < 0:
        parser.error("sizes and steps must be positive; warmup must be nonnegative")
    if args.mimo_rank < 0:
        parser.error("mimo_rank must be nonnegative (0 selects SISO)")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable on this node")
    torch.manual_seed(0)
    model = Mamba3Stack(
        args.d_model, args.n_layer, is_mimo=args.mimo_rank > 0, mimo_rank=args.mimo_rank,
    ).to(device=device, dtype=dtype)
    model.train()
    u = torch.randn(args.batch, args.seqlen, args.d_model, device=device, dtype=dtype, requires_grad=True)

    def iteration():
        model.zero_grad(set_to_none=True)
        u.grad = None
        model(u).float().square().mean().backward()

    for _ in range(args.warmup):
        iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        iteration()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000 / args.steps
    print(
        f"device={device} dtype={args.dtype} batch={args.batch} seqlen={args.seqlen} "
        f"d_model={args.d_model} n_layer={args.n_layer} mimo_rank={args.mimo_rank}"
    )
    print(f"Forward + backward: {elapsed_ms:.3f} ms/iteration ({args.steps} iterations)")
    if device.type == "cuda":
        print(f"Peak allocated CUDA memory: {torch.cuda.max_memory_allocated(device) / 2**20:.2f} MiB")


if __name__ == "__main__":
    main()
