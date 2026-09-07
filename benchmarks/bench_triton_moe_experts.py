"""Benchmark: fused MoE expert path (Triton) vs eager per-expert loop.

Timing covers the whole expert stage (align + GEMM1 + silu + GEMM2 +
combine) for both paths. Expert weights are kept small enough for a laptop
GPU; shapes echo Qwen/V2-Lite proportions at reduced width.

Usage (WSL 5060 / gongga 4090)::

    python benchmarks/bench_triton_moe_experts.py

The Triton rows report SKIPPED until the kernel body in
``triton_moe.moe_grouped_gemm_kernel`` is filled in.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

# Bare-script runs bypass pytest's pythonpath=["src"] (pyproject), and some
# environments carry only the maturin-installed control-plane half of the
# package in site-packages. Bootstrap the source tree explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from einf.executors.torch.triton_moe import (
    moe_expert_mlp_reference,
    moe_expert_mlp_triton,
)

# (num_tokens, num_experts, top_k, h_dim, inter)
_SHAPES = [
    (1, 8, 8, 512, 256),
    (64, 8, 8, 512, 256),  # decode step c64
    (512, 8, 8, 512, 256),
    (4096, 8, 8, 512, 256),  # prefill chunk aggregate
]

_WARMUP = 10
_ITERS = 50


def _time(fn) -> float:
    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(_ITERS):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required"
    device = torch.device("cuda")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"warmup={_WARMUP} iters={_ITERS} (median ms)")
    print()

    header = (
        f"{'shape':>26} {'eager ms':>10} {'triton ms':>10} {'speedup':>8}"
    )
    print(header)

    for num_tokens, num_experts, top_k, h_dim, inter in _SHAPES:
        generator = torch.Generator(device=device).manual_seed(num_tokens)
        hidden = torch.randn(
            (num_tokens, h_dim), generator=generator, device=device, dtype=torch.bfloat16
        )
        logits = torch.randn(
            (num_tokens, num_experts), generator=generator, device=device
        )
        topk_weights, topk_ids = torch.topk(logits, top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
        w13 = torch.randn(
            (num_experts, 2 * inter, h_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ) * 0.05
        w2 = torch.randn(
            (num_experts, h_dim, inter),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        ) * 0.05

        eager = lambda: moe_expert_mlp_reference(  # noqa: E731
            hidden, topk_weights, topk_ids, w13, w2
        )
        fused = lambda: moe_expert_mlp_triton(  # noqa: E731
            hidden, topk_weights, topk_ids, w13, w2
        )

        eager_ms = _time(eager)
        row = f"{f'({num_tokens},{num_experts},{top_k},{h_dim},{inter})':>26}"
        row += f" {eager_ms:>10.4f}"

        try:
            fused_out = fused()
            ref_out = moe_expert_mlp_reference(
                hidden, topk_weights, topk_ids, w13, w2,
                quantize_intermediates=True,
            )
            rel = (
                (fused_out.float() - ref_out.float()).norm(dim=-1)
                / ref_out.float().norm(dim=-1).clamp_min(1e-6)
            )
            assert rel.max() < 0.02, f"row rel-L2 {rel.max():.4e}"
            fused_ms = _time(fused)
            row += f" {fused_ms:>10.4f} {eager_ms / fused_ms:>7.2f}x"
        except Exception as exc:  # noqa: BLE001 - scaffold reporting
            first = str(exc).splitlines()[0][:70]
            row += f" {'SKIPPED':>10} {'':>8}"
            row += f"  # {first}"

        print(row)


if __name__ == "__main__":
    main()
