"""Paired benchmark for the fused MoE routing kernel vs torch composition.

Timing reports tokens routed per second; the data volume is tiny, so
bandwidth is not the interesting metric — launch overhead and the
in-register top-k loop dominate. The torch baseline is the reference
composition (softmax + topk + gather + renorm), which is what eager
inference would run.

Usage (WSL 5060 / gongga 4090)::

    python benchmarks/bench_triton_moe_routing.py

The Triton row reports SKIPPED until the kernel body in
``triton_ops.moe_topk_softmax_kernel`` is filled in.
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

from einf.executors.torch.triton_ops import (
    moe_topk_softmax_reference,
    moe_topk_softmax_triton,
)

# (num_tokens, num_experts, top_k)
_SHAPES = [
    (1, 256, 8),  # DeepSeek-V3, single token
    (64, 256, 8),  # decode step c64
    (512, 256, 8),
    (4096, 256, 8),  # prefill chunk aggregate
    (4096, 64, 6),  # DeepSeek-V2-Lite
]

_WARMUP = 10
_ITERS = 50


def _time(fn, *args) -> float:
    for _ in range(_WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(_ITERS):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _torch_routing(logits, bias, top_k, scaling):
    weights, ids = moe_topk_softmax_reference(
        logits, bias, top_k=top_k, scaling=scaling
    )
    return weights, ids


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required"
    device = torch.device("cuda")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"warmup={_WARMUP} iters={_ITERS} (median ms)")
    print()

    header = (
        f"{'shape':>20} {'torch ms':>10} {'triton ms':>10} "
        f"{'tok/s':>12} {'speedup':>8}"
    )
    print(header)

    for num_tokens, num_experts, top_k in _SHAPES:
        logits = torch.randn((num_tokens, num_experts), device=device)
        bias = torch.randn(num_experts, device=device)
        scaling = 2.5

        row = f"{f'({num_tokens},{num_experts},{top_k})':>20}"

        elapsed = _time(_torch_routing, logits, bias, top_k, scaling)
        row += f" {elapsed:>10.4f}"

        try:
            triton_routing = lambda: moe_topk_softmax_triton(  # noqa: E731
                logits, bias, top_k=top_k, scaling=scaling
            )
            weights, ids = triton_routing()
            expected_w, expected_ids = moe_topk_softmax_reference(
                logits, bias, top_k=top_k, scaling=scaling
            )
            torch.testing.assert_close(weights, expected_w, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(
                ids.float(), expected_ids.float(), rtol=0.0, atol=0.0
            )

            elapsed = _time(triton_routing)
            row += f" {elapsed:>10.4f}"
            row += f" {num_tokens / (elapsed * 1e-3):>12.0f}"
            torch_elapsed = _time(_torch_routing, logits, bias, top_k, scaling)
            row += f" {torch_elapsed / elapsed:>7.2f}x"
        except Exception as exc:  # noqa: BLE001 - scaffold reporting
            first = str(exc).splitlines()[0][:60]
            row += f" {'SKIPPED':>10} {'':>12} {'':>8}"
            row += f"  # {first}"

        print(row)
        del logits, bias


if __name__ == "__main__":
    main()
