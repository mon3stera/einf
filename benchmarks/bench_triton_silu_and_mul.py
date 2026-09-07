"""Paired A/B benchmark for the hand-written Triton ``silu_and_mul``.

Same-session paired comparison per the project convention: every
implementation runs interleaved on the same shapes with CUDA-event timing
after warmup (which also absorbs Triton's first-call JIT compile).

Usage (gongga 4090)::

    python benchmarks/bench_triton_silu_and_mul.py

The Triton row reports SKIPPED until the kernel body in
``triton_ops.silu_and_mul_kernel`` is filled in.
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

from einf.executors.torch.fused_ops import use_flashinfer_fused
from einf.executors.torch.triton_ops import (
    silu_and_mul_reference,
    silu_and_mul_triton,
)

_SHAPES = [
    # (num_tokens, hidden) — hidden = model intermediate_size
    (1, 18944),  # Qwen2.5-7B, single token
    (64, 18944),  # c64 decode step
    (512, 18944),  # c512 batched decode
    (4096, 18944),  # prefill chunk aggregate
    (4096, 10944),  # DeepSeek-V2-Lite intermediate
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


def _torch_composed(gate_up: torch.Tensor) -> torch.Tensor:
    hidden = gate_up.shape[-1] // 2
    return torch.nn.functional.silu(gate_up[..., :hidden]) * gate_up[..., hidden:]


def _flashinfer_impl(gate_up: torch.Tensor) -> torch.Tensor | None:
    if not use_flashinfer_fused(gate_up):
        return None
    import flashinfer

    return flashinfer.silu_and_mul(gate_up)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required"
    device = torch.device("cuda")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"warmup={_WARMUP} iters={_ITERS} (median ms)")
    print()

    header = (
        f"{'shape':>22} {'dtype':>8} {'torch ms':>10} {'fi ms':>10} "
        f"{'triton ms':>10} {'triton GB/s':>12}"
    )
    print(header)

    for num_tokens, hidden in _SHAPES:
        for dtype in (torch.bfloat16,):
            gate_up = torch.randn(
                (num_tokens, 2 * hidden), device=device, dtype=dtype
            )

            row = f"{f'({num_tokens},{hidden})':>22} {str(dtype)[6:]:>8}"

            elapsed = _time(_torch_composed, gate_up)
            row += f" {elapsed:>10.4f}"

            fi_impl = _flashinfer_impl(gate_up)
            if fi_impl is not None:
                elapsed = _time(_flashinfer_impl, gate_up)
                row += f" {elapsed:>10.4f}"
            else:
                row += f" {'n/a':>10}"

            try:
                out = silu_and_mul_triton(gate_up)
                expected = silu_and_mul_reference(gate_up)
                torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)
                elapsed = _time(silu_and_mul_triton, gate_up)
                row += f" {elapsed:>10.4f}"
                bytes_moved = num_tokens * 3 * hidden * dtype.itemsize
                row += f" {bytes_moved / (elapsed * 1e-3) / 1e9:>12.1f}"
            except Exception as exc:  # noqa: BLE001 - scaffold reporting
                first = str(exc).splitlines()[0][:60]
                row += f" {'SKIPPED':>10}"
                row += f" {'(' + first + ')':>12}"

            print(row)
            del gate_up


if __name__ == "__main__":
    main()
