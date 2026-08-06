from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from einf.executors.torch.model_runner import build_causal_mask, repeat_kv
from einf.executors.torch.ops import (
    contiguous_attention,
    flash_attention,
    load_custom_ops,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class AttentionCase:
    name: str
    q_len: int
    kv_len: int


CASES = (
    AttentionCase("decode-128", 1, 128),
    AttentionCase("decode-512", 1, 512),
    AttentionCase("chunk-16/128", 16, 128),
    AttentionCase("chunk-64/512", 64, 512),
    AttentionCase("prefill-128", 128, 128),
    AttentionCase("prefill-512", 512, 512),
)


def benchmark_us(
    function: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def qwen_eager_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    start_pos: int,
    scale: float,
) -> torch.Tensor:
    q_len, num_attention_heads, _ = Q.shape
    _, num_kv_heads, _ = K.shape
    groups = num_attention_heads // num_kv_heads

    request_Q = Q.transpose(0, 1)
    context_K = repeat_kv(K, groups).transpose(0, 1)
    context_V = repeat_kv(V, groups).transpose(0, 1)
    scores = (request_Q @ context_K.transpose(-1, -2)) * scale
    scores = scores + build_causal_mask(
        q_len,
        start_pos,
        device=Q.device,
        dtype=scores.dtype,
    )
    probabilities = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
    return (probabilities @ context_V).transpose(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen eager, naive native, and FlashAttention paths"
    )
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--num-attention-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the two largest Prefill cases",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.head_dim != 64:
        raise ValueError("FlashAttention v0 requires --head-dim 64")
    if args.num_attention_heads % args.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")

    torch.manual_seed(0)
    load_custom_ops()
    dtype = DTYPES[args.dtype]
    scale = 1.0 / math.sqrt(args.head_dim)
    cases = CASES[:4] if args.quick else CASES

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(
        f"Hq={args.num_attention_heads}, Hkv={args.num_kv_heads}, "
        f"D={args.head_dim}, warmup={args.warmup}, iterations={args.iterations}"
    )
    print("Qwen eager includes repeat_kv and causal-mask construction.")
    print(
        f"{'case':>16} {'qwen_us':>11} {'naive_us':>11} {'flash_us':>11} "
        f"{'qwen/flash':>11} {'naive/flash':>12}"
    )

    with torch.inference_mode():
        for case in cases:
            Q = torch.randn(
                (case.q_len, args.num_attention_heads, args.head_dim),
                dtype=dtype,
                device="cuda",
            )
            K = torch.randn(
                (case.kv_len, args.num_kv_heads, args.head_dim),
                dtype=dtype,
                device="cuda",
            )
            V = torch.randn_like(K)
            start_pos = case.kv_len - case.q_len

            def run_qwen() -> torch.Tensor:
                return qwen_eager_attention(
                    Q,
                    K,
                    V,
                    start_pos=start_pos,
                    scale=scale,
                )

            def run_naive() -> torch.Tensor:
                return contiguous_attention(Q, K, V, start_pos, scale)

            def run_flash() -> torch.Tensor:
                return flash_attention(Q, K, V, start_pos, scale)

            expected = run_qwen()
            naive_output = run_naive()
            flash_output = run_flash()
            if dtype == torch.float32:
                rtol = atol = 1e-4
            else:
                rtol = atol = 3e-2
            torch.testing.assert_close(naive_output, expected, rtol=rtol, atol=atol)
            torch.testing.assert_close(flash_output, expected, rtol=rtol, atol=atol)

            qwen_us = benchmark_us(
                run_qwen,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            naive_us = benchmark_us(
                run_naive,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            flash_us = benchmark_us(
                run_flash,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(
                f"{case.name:>16} {qwen_us:11.3f} {naive_us:11.3f} "
                f"{flash_us:11.3f} {qwen_us / flash_us:11.2f} "
                f"{naive_us / flash_us:12.2f}"
            )


if __name__ == "__main__":
    main()
