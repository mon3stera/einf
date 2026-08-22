from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from einf.executors.torch.model_runner import build_causal_mask, repeat_kv
from einf.executors.torch.ops import (
    contiguous_attention,
    flash_attention,
    load_custom_ops,
)


DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class AttentionCase:
    name: str
    q_len: int
    kv_len: int


CASES = (
    AttentionCase("prefill-128", 128, 128),
    AttentionCase("prefill-512", 512, 512),
    AttentionCase("prefill-2048", 2048, 2048),
    AttentionCase("prefill-4096", 4096, 4096),
    AttentionCase("chunk-128/512", 128, 512),
    AttentionCase("chunk-128/2048", 128, 2048),
    AttentionCase("chunk-128/8192", 128, 8192),
    AttentionCase("chunk-128/32768", 128, 32768),
)

QUICK_CASE_NAMES = {
    "prefill-128",
    "prefill-512",
    "chunk-128/512",
    "chunk-128/2048",
}


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


def pytorch_flash_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    causal_bias: object,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        Q.transpose(0, 1).unsqueeze(0),
        K.transpose(0, 1).unsqueeze(0),
        V.transpose(0, 1).unsqueeze(0),
        attn_mask=causal_bias,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=True,
    )
    return output.squeeze(0).transpose(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Qwen eager, PyTorch forced Flash SDPA, and einf's "
            "correctness-first native FlashAttention"
        )
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
        help="run two small full-Prefill and two small chunked-Prefill cases",
    )
    parser.add_argument(
        "--include-naive",
        action="store_true",
        help="also time the three-kernel native oracle (expensive on large cases)",
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
    cases = tuple(
        case for case in CASES if not args.quick or case.name in QUICK_CASE_NAMES
    )

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(
        f"Hq={args.num_attention_heads}, Hkv={args.num_kv_heads}, "
        f"D={args.head_dim}, warmup={args.warmup}, iterations={args.iterations}"
    )
    print("Qwen eager includes repeat_kv and causal-mask construction.")
    print("PyTorch SDPA is forced to SDPBackend.FLASH_ATTENTION.")
    print(
        f"{'case':>18} {'eager_us':>11} {'sdpa_us':>11} {'native_us':>11} "
        f"{'native_%':>10} {'eager/native':>13}"
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
            causal_bias = causal_lower_right(case.q_len, case.kv_len)

            def run_eager() -> torch.Tensor:
                return qwen_eager_attention(
                    Q,
                    K,
                    V,
                    start_pos=start_pos,
                    scale=scale,
                )

            def run_sdpa() -> torch.Tensor:
                return pytorch_flash_sdpa(
                    Q,
                    K,
                    V,
                    scale=scale,
                    causal_bias=causal_bias,
                )

            def run_native() -> torch.Tensor:
                return flash_attention(Q, K, V, start_pos, scale)

            expected = run_eager()
            native_output = run_native()
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                sdpa_output = run_sdpa()

            rtol = atol = 3e-2
            torch.testing.assert_close(
                native_output,
                expected,
                rtol=rtol,
                atol=atol,
            )
            torch.testing.assert_close(
                sdpa_output,
                expected,
                rtol=rtol,
                atol=atol,
            )

            eager_us = benchmark_us(
                run_eager,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                sdpa_us = benchmark_us(
                    run_sdpa,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            native_us = benchmark_us(
                run_native,
                warmup=args.warmup,
                iterations=args.iterations,
            )

            print(
                f"{case.name:>18} {eager_us:11.3f} {sdpa_us:11.3f} "
                f"{native_us:11.3f} {100 * sdpa_us / native_us:9.1f}% "
                f"{eager_us / native_us:13.2f}"
            )

            if args.include_naive:
                naive_output = contiguous_attention(Q, K, V, start_pos, scale)
                torch.testing.assert_close(
                    naive_output,
                    expected,
                    rtol=rtol,
                    atol=atol,
                )
                naive_us = benchmark_us(
                    lambda: contiguous_attention(Q, K, V, start_pos, scale),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                print(f"{'three-kernel oracle':>18} {naive_us:11.3f} us")


if __name__ == "__main__":
    main()
