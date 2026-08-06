from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from einf.executors.torch.model_runner import build_causal_mask, repeat_kv
from einf.executors.torch.ops import (
    contiguous_attention,
    gather_context,
    load_custom_ops,
    paged_decode_attention,
    paged_decode_attention_split_kv,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

CONTEXT_LENGTHS = (128, 512, 2048, 8192)


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


def qwen_eager_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    num_attention_heads = Q.shape[0]
    groups = num_attention_heads // K.shape[1]
    context_len = K.shape[0]
    request_Q = Q.unsqueeze(1)
    context_K = repeat_kv(K, groups).transpose(0, 1)
    context_V = repeat_kv(V, groups).transpose(0, 1)
    scores = (request_Q @ context_K.transpose(-1, -2)) * scale
    scores = scores + build_causal_mask(
        1,
        context_len - 1,
        device=Q.device,
        dtype=scores.dtype,
    )
    probabilities = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
    return (probabilities @ context_V).squeeze(1)


def sdpa_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        Q.unsqueeze(0).unsqueeze(2),
        K.transpose(0, 1).unsqueeze(0),
        V.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        enable_gqa=Q.shape[0] != K.shape[1],
    )
    return output.squeeze(0).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark gathered Decode Attention against direct paged reads"
    )
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--num-attention-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument(
        "--context-lengths",
        default=",".join(str(value) for value in CONTEXT_LENGTHS),
        help="comma-separated Context lengths to benchmark",
    )
    parser.add_argument(
        "--splits",
        default="1,2,4,8,16,32,64,128",
        help="comma-separated Split-KV counts to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.head_dim > 256 or args.head_dim % 32 != 0:
        raise ValueError("Paged Decode v0 requires D <= 256 and divisible by 32")
    if args.num_attention_heads % args.num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    split_counts = tuple(
        dict.fromkeys(int(value) for value in args.splits.split(","))
    )
    if not split_counts or any(value <= 0 for value in split_counts):
        raise ValueError("--splits must contain positive integers")
    context_lengths = tuple(
        dict.fromkeys(int(value) for value in args.context_lengths.split(","))
    )
    if not context_lengths or any(value <= 0 for value in context_lengths):
        raise ValueError("--context-lengths must contain positive integers")

    torch.manual_seed(0)
    load_custom_ops()
    dtype = DTYPES[args.dtype]
    scale = 1.0 / math.sqrt(args.head_dim)
    if args.quick:
        context_lengths = context_lengths[:2]
    force_flash_sdpa = dtype in (torch.float16, torch.bfloat16)
    sdpa_context = (
        sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if force_flash_sdpa
        else nullcontext()
    )
    sdpa_name = "gather+flash" if force_flash_sdpa else "gather+sdpa"

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(
        f"Hq={args.num_attention_heads}, Hkv={args.num_kv_heads}, "
        f"D={args.head_dim}, block_len={args.block_len}, "
        f"warmup={args.warmup}, iterations={args.iterations}"
    )
    print(f"Split-KV counts: {split_counts}")
    print(f"Context lengths: {context_lengths}")
    print("gather+qwen includes custom gather, repeat_kv, and mask construction.")
    print(
        "SDPA is forced to FLASH_ATTENTION for FP16/BF16; "
        "FP32 uses automatic backend selection."
    )
    print(
        f"{'context':>9} {'gather+qwen':>13} {sdpa_name:>13} "
        f"{'gather+naive':>14} {'paged':>11} {'sdpa/paged':>12}"
    )

    split_rows: list[tuple[int, float, float, dict[int, float]]] = []

    with torch.inference_mode(), sdpa_context:
        for context_len in context_lengths:
            num_logical_blocks = (
                context_len + args.block_len - 1
            ) // args.block_len
            num_physical_blocks = num_logical_blocks + 8
            block_table = torch.randperm(
                num_physical_blocks,
                device="cuda",
                dtype=torch.long,
            )[:num_logical_blocks].contiguous()
            Q = torch.randn(
                (args.num_attention_heads, args.head_dim),
                device="cuda",
                dtype=dtype,
            )
            K_cache = torch.randn(
                (
                    num_physical_blocks,
                    args.block_len,
                    args.num_kv_heads,
                    args.head_dim,
                ),
                device="cuda",
                dtype=dtype,
            )
            V_cache = torch.randn_like(K_cache)

            def run_gather_qwen() -> torch.Tensor:
                K, V = gather_context(
                    K_cache,
                    V_cache,
                    block_table,
                    context_len,
                )
                return qwen_eager_decode(Q, K, V, scale=scale)

            def run_gather_naive() -> torch.Tensor:
                K, V = gather_context(
                    K_cache,
                    V_cache,
                    block_table,
                    context_len,
                )
                return contiguous_attention(
                    Q.unsqueeze(0),
                    K,
                    V,
                    context_len - 1,
                    scale,
                ).squeeze(0)

            def run_gather_sdpa() -> torch.Tensor:
                K, V = gather_context(
                    K_cache,
                    V_cache,
                    block_table,
                    context_len,
                )
                return sdpa_decode(Q, K, V, scale=scale)

            def run_paged() -> torch.Tensor:
                return paged_decode_attention(
                    Q,
                    K_cache,
                    V_cache,
                    block_table,
                    context_len,
                    scale,
                )

            expected = run_gather_qwen()
            sdpa_output = run_gather_sdpa()
            naive_output = run_gather_naive()
            paged_output = run_paged()
            if dtype == torch.float32:
                rtol = atol = 1e-4
            else:
                rtol = atol = 3e-2
            torch.testing.assert_close(naive_output, expected, rtol=rtol, atol=atol)
            torch.testing.assert_close(sdpa_output, expected, rtol=rtol, atol=atol)
            torch.testing.assert_close(paged_output, expected, rtol=rtol, atol=atol)

            split_times: dict[int, float] = {}
            for num_splits in split_counts:
                if num_splits > num_logical_blocks:
                    continue

                def run_split_kv(current_splits: int = num_splits) -> torch.Tensor:
                    return paged_decode_attention_split_kv(
                        Q,
                        K_cache,
                        V_cache,
                        block_table,
                        context_len,
                        current_splits,
                        scale,
                    )

                split_output = run_split_kv()
                torch.testing.assert_close(
                    split_output,
                    expected,
                    rtol=rtol,
                    atol=atol,
                )
                split_times[num_splits] = benchmark_us(
                    run_split_kv,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )

            qwen_us = benchmark_us(
                run_gather_qwen,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            naive_us = benchmark_us(
                run_gather_naive,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            sdpa_us = benchmark_us(
                run_gather_sdpa,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            paged_us = benchmark_us(
                run_paged,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(
                f"{context_len:9d} {qwen_us:13.3f} {sdpa_us:13.3f} "
                f"{naive_us:14.3f} {paged_us:11.3f} "
                f"{sdpa_us / paged_us:12.2f}"
            )
            split_rows.append((context_len, sdpa_us, paged_us, split_times))

    print()
    print("Split-KV includes Stage 1, FP32 workspace, and Stage 2 reduction.")
    split_headers = " ".join(f"s={value:>2}" for value in split_counts)
    print(
        f"{'context':>9} {'paged':>10} {split_headers} "
        f"{'best-s':>7} {'best':>10} {'paged/best':>11} {'sdpa/best':>10}"
    )
    for context_len, sdpa_us, paged_us, split_times in split_rows:
        if not split_times:
            continue
        best_splits, best_us = min(split_times.items(), key=lambda item: item[1])
        split_values = " ".join(
            f"{split_times[value]:5.1f}" if value in split_times else f"{'-':>5}"
            for value in split_counts
        )
        print(
            f"{context_len:9d} {paged_us:10.3f} {split_values} "
            f"{best_splits:7d} {best_us:10.3f} "
            f"{paged_us / best_us:11.2f} {sdpa_us / best_us:10.2f}"
        )


if __name__ == "__main__":
    main()
