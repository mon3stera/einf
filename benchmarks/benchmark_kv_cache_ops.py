from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch

from einf.executors.torch.ops import load_custom_ops


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
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


def effective_bandwidth_gbps(
    *,
    num_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    element_size: int,
    elapsed_us: float,
) -> float:
    # Read K/V and write K/V: four tensor element transfers.
    transferred_bytes = 4 * num_tokens * num_kv_heads * head_dim * element_size
    return transferred_bytes / elapsed_us / 1000


def benchmark_write(
    *,
    token_counts: list[int],
    block_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> None:
    write_op = torch.ops.einf.write_slots_
    max_tokens = max(token_counts)
    num_blocks = math.ceil(max_tokens / block_len) + 16
    num_slots = num_blocks * block_len

    print("\nwrite_slots_ (two PyTorch index_copy_ calls vs one custom kernel)")
    print(f"{'tokens':>8} {'pytorch_us':>12} {'custom_us':>12} {'speedup':>9} {'custom_GB/s':>12}")

    for num_tokens in token_counts:
        K_cache_reference = torch.empty(
            (num_blocks, block_len, num_kv_heads, head_dim),
            dtype=dtype,
            device="cuda",
        )
        V_cache_reference = torch.empty_like(K_cache_reference)
        K_cache_custom = torch.empty_like(K_cache_reference)
        V_cache_custom = torch.empty_like(K_cache_reference)
        K = torch.randn(
            (num_tokens, num_kv_heads, head_dim),
            dtype=dtype,
            device="cuda",
        )
        V = torch.randn_like(K)
        slot_mapping = torch.randperm(num_slots, device="cuda")[:num_tokens]

        flat_K_reference = K_cache_reference.view(-1, num_kv_heads, head_dim)
        flat_V_reference = V_cache_reference.view(-1, num_kv_heads, head_dim)

        def pytorch_write() -> None:
            flat_K_reference.index_copy_(0, slot_mapping, K)
            flat_V_reference.index_copy_(0, slot_mapping, V)

        def custom_write() -> None:
            write_op(K_cache_custom, V_cache_custom, slot_mapping, K, V)

        pytorch_us = benchmark_us(pytorch_write, warmup=warmup, iterations=iterations)
        custom_us = benchmark_us(custom_write, warmup=warmup, iterations=iterations)
        bandwidth = effective_bandwidth_gbps(
            num_tokens=num_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            element_size=K.element_size(),
            elapsed_us=custom_us,
        )
        print(
            f"{num_tokens:8d} {pytorch_us:12.3f} {custom_us:12.3f} "
            f"{pytorch_us / custom_us:9.2f} {bandwidth:12.2f}"
        )


def benchmark_gather(
    *,
    context_lengths: list[int],
    block_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> None:
    gather_op = torch.ops.einf.gather_context

    print("\ngather_context (full PyTorch mapping, precomputed slots, custom kernel)")
    print(
        f"{'context':>8} {'full_us':>11} {'slots_us':>11} {'custom_us':>11} "
        f"{'vs_full':>9} {'custom_GB/s':>12}"
    )

    for context_len in context_lengths:
        required_blocks = math.ceil(context_len / block_len)
        num_blocks = required_blocks + max(16, required_blocks // 4)
        K_cache = torch.randn(
            (num_blocks, block_len, num_kv_heads, head_dim),
            dtype=dtype,
            device="cuda",
        )
        V_cache = torch.randn_like(K_cache)
        block_table = torch.randperm(num_blocks, device="cuda")[:required_blocks]
        flat_K = K_cache.view(-1, num_kv_heads, head_dim)
        flat_V = V_cache.view(-1, num_kv_heads, head_dim)

        positions = torch.arange(context_len, dtype=torch.long, device="cuda")
        logical_blocks = positions // block_len
        block_offsets = positions % block_len
        physical_blocks = block_table[logical_blocks]
        slots = physical_blocks * block_len + block_offsets

        def pytorch_full_gather() -> tuple[torch.Tensor, torch.Tensor]:
            current_positions = torch.arange(
                context_len,
                dtype=torch.long,
                device="cuda",
            )
            current_logical_blocks = current_positions // block_len
            current_block_offsets = current_positions % block_len
            current_physical_blocks = block_table[current_logical_blocks]
            current_slots = current_physical_blocks * block_len + current_block_offsets
            return (
                flat_K.index_select(0, current_slots),
                flat_V.index_select(0, current_slots),
            )

        def pytorch_precomputed_slots() -> tuple[torch.Tensor, torch.Tensor]:
            return flat_K.index_select(0, slots), flat_V.index_select(0, slots)

        def custom_gather() -> tuple[torch.Tensor, torch.Tensor]:
            return gather_op(K_cache, V_cache, block_table, context_len)

        full_us = benchmark_us(
            pytorch_full_gather,
            warmup=warmup,
            iterations=iterations,
        )
        slots_us = benchmark_us(
            pytorch_precomputed_slots,
            warmup=warmup,
            iterations=iterations,
        )
        custom_us = benchmark_us(custom_gather, warmup=warmup, iterations=iterations)
        bandwidth = effective_bandwidth_gbps(
            num_tokens=context_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            element_size=K_cache.element_size(),
            elapsed_us=custom_us,
        )
        print(
            f"{context_len:8d} {full_us:11.3f} {slots_us:11.3f} "
            f"{custom_us:11.3f} {full_us / custom_us:9.2f} {bandwidth:12.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark einf KV Cache CUDA ops")
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(0)
    load_custom_ops()
    dtype = DTYPES[args.dtype]

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(
        f"block_len: {args.block_len}, num_kv_heads: {args.num_kv_heads}, "
        f"head_dim: {args.head_dim}, warmup: {args.warmup}, "
        f"iterations: {args.iterations}"
    )

    benchmark_write(
        token_counts=[1, 8, 32, 128, 512, 2048],
        block_len=args.block_len,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    benchmark_gather(
        context_lengths=[1, 128, 512, 2048, 8192],
        block_len=args.block_len,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
