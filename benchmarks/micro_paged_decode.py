"""Isolated micro-benchmark for the paged decode attention operators.

Answers two questions that the end-to-end step decomposition cannot separate:

1. How efficient is the single-request Split-KV kernel *on its own*, with no
   Python loop, no `.item()` synchronisations and no scheduler around it?
2. Does a longer context supply enough parallelism by itself, i.e. does the
   `num_splits` grid dimension eventually saturate the device?

Method notes that matter for believing the numbers:

* The KV cache is deliberately sized well past the RTX 4090's 72 MB L2 and each
  timed iteration reads a *different* region through a different block table.
  Timing one small context in place would keep the whole working set resident in
  L2 and report a bandwidth figure that the real decode loop never sees.
* Bytes counted are the unavoidable KV traffic:
  `context_len * 2 (K and V) * num_kv_heads * head_dim * itemsize`.
  Split-KV additionally writes and re-reads a partial state of
  `num_q_heads * num_splits * (head_dim * 4 + 8)` bytes, reported separately,
  because that is the cost the split pays for its parallelism.
* `num_splits == 1` is served by the non-split `paged_decode_attention` op,
  exactly as the production path does.

Usage
    python benchmarks/micro_paged_decode.py
    python benchmarks/micro_paged_decode.py --contexts 544,4096 --iters 400
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from einf.executors.torch.ops import (
    paged_decode_attention,
    paged_decode_attention_split_kv,
)
from einf.executors.torch.qwen import choose_num_splits


# RTX 4090: 384-bit GDDR6X at 21 Gbps.
PEAK_GB_S = 1008.0
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="544,1024,2048,4096,8192,16384")
    parser.add_argument("--num-q-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--cache-bytes",
        type=float,
        default=512e6,
        help="total KV cache size; must exceed L2 to avoid a cached-in-place lie",
    )
    parser.add_argument("--max-rotations", type=int, default=64)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="batch size used for the per-request-loop projection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = DTYPES[args.dtype]
    itemsize = torch.empty((), dtype=dtype).element_size()
    contexts = [int(part) for part in args.contexts.split(",") if part.strip()]

    bytes_per_block = args.block_len * args.num_kv_heads * args.head_dim * itemsize
    num_blocks = max(
        max(math.ceil(context / args.block_len) for context in contexts),
        int(args.cache_bytes / (2 * bytes_per_block)),
    )

    K_cache = torch.randn(
        (num_blocks, args.block_len, args.num_kv_heads, args.head_dim),
        device="cuda",
        dtype=dtype,
    )
    V_cache = torch.randn_like(K_cache)
    Q = torch.randn(
        (args.num_q_heads, args.head_dim), device="cuda", dtype=dtype
    )
    scale = 1.0 / math.sqrt(args.head_dim)

    total_cache_mb = 2 * K_cache.numel() * itemsize / 1e6
    print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"heads: {args.num_q_heads} Q / {args.num_kv_heads} KV, "
        f"head_dim {args.head_dim}, block_len {args.block_len}, dtype {dtype}"
    )
    print(
        f"KV cache: {num_blocks} blocks, {total_cache_mb:.0f} MB total "
        f"(rotated across up to {args.max_rotations} disjoint regions per config)"
    )
    print(f"iters {args.iters}, warmup {args.warmup}, peak {PEAK_GB_S:.0f} GB/s")

    def time_config(
        context: int, num_splits: int, tables: list[torch.Tensor]
    ) -> tuple[float, float]:
        """Mean device and host microseconds per call, cycling tables to defeat L2.

        The host figure is the wall time of the enqueue loop alone. If it is close
        to the device figure, the call path -- Python wrapper, dispatch, the four
        tensor allocations -- is the limiter rather than the kernel.
        """

        def call(index: int) -> None:
            table = tables[index % len(tables)]
            if num_splits == 1:
                paged_decode_attention(Q, K_cache, V_cache, table, context, scale)
            else:
                paged_decode_attention_split_kv(
                    Q, K_cache, V_cache, table, context, num_splits, scale
                )

        for index in range(args.warmup):
            call(index)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        for index in range(args.iters):
            call(index)
        end.record()
        host_us = (time.perf_counter() - host_start) * 1e6 / args.iters
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / args.iters, host_us

    def build_tables(num_logical: int, *, shuffled: bool = False) -> list[torch.Tensor]:
        rotations = max(1, min(args.max_rotations, num_blocks // num_logical))
        tables = []
        for rotation in range(rotations):
            base = rotation * num_logical
            ids = torch.arange(base, base + num_logical, dtype=torch.long)
            if shuffled:
                ids = ids[torch.randperm(num_logical)]
            tables.append(ids.cuda())
        return tables

    print()
    print("=== single request, Split-KV parallelism sweep ===")
    print(
        f"{'context':>8}{'splits':>8}{'blocks':>8}{'waves':>7}"
        f"{'dev us':>9}{'host us':>9}{'KV GB/s':>10}{'%peak':>7}"
        f"{'partial KB':>12}  note"
    )
    best: dict[int, tuple[int, float]] = {}
    production: dict[int, tuple[int, float]] = {}
    for context in contexts:
        num_logical = math.ceil(context / args.block_len)
        prod_splits = choose_num_splits(num_logical, 64)
        candidates = sorted(
            {1, 2, 4, 8, 16, 32, 64, 128, 256, num_logical, prod_splits}
            & set(range(1, num_logical + 1))
        )
        tables = build_tables(num_logical)
        kv_bytes = context * 2 * args.num_kv_heads * args.head_dim * itemsize
        for num_splits in candidates:
            microseconds, host_us = time_config(context, num_splits, tables)
            gb_s = kv_bytes / (microseconds * 1e-6) / 1e9
            blocks = args.num_q_heads * num_splits
            partial_kb = 0.0
            if num_splits > 1:
                partial_kb = (
                    args.num_q_heads
                    * num_splits
                    * (args.head_dim * 4 + 8)
                    * 2
                    / 1024.0
                )
            notes = []
            if num_splits == prod_splits:
                notes.append("<- production heuristic")
            if num_splits == num_logical:
                notes.append("max legal")
            print(
                f"{context:>8}{num_splits:>8}{blocks:>8}{blocks / 128.0:>7.2f}"
                f"{microseconds:>9.2f}{host_us:>9.2f}{gb_s:>10.1f}"
                f"{100.0 * gb_s / PEAK_GB_S:>7.1f}{partial_kb:>12.1f}  "
                + " ".join(notes)
            )
            if context not in best or microseconds < best[context][1]:
                best[context] = (num_splits, microseconds)
            if num_splits == prod_splits:
                production[context] = (num_splits, microseconds)
        shuffled_us, shuffled_host = time_config(
            context, best[context][0], build_tables(num_logical, shuffled=True)
        )
        print(
            f"{context:>8}{best[context][0]:>8}{'':>8}{'':>7}{shuffled_us:>9.2f}"
            f"{shuffled_host:>9.2f}{kv_bytes / (shuffled_us * 1e-6) / 1e9:>10.1f}"
            f"{'':>7}{'':>12}  shuffled block table, best splits"
        )

    print()
    print("=== what this means per decode step (24 layers) ===")
    print(
        f"{'context':>8}{'prod us':>9}{'best us':>9}{'gain':>7}"
        f"{'loop ms/step':>14}{'floor ms/step':>15}{'ratio':>8}"
    )
    for context in contexts:
        prod_splits, prod_us = production[context]
        best_splits, best_us = best[context]
        kv_bytes = context * 2 * args.num_kv_heads * args.head_dim * itemsize
        loop_ms = prod_us * args.batch * 24 / 1000.0
        floor_ms = (
            kv_bytes * args.batch * 24 / (PEAK_GB_S * 1e9) * 1000.0
        )
        print(
            f"{context:>8}{prod_us:>9.2f}{best_us:>9.2f}"
            f"{prod_us / best_us:>6.2f}x{loop_ms:>14.2f}{floor_ms:>15.3f}"
            f"{loop_ms / floor_ms:>7.1f}x"
        )
    print(
        f"\nloop ms/step = measured single-request us x batch {args.batch} x 24 layers, "
        "i.e. what the current per-request path costs in device time alone.\n"
        "floor ms/step = the same KV traffic at peak bandwidth."
    )


if __name__ == "__main__":
    main()
