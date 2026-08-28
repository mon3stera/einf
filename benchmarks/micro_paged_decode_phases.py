"""Explain the paged decode kernel's low effective bandwidth.

`micro-paged-decode-2026-08-23.md` measured 2.4% of peak bandwidth at context 544
and ~17.6% at 16384. That figure divides *unique* KV bytes by total time, which
conflates three different things. This script separates them:

1. Fixed cost versus streaming cost. The op launches two dependent kernels
   (partial, then reduce). Profiler self-device-time per kernel plus the
   event-timed total gives the split and the exposed inter-kernel gap.

2. Issued loads versus unique bytes. `grid.x` runs over the 14 Query heads while
   there are only 2 KV heads (`head_kv = q_head / 7` at line 91 of
   paged_attention_split_kv_cuda.cu), so seven CTAs each load the same KV data
   independently. The hardware moves 7x the unique bytes, which is the work the
   machine actually does and the honest denominator for a bandwidth claim.

3. Which memory level supplies those loads. A single request's KV working set is
   small enough to sit in the 4090's 72 MB L2, so the 7x amplified traffic is
   largely served by L2, not HBM. Comparing issued throughput against HBM peak
   is therefore the wrong comparison at long context.

Usage
    python benchmarks/micro_paged_decode_phases.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch.profiler import ProfilerActivity, profile

from einf.executors.torch.ops import (
    paged_decode_attention,
    paged_decode_attention_split_kv,
)


HBM_PEAK_GB_S = 1008.0
# Measured L2 read bandwidth on AD102 is roughly 2-3 TB/s; used only as a scale
# reference, never as a precise limit.
L2_REFERENCE_GB_S = 2500.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        default="128:8,544:8,544:16,1024:16,4096:64,16384:64",
        help="comma-separated context:num_splits pairs",
    )
    parser.add_argument("--num-q-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--profile-iters", type=int, default=60)
    parser.add_argument("--cache-bytes", type=float, default=512e6)
    parser.add_argument("--max-rotations", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = torch.bfloat16
    itemsize = 2
    configs = []
    for part in args.configs.split(","):
        if not part.strip():
            continue
        context_text, splits_text = part.split(":")
        configs.append((int(context_text), int(splits_text)))

    bytes_per_block = args.block_len * args.num_kv_heads * args.head_dim * itemsize
    num_blocks = max(
        max(math.ceil(context / args.block_len) for context, _ in configs),
        int(args.cache_bytes / (2 * bytes_per_block)),
    )
    K_cache = torch.randn(
        (num_blocks, args.block_len, args.num_kv_heads, args.head_dim),
        device="cuda",
        dtype=dtype,
    )
    V_cache = torch.randn_like(K_cache)
    Q = torch.randn((args.num_q_heads, args.head_dim), device="cuda", dtype=dtype)
    scale = 1.0 / math.sqrt(args.head_dim)
    groups = args.num_q_heads // args.num_kv_heads

    print(f"device: {torch.cuda.get_device_name()}")
    print(
        f"{args.num_q_heads} Q heads / {args.num_kv_heads} KV heads "
        f"(GQA group {groups}), head_dim {args.head_dim}, bf16"
    )
    print(f"KV cache {2 * K_cache.numel() * itemsize / 1e6:.0f} MB, {num_blocks} blocks")

    def tables_for(context: int) -> list[torch.Tensor]:
        num_logical = math.ceil(context / args.block_len)
        rotations = max(1, min(args.max_rotations, num_blocks // num_logical))
        return [
            torch.arange(
                rotation * num_logical,
                rotation * num_logical + num_logical,
                dtype=torch.long,
                device="cuda",
            )
            for rotation in range(rotations)
        ]

    def make_call(context: int, num_splits: int, tables: list[torch.Tensor]):
        def call(index: int) -> None:
            table = tables[index % len(tables)]
            if num_splits == 1:
                paged_decode_attention(Q, K_cache, V_cache, table, context, scale)
            else:
                paged_decode_attention_split_kv(
                    Q, K_cache, V_cache, table, context, num_splits, scale
                )

        return call

    rows = []
    for context, num_splits in configs:
        tables = tables_for(context)
        call = make_call(context, num_splits, tables)

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
        total_us = start.elapsed_time(end) * 1000.0 / args.iters

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for index in range(args.profile_iters):
                call(index)
            torch.cuda.synchronize()

        partial_us = reduce_us = 0.0
        for row in prof.key_averages():
            device_us = 0.0
            for attribute in ("self_device_time_total", "self_cuda_time_total"):
                value = getattr(row, attribute, 0.0) or 0.0
                if value:
                    device_us = float(value)
                    break
            if device_us <= 0:
                continue
            if "partial_kernel" in row.key:
                partial_us += device_us / args.profile_iters
            elif "reduce_kernel" in row.key or "paged_decode_attention_kernel" in row.key:
                reduce_us += device_us / args.profile_iters

        unique_bytes = (
            context * 2 * args.num_kv_heads * args.head_dim * itemsize
        )
        issued_bytes = context * 2 * args.num_q_heads * args.head_dim * itemsize
        rows.append(
            {
                "context": context,
                "splits": num_splits,
                "total_us": total_us,
                "host_us": host_us,
                "partial_us": partial_us,
                "reduce_us": reduce_us,
                "gap_us": total_us - partial_us - reduce_us,
                "unique_bytes": unique_bytes,
                "issued_bytes": issued_bytes,
            }
        )

    print()
    print("=== fixed cost versus streaming cost ===")
    print(
        f"{'ctx':>7}{'splits':>7}{'total us':>10}{'partial':>9}{'reduce':>8}"
        f"{'gap':>7}{'fixed %':>9}"
    )
    for row in rows:
        kernels = row["partial_us"] + row["reduce_us"]
        fixed_share = 100.0 * (row["reduce_us"] + row["gap_us"]) / row["total_us"]
        print(
            f"{row['context']:>7}{row['splits']:>7}{row['total_us']:>10.2f}"
            f"{row['partial_us']:>9.2f}{row['reduce_us']:>8.2f}"
            f"{row['gap_us']:>7.2f}{fixed_share:>8.1f}%"
        )
        del kernels

    print()
    print("=== unique bytes versus issued loads (GQA amplification) ===")
    print(
        f"{'ctx':>7}{'unique KB':>11}{'issued KB':>11}"
        f"{'unique GB/s':>13}{'issued GB/s':>13}{'%HBM':>7}{'%L2ref':>8}"
    )
    for row in rows:
        unique_gb_s = row["unique_bytes"] / (row["total_us"] * 1e-6) / 1e9
        issued_gb_s = row["issued_bytes"] / (row["total_us"] * 1e-6) / 1e9
        print(
            f"{row['context']:>7}{row['unique_bytes'] / 1024:>11.1f}"
            f"{row['issued_bytes'] / 1024:>11.1f}{unique_gb_s:>13.1f}"
            f"{issued_gb_s:>13.1f}{100.0 * issued_gb_s / HBM_PEAK_GB_S:>7.1f}"
            f"{100.0 * issued_gb_s / L2_REFERENCE_GB_S:>8.1f}"
        )

    print()
    print("=== per-token cost inside the partial kernel ===")
    print(
        f"{'ctx':>7}{'splits':>7}{'CTAs':>7}{'waves':>7}"
        f"{'tok/warp':>10}{'ns/token':>10}{'cyc/token':>11}"
    )
    for row in rows:
        context = row["context"]
        num_splits = row["splits"]
        num_logical = math.ceil(context / args.block_len)
        ctas = args.num_q_heads * num_splits
        tokens_per_cta = math.ceil(num_logical / num_splits) * args.block_len
        tokens_per_warp = tokens_per_cta / 4.0
        nanoseconds = row["partial_us"] * 1000.0 / max(tokens_per_warp, 1.0)
        print(
            f"{context:>7}{num_splits:>7}{ctas:>7}{ctas / 128.0:>7.2f}"
            f"{tokens_per_warp:>10.1f}{nanoseconds:>10.1f}{nanoseconds * 2.5:>11.0f}"
        )

    print(
        "\nns/token is partial-kernel time divided by tokens per warp, i.e. the "
        "latency of one iteration of the sequential per-token loop, not a "
        "throughput figure. Cycles assume a 2.5 GHz clock."
    )


if __name__ == "__main__":
    main()
