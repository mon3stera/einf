from __future__ import annotations

import argparse
import math

import torch

from einf.executors.torch.ops import flash_attention, load_custom_ops


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one einf FlashAttention shape for Nsight Compute."
    )
    parser.add_argument("--q-len", type=int, required=True)
    parser.add_argument("--kv-len", type=int, required=True)
    parser.add_argument("--num-attention-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.head_dim != 64:
        raise ValueError("the current native FlashAttention requires D=64")
    if args.num_attention_heads % args.num_kv_heads != 0:
        raise ValueError("Hq must be divisible by Hkv")
    if args.kv_len < args.q_len:
        raise ValueError("kv_len must be >= q_len")

    torch.manual_seed(0)
    load_custom_ops()
    dtype = torch.bfloat16
    device = torch.device("cuda")
    Q = torch.randn(
        args.q_len,
        args.num_attention_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    K = torch.randn(
        args.kv_len,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    V = torch.randn_like(K)
    scale = 1.0 / math.sqrt(args.head_dim)
    start_pos = args.kv_len - args.q_len

    with torch.inference_mode():
        for _ in range(args.warmup):
            flash_attention(Q, K, V, start_pos, scale)
        torch.cuda.synchronize()
        flash_attention(Q, K, V, start_pos, scale)
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
