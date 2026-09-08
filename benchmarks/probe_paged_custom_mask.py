"""Probe: paged packed-custom-mask batch prefill at tree-verify geometry.

Reproduces the verify shape of the failing step (qo=15, kv=28 tokens,
page_size=32 => one partial page, head_dim=64, GQA 14/2) and compares
FlashInfer's custom-mask attention against a manual per-head reference.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import flashinfer

sys.path.insert(0, "src")

from einf.executors.torch.spec_tree import pack_mask_flashinfer

DEVICE = torch.device("cuda")


def manual_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    """q: [qo, H, D], k/v: [kv, KVH, D], mask: [qo, kv] bool -> [qo, H, D]."""
    qo, heads, dim = q.shape
    kvh = k.shape[2]
    group = heads // kvh
    qg = q.view(qo, group, kvh, dim)  # head h -> kv head h // group
    scores = torch.einsum(
        "qghd,khd->qghk", qg.float(), k.float()
    ) * sm_scale  # [qo, group, kvh, kv]
    scores = scores.masked_fill(~mask.view(qo, 1, 1, -1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("qghk,khd->qghd", probs, v.float())
    return out.reshape(qo, heads, dim)


def main() -> None:
    torch.manual_seed(7)

    qo, kv, page = 15, 28, 32
    heads, kvh, dim = 14, 2, 64
    sm_scale = dim**-0.5

    # tree-shaped visibility: rows see history block (14), pending (col 14),
    # then ancestors among columns 15..27; row 0 sees history+pending only.
    depth_of = {0: 0}
    col_of = {}
    next_col = 15
    for node in range(1, qo):
        parent = (node - 1) // 2
        depth_of[node] = depth_of[parent] + 1
        if next_col < kv:
            col_of[node] = next_col
            next_col += 1

    mask = torch.zeros(qo, kv, dtype=torch.bool)
    for node in range(qo):
        mask[node, : 14 + 1] = True  # history + pending

        if node > 0:
            if node in col_of:
                mask[node, col_of[node]] = True

            walk = (node - 1) // 2

            while walk in col_of:
                mask[node, col_of[walk]] = True
                walk = (walk - 1) // 2

    packed = pack_mask_flashinfer(mask)

    q = torch.randn(qo, heads, dim, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.randn(
        1, 2, page, kvh, dim, dtype=torch.bfloat16, device=DEVICE
    )

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    out = wrapper.plan(
        qo_indptr=torch.tensor([0, qo], dtype=torch.int32, device=DEVICE),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        paged_kv_last_page_len=torch.tensor([kv], dtype=torch.int32, device=DEVICE),
        num_qo_heads=heads,
        num_kv_heads=kvh,
        head_dim_qk=dim,
        head_dim_vo=dim,
        page_size=page,
        causal=False,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        packed_custom_mask=packed,
    )
    result = wrapper.run(q, kv_cache).float()

    k, v = kv_cache[:, 0], kv_cache[:, 1]
    reference = manual_attention(q, k, v, mask, sm_scale)
    diff = (result - reference).abs()
    per_row = diff.amax(dim=(1, 2))

    print(f"per-row max abs diff vs manual reference (bf16 inputs):")
    for r in range(qo):
        mark = "ok" if per_row[r] < 5e-2 else "MISMATCH"
        print(f"  row {r:2d}: {per_row[r]:.4f} {mark}")

    print(f"worst {per_row.max().item():.4f}")


if __name__ == "__main__":
    main()
