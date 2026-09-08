"""Probe: paged packed-custom-mask batch prefill at tree-verify geometry.

Reproduces the verify shape of the failing step (qo=15, kv=28 tokens,
page_size=32 => one partial page, head_dim=64, GQA 14/2) and compares
FlashInfer's custom-mask attention against a manual per-head reference.
Variants isolate the trigger: mask content, GQA group size, page size.
"""

from __future__ import annotations

import sys

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
    kvh = k.shape[1]
    group = heads // kvh
    qg = q.view(qo, kvh, group, dim)  # head h -> kv head h // group
    scores = torch.einsum("qahd,kad->qahk", qg.float(), k.float()) * sm_scale
    scores = scores.masked_fill(~mask.view(qo, 1, 1, -1), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("qahk,kad->qahd", probs, v.float())

    return out.reshape(qo, heads, dim)


def build_mask(qo: int, kv: int) -> torch.Tensor:
    """Tree visibility: row r sees history block, pending, ancestors, self."""
    history = kv - 1 - (qo - 1)
    col_of: dict[int, int] = {}
    next_col = history + 1

    for node in range(1, qo):
        if next_col < kv:
            col_of[node] = next_col
            next_col += 1

    mask = torch.zeros(qo, kv, dtype=torch.bool)

    for node in range(qo):
        mask[node, : history + 1] = True

        if node > 0:
            if node in col_of:
                mask[node, col_of[node]] = True

            walk = (node - 1) // 2

            while walk in col_of:
                mask[node, col_of[walk]] = True
                walk = (walk - 1) // 2

    return mask


def run_case(
    label: str,
    mask: torch.Tensor | None,
    qo: int,
    kv: int,
    page: int,
    heads: int,
    kvh: int,
    dim: int = 64,
    causal: bool = False,
) -> None:
    torch.manual_seed(7)
    sm_scale = dim**-0.5
    q = torch.randn(qo, heads, dim, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.randn(1, 2, page, kvh, dim, dtype=torch.bfloat16, device=DEVICE)
    k, v = kv_cache[0, 0, :kv], kv_cache[0, 1, :kv]

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")

    plan_kwargs: dict[str, object] = {
        "causal": causal,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
    }

    if mask is not None:
        plan_kwargs["packed_custom_mask"] = pack_mask_flashinfer(mask).to(DEVICE)

    wrapper.plan(
        qo_indptr=torch.tensor([0, qo], dtype=torch.int32, device=DEVICE),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32, device=DEVICE),
        paged_kv_last_page_len=torch.tensor([kv], dtype=torch.int32, device=DEVICE),
        num_qo_heads=heads,
        num_kv_heads=kvh,
        head_dim_qk=dim,
        head_dim_vo=dim,
        page_size=page,
        **plan_kwargs,
    )
    result = wrapper.run(q, kv_cache).float()

    mask_d = mask.to(DEVICE) if mask is not None else torch.ones(
        qo, kv, dtype=torch.bool, device=DEVICE
    )
    reference = manual_attention(q, k, v, mask_d, sm_scale)
    diff = (result - reference).abs().amax()
    print(f"{label}: max abs diff {diff.item():.4f}", flush=True)


def main() -> None:
    qo, kv, page = 15, 28, 32
    tree = build_mask(qo, kv)
    ones = torch.ones(qo, kv, dtype=torch.bool)

    run_case("tree mask, GQA 14/2, page32", tree, qo, kv, page, 14, 2)
    run_case("all-ones mask, GQA 14/2, page32", ones, qo, kv, page, 14, 2)
    run_case(
        "no mask, causal, GQA 14/2, page32",
        None,
        qo,
        kv,
        page,
        14,
        2,
        causal=True,
    )
    run_case("tree mask, no GQA 2/2, page32", tree, qo, kv, page, 2, 2)
    run_case("all-ones mask, no GQA 2/2, page32", ones, qo, kv, page, 2, 2)
    run_case("tree mask, GQA 14/2, page1", tree, qo, kv, 1, 14, 2)
    run_case("tree mask, GQA 4/2, page32", tree, qo, kv, page, 4, 2)


if __name__ == "__main__":
    main()
