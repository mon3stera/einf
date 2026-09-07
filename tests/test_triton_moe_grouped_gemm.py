"""Tests for the fused MoE expert path (align + grouped GEMM + combine).

CPU cases pin the alignment layout and the reference math. CUDA cases run
the Triton grouped-GEMM pipeline against the eager reference; they are the
acceptance test for the ``YOUR CODE HERE`` body in
``triton_moe.moe_grouped_gemm_kernel``.
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.triton_moe import (
    moe_align_block_size,
    moe_expert_mlp_reference,
    moe_expert_mlp_triton,
)


def test_align_groups_pairs_and_pads() -> None:
    topk_ids = torch.tensor([[0, 2], [2, 0], [1, 2]], dtype=torch.int32)
    block_m = 4

    sorted_ids, expert_ids, post_padded, total_padded = moe_align_block_size(
        topk_ids, block_m, num_experts=4
    )

    flat = topk_ids.reshape(-1)
    counts = torch.bincount(flat, minlength=4)

    assert total_padded == sum(
        ((c + block_m - 1) // block_m) * block_m for c in counts.tolist()
    )
    assert post_padded.item() == total_padded
    assert sorted_ids.numel() == total_padded
    assert expert_ids.numel() == total_padded // block_m

    valid = sorted_ids[sorted_ids >= 0].long()
    assert valid.numel() == flat.numel()
    # Grouped by expert and stable within expert.
    assert torch.equal(flat[valid], torch.sort(flat[valid])[0])
    for expert, count in enumerate(counts.tolist()):
        if count:
            assert (flat[valid] == expert).sum() == count

    # Every expert's run occupies a whole number of blocks; sentinels fill
    # the tails.
    sentinel_count = (sorted_ids == -1).sum()
    assert sentinel_count == total_padded - flat.numel()


def test_align_single_expert_run() -> None:
    topk_ids = torch.full((5, 2), 3, dtype=torch.int32)

    sorted_ids, expert_ids, post_padded, total_padded = moe_align_block_size(
        topk_ids, block_m=8, num_experts=8
    )

    assert total_padded == 16
    assert torch.equal(expert_ids, torch.tensor([3, 3], dtype=torch.int32))
    assert (sorted_ids[:10] >= 0).all()
    assert (sorted_ids[10:] == -1).all()


def _assert_rows_close(actual: torch.Tensor, expected: torch.Tensor, *, max_row_rel: float = 0.02) -> None:
    """Per-row relative-L2 comparison.

    bf16 pipelines carry ~0.5% quantization noise per element (intermediate
    storage + divergent fp32 summation order), which breaks per-element
    absolute tolerances on small-magnitude elements while the row as a whole
    matches far tighter. Row-relative L2 is the standard tolerance for this.
    """
    rel = (
        (actual.float() - expected.float()).norm(dim=-1)
        / expected.float().norm(dim=-1).clamp_min(1e-6)
    )
    assert rel.max() < max_row_rel, f"row rel-L2 {rel.max():.4e} exceeds {max_row_rel}"


def _make_case(
    num_tokens: int,
    k_pairs: int,
    num_experts: int,
    h_dim: int,
    inter: int,
    dtype: torch.dtype,
    seed: int,
    device: torch.device,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    hidden = torch.randn(
        (num_tokens, h_dim), generator=generator, device=device, dtype=dtype
    )
    logits = torch.randn(
        (num_tokens, num_experts), generator=generator, device=device
    )
    topk_weights, topk_ids = torch.topk(logits, k_pairs, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)

    w13 = torch.randn(
        (num_experts, 2 * inter, h_dim),
        generator=generator,
        device=device,
        dtype=dtype,
    ) * 0.05
    w2 = torch.randn(
        (num_experts, h_dim, inter),
        generator=generator,
        device=device,
        dtype=dtype,
    ) * 0.05

    return hidden, topk_weights, topk_ids, w13, w2


# CUDA cases run in bf16 — the serving dtype — against the
# quantize_intermediates reference. fp32 cases are omitted: tl.dot defaults
# to tf32 for fp32 inputs, which would demand ieee-precision comparisons.
@pytest.mark.parametrize(
    "num_tokens,k_pairs,num_experts,h_dim,inter,dtype",
    [
        (1, 8, 8, 64, 32, torch.bfloat16),
        (13, 3, 5, 64, 32, torch.bfloat16),  # uneven counts -> sentinels
        (64, 2, 4, 32, 16, torch.bfloat16),
        (64, 8, 8, 512, 256, torch.bfloat16),  # multi-block experts, no sentinels
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_moe_expert_mlp_matches_reference(
    num_tokens: int,
    k_pairs: int,
    num_experts: int,
    h_dim: int,
    inter: int,
    dtype: torch.dtype,
) -> None:
    hidden, topk_weights, topk_ids, w13, w2 = _make_case(
        num_tokens,
        k_pairs,
        num_experts,
        h_dim,
        inter,
        dtype,
        seed=num_tokens * 7 + k_pairs,
        device=torch.device("cuda"),
    )

    actual = moe_expert_mlp_triton(hidden, topk_weights, topk_ids, w13, w2)
    expected = moe_expert_mlp_reference(
        hidden, topk_weights, topk_ids, w13, w2, quantize_intermediates=True
    )

    _assert_rows_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_moe_expert_mlp_matches_reference_all_experts_hit() -> None:
    # Dense routing: every expert receives tokens, maximising block coverage.
    hidden, topk_weights, topk_ids, w13, w2 = _make_case(
        16, 8, 8, 64, 32, torch.float32, seed=99, device=torch.device("cuda")
    )

    actual = moe_expert_mlp_triton(hidden, topk_weights, topk_ids, w13, w2)
    expected = moe_expert_mlp_reference(
        hidden, topk_weights, topk_ids, w13, w2, quantize_intermediates=True
    )

    _assert_rows_close(actual, expected)
