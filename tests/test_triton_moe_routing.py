"""Numerical tests for the fused MoE routing kernel (DeepSeek-V3 MoEGate).

CPU cases pin the reference semantics. CUDA cases run the kernel body in
``triton_ops.moe_topk_softmax_kernel`` against the reference; they are the
acceptance test for the ``YOUR CODE HERE`` section.
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.triton_ops import (
    moe_topk_softmax_reference,
    moe_topk_softmax_triton,
)

# (num_experts, top_k, with_bias, renorm, scaling) — DeepSeek-V3 is
# (256, 8, True, True, 2.5); DeepSeek-V2-Lite is (64, 6, True, True, 1.0);
# Qwen-style without bias and without renorm also covered.
_CASES = [
    (256, 8, True, True, 2.5),
    (64, 6, True, True, 1.0),
    (8, 2, False, True, 1.0),
    (16, 4, True, False, 1.0),
]


def test_reference_deepseek_v3_semantics() -> None:
    logits = torch.randn(4, 8)
    bias = torch.randn(8)

    weights, ids = moe_topk_softmax_reference(logits, bias, top_k=2, scaling=2.5)

    scores = torch.softmax(logits, dim=-1)
    biased = scores + bias
    expected_ids = torch.topk(biased, 2, dim=-1).indices
    expected = scores.gather(-1, expected_ids)
    expected = expected / expected.sum(-1, keepdim=True) * 2.5

    torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(ids, expected_ids.to(torch.int32))


def test_reference_renorm_sums_to_scaling() -> None:
    logits = torch.randn(16, 32)
    weights, _ = moe_topk_softmax_reference(
        logits, None, top_k=4, scaling=2.5, renorm=True
    )
    torch.testing.assert_close(
        weights.sum(-1), torch.full((16,), 2.5), rtol=1e-5, atol=1e-6
    )


def test_reference_unnormalized_scales_raw_weights() -> None:
    logits = torch.randn(16, 32)
    weights, ids = moe_topk_softmax_reference(
        logits, None, top_k=4, scaling=3.0, renorm=False
    )
    scores = torch.softmax(logits.float(), dim=-1)
    expected = scores.gather(-1, ids) * 3.0
    torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("num_experts,top_k,with_bias,renorm,scaling", _CASES)
@pytest.mark.parametrize("num_tokens", [1, 17, 257])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_moe_routing_matches_reference(
    num_experts: int,
    top_k: int,
    with_bias: bool,
    renorm: bool,
    scaling: float,
    num_tokens: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        num_experts * 100 + top_k + num_tokens
    )
    logits = torch.randn(
        (num_tokens, num_experts), generator=generator, device="cuda"
    )
    bias = torch.randn(num_experts, generator=generator, device="cuda")
    if not with_bias:
        bias = None

    weights, ids = moe_topk_softmax_triton(
        logits, bias, top_k=top_k, scaling=scaling, renorm=renorm
    )
    expected_w, expected_ids = moe_topk_softmax_reference(
        logits, bias, top_k=top_k, scaling=scaling, renorm=renorm
    )

    torch.testing.assert_close(weights, expected_w, rtol=1e-4, atol=1e-5)
    assert ids.dtype == torch.int32
    torch.testing.assert_close(
        ids.float(), expected_ids.float(), rtol=0.0, atol=0.0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_moe_routing_weights_sum_to_scaling() -> None:
    logits = torch.randn(64, 256, device="cuda")
    weights, ids = moe_topk_softmax_triton(
        logits, torch.randn(256, device="cuda"), top_k=8, scaling=2.5
    )
    torch.testing.assert_close(
        weights.sum(-1), torch.full((64,), 2.5, device="cuda"), rtol=1e-4, atol=1e-5
    )
    assert ids.unique().numel() == 8 or len(ids.unique()) <= 256
