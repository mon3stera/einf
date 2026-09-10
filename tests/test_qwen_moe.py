"""Qwen3-MoE plumbing tests: config extension, sparse MLP parity against
the reference oracle, and QK-norm wiring.

The Triton grouped-GEMM path requires CUDA; the reference implementation
runs anywhere, so CPU cases exercise only the routing/config plumbing.
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.qwen import (
    QwenAttention,
    QwenConfig,
    QwenMoEMLP,
)


def _moe_config_dict(**overrides: object) -> dict:
    data: dict = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "rope_theta": 10000.0,
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "num_experts": 8,
        "num_experts_per_tok": 3,
        "moe_intermediate_size": 48,
        "norm_topk_prob": True,
        "architectures": ["Qwen3MoeForCausalLM"],
    }
    data.update(overrides)
    return data


def test_config_parses_moe_fields_and_infers_qk_norm() -> None:
    config = QwenConfig.from_dict(_moe_config_dict())

    assert config.num_experts == 8
    assert config.num_experts_per_tok == 3
    assert config.moe_intermediate_size == 48
    assert config.norm_topk_prob is True
    assert config.qk_norm is True  # inferred from Qwen3 architectures


def test_dense_config_unchanged() -> None:
    data = _moe_config_dict()
    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
        del data[key]
    data["architectures"] = ["Qwen2ForCausalLM"]

    config = QwenConfig.from_dict(data)

    assert config.num_experts is None
    assert config.qk_norm is False


def test_moe_explicit_qk_norm_override() -> None:
    config = QwenConfig.from_dict(_moe_config_dict(qk_norm=False))

    assert config.qk_norm is False


def _build_moe_mlp(
    device: torch.device, dtype: torch.dtype, *, norm_topk_prob: bool = True
) -> tuple[QwenMoEMLP, torch.Tensor, torch.Generator]:
    generator = torch.Generator(device=device).manual_seed(17)
    config = QwenConfig.from_dict(
        _moe_config_dict(norm_topk_prob=norm_topk_prob)
    )
    mlp = QwenMoEMLP(config).to(device=device, dtype=dtype)

    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape, generator=generator, device=device
                )
                * 0.1
            )

    hidden = torch.randn(
        (7, config.hidden_size), generator=generator, device=device, dtype=dtype
    )
    return mlp, hidden, generator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_moe_mlp_triton_matches_reference(norm_topk_prob: bool) -> None:
    device = torch.device("cuda")
    mlp, hidden, _ = _build_moe_mlp(
        device, torch.bfloat16, norm_topk_prob=norm_topk_prob
    )

    router_logits = mlp.gate(hidden).float()
    routing_weights, topk_ids = torch.topk(
        torch.softmax(router_logits, dim=-1), mlp.num_experts_per_tok, dim=-1
    )
    if norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )

    w13, w2 = mlp._packed_experts(hidden)
    expected = mlp.reference_forward(
        hidden, routing_weights, topk_ids
    )

    actual = mlp.forward(hidden)

    rel = (
        (actual.float() - expected.float()).norm()
        / expected.float().norm().clamp_min(1e-6)
    )
    assert rel.item() < 0.02, f"relative L2 {rel.item():.4e} exceeds 0.02"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_moe_mlp_reference_path_matches_reference_direct() -> None:
    """The use_reference_mlp oracle path must reproduce the direct
    reference call bit-for-bit — it is the parity anchor for kernel work."""
    device = torch.device("cuda")
    mlp, hidden, _ = _build_moe_mlp(device, torch.bfloat16)

    router_logits = mlp.gate(hidden).float()
    routing_weights, topk_ids = torch.topk(
        torch.softmax(router_logits, dim=-1), mlp.num_experts_per_tok, dim=-1
    )
    routing_weights = routing_weights / routing_weights.sum(
        dim=-1, keepdim=True
    )

    w13, w2 = mlp._packed_experts(hidden)
    direct = mlp.reference_forward(hidden, routing_weights, topk_ids)

    mlp.use_reference_mlp = True
    through_model = mlp.forward(hidden)
    mlp.use_reference_mlp = False

    assert torch.equal(direct, through_model)


def test_qk_norm_modules_created_only_when_configured() -> None:
    attention = QwenAttention(QwenConfig.from_dict(_moe_config_dict()), 0)
    assert attention.q_norm is not None
    assert attention.k_norm is not None
    assert attention.q_norm.weight.shape == (16,)  # head_dim = 64/4

    dense = QwenConfig.from_dict(
        _moe_config_dict(qk_norm=False, architectures=["Qwen2ForCausalLM"])
    )
    dense_attention = QwenAttention(dense, 0)
    assert dense_attention.q_norm is None
    assert dense_attention.k_norm is None
