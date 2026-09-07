from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from einf.executors.torch.flashinfer_attn import flashinfer_available
from einf.executors.torch.fused_ops import (
    apply_rope_inplace,
    fused_add_rmsnorm,
    rmsnorm,
    silu_and_mul,
)
from einf.executors.torch.model_runner import apply_rotary_pos_emb


def _pytorch_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = input * torch.rsqrt(variance + eps)
    return (weight * normalized).to(dtype=input.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_flashinfer_rmsnorm_matches_pytorch() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 896, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(896, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        rmsnorm(x, weight, 1e-6).float(),
        _pytorch_rmsnorm(x, weight, 1e-6).float(),
        rtol=1e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_flashinfer_fused_add_rmsnorm_matches_pytorch() -> None:
    torch.manual_seed(1)
    residual = torch.randn(8, 896, device="cuda", dtype=torch.bfloat16)
    incoming = torch.randn(8, 896, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(896, device="cuda", dtype=torch.bfloat16)
    expected_residual = residual + incoming
    expected = _pytorch_rmsnorm(expected_residual, weight, 1e-6)
    fused_add_rmsnorm(incoming, residual, weight, 1e-6)
    torch.testing.assert_close(residual.float(), expected_residual.float(), rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(incoming.float(), expected.float(), rtol=1e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_flashinfer_silu_and_mul_matches_pytorch() -> None:
    torch.manual_seed(2)
    gate_up = torch.randn(8, 9728, device="cuda", dtype=torch.bfloat16)
    hidden = gate_up.shape[-1] // 2
    expected = F.silu(gate_up[..., :hidden]) * gate_up[..., hidden:]
    torch.testing.assert_close(
        silu_and_mul(gate_up).float(),
        expected.float(),
        rtol=1e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_flashinfer_rope_matches_rotate_half() -> None:
    torch.manual_seed(3)
    q = torch.randn(8, 14, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 2, 64, device="cuda", dtype=torch.bfloat16)
    position = torch.arange(8, device="cuda")
    inv_freq = 1.0 / (1_000_000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    freqs = position[:, None].float() * inv_freq[None, :]
    embedding = torch.cat((freqs, freqs), dim=-1)
    cos = embedding.cos().unsqueeze(1).to(dtype=q.dtype)
    sin = embedding.sin().unsqueeze(1).to(dtype=q.dtype)
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos, sin)
    q_out = q.clone()
    k_out = k.clone()
    apply_rope_inplace(q_out, k_out, position, rope_theta=1_000_000.0)
    torch.testing.assert_close(q_out.float(), expected_q.float(), rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(k_out.float(), expected_k.float(), rtol=2e-2, atol=5e-2)


def test_cpu_fused_add_rmsnorm_fallback() -> None:
    residual = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    incoming = torch.tensor([[0.5, -0.5, 0.25, -0.25]])
    weight = torch.ones(4)
    out, new_residual = fused_add_rmsnorm(incoming, residual, weight, 1e-6)
    expected_residual = residual + incoming
    torch.testing.assert_close(new_residual, expected_residual)
    torch.testing.assert_close(out, _pytorch_rmsnorm(expected_residual, weight, 1e-6))
