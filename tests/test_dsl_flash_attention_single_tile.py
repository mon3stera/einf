"""第 3.0 级单 K/V tile P @ V 学习脚手架测试。

正确性测试使用 ``xfail(raises=NotImplementedError, strict=False)``：

    未实现      -> XFAIL
    实现且正确  -> XPASS
    实现但算错  -> FAIL（AssertionError 不匹配 NotImplementedError）

实现通过后必须摘除 xfail 标记，让这些测试成为长期回归保护。
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.dsl import dsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not dsl_available(),
    reason="CuTe DSL (nvidia-cutlass-dsl) and a CUDA device are required",
)

M_QUERIES = 32
N_TILE = 16
HEAD_DIM = 16


_pv_scaffold = pytest.mark.xfail(
    reason=(
        "第 3.0 级练习：单 tile S -> P -> P @ V kernel 尚未实现"
        "（实现正确后自动 XPASS）"
    ),
    raises=NotImplementedError,
    strict=False,
)


def _reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Match the intended BF16-P Tensor Core data path exactly."""
    scores = Q.float() @ K.float().transpose(0, 1)
    row_max = scores.max(dim=-1).values
    exp_scores = torch.exp(scores - row_max[:, None])
    numerator = exp_scores.to(torch.bfloat16).float() @ V.float()
    return numerator / exp_scores.sum(dim=-1)[:, None]


@pytest.mark.parametrize("seed", [0, 71, 1234])
def test_cute_flash_attention_single_tile_matches_reference(seed: int) -> None:
    from einf.executors.torch.dsl import cute_flash_attention_single_tile

    torch.manual_seed(seed)
    Q = torch.randn((M_QUERIES, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((N_TILE, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.randn((N_TILE, HEAD_DIM), device="cuda", dtype=torch.bfloat16)

    actual = cute_flash_attention_single_tile(Q, K, V)
    expected = _reference(Q, K, V)

    assert actual.dtype is torch.float32
    assert actual.shape == (M_QUERIES, HEAD_DIM)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


def test_cute_flash_attention_single_tile_uniform_scores() -> None:
    """Uniform scores make every P entry exactly 1/16 before P @ V."""
    from einf.executors.torch.dsl import cute_flash_attention_single_tile

    Q = torch.zeros((M_QUERIES, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((N_TILE, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.arange(
        N_TILE * HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    ).reshape(N_TILE, HEAD_DIM).to(torch.bfloat16)

    actual = cute_flash_attention_single_tile(Q, K, V)
    expected_row = V.float().mean(dim=0)
    expected = expected_row.expand(M_QUERIES, HEAD_DIM)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


def test_cute_flash_attention_single_tile_is_score_shift_stable() -> None:
    """Large equal scores must not overflow the S -> exp(S-m) path."""
    from einf.executors.torch.dsl import cute_flash_attention_single_tile

    Q = torch.full((M_QUERIES, HEAD_DIM), 40.0, device="cuda", dtype=torch.bfloat16)
    K = torch.full((N_TILE, HEAD_DIM), 40.0, device="cuda", dtype=torch.bfloat16)
    V = torch.randn((N_TILE, HEAD_DIM), device="cuda", dtype=torch.bfloat16)

    actual = cute_flash_attention_single_tile(Q, K, V)
    assert torch.isfinite(actual).all(), "O contains inf/NaN"
    expected = V.float().mean(dim=0).expand(M_QUERIES, HEAD_DIM)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("shape_q", "shape_k", "shape_v", "dtype", "match"),
    [
        (
            (16, HEAD_DIM),
            (N_TILE, HEAD_DIM),
            (N_TILE, HEAD_DIM),
            torch.bfloat16,
            "Q must have shape",
        ),
        (
            (M_QUERIES, HEAD_DIM),
            (8, HEAD_DIM),
            (N_TILE, HEAD_DIM),
            torch.bfloat16,
            "K must have shape",
        ),
        (
            (M_QUERIES, HEAD_DIM),
            (N_TILE, HEAD_DIM),
            (8, HEAD_DIM),
            torch.bfloat16,
            "V must have shape",
        ),
        (
            (M_QUERIES, HEAD_DIM),
            (N_TILE, HEAD_DIM),
            (N_TILE, HEAD_DIM),
            torch.float32,
            "requires BF16",
        ),
    ],
)
def test_cute_flash_attention_single_tile_rejects_unsupported_inputs(
    shape_q: tuple[int, int],
    shape_k: tuple[int, int],
    shape_v: tuple[int, int],
    dtype: torch.dtype,
    match: str,
) -> None:
    from einf.executors.torch.dsl import cute_flash_attention_single_tile

    Q = torch.zeros(shape_q, device="cuda", dtype=dtype)
    K = torch.zeros(shape_k, device="cuda", dtype=dtype)
    V = torch.zeros(shape_v, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=match):
        cute_flash_attention_single_tile(Q, K, V)


def test_cute_flash_attention_single_tile_rejects_device_mismatch() -> None:
    """The host validation runs before the intentional scaffold gate."""
    from einf.executors.torch.dsl import cute_flash_attention_single_tile

    Q = torch.zeros((M_QUERIES, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((N_TILE, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.zeros((N_TILE, HEAD_DIM), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must be CUDA tensors"):
        cute_flash_attention_single_tile(Q, K, V)
