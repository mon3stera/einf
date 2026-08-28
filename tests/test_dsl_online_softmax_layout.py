"""第 2.5 级 layout-derived online softmax 的回归测试。

练习期间正确性测试使用 ``xfail(raises=NotImplementedError, strict=False)``，用于区分
「尚未实现」和「实现错误」。kernel 已实现并通过，xfail 标记现已摘除；这些测试都是
真实断言，后续回归不能退回 XFAIL。
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


def _reference(Q: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference for the row max and normalized exponential sum."""
    scores = Q.float() @ K.float().transpose(0, 1)
    m = scores.max(dim=-1).values
    l = torch.exp(scores - m[:, None]).sum(dim=-1)
    return m, l


@pytest.mark.parametrize("kv_len", [N_TILE, 2 * N_TILE, 4 * N_TILE, 8 * N_TILE])
@pytest.mark.parametrize("seed", [0, 71])
def test_cute_online_softmax_layout_matches_reference(
    kv_len: int,
    seed: int,
) -> None:
    from einf.executors.torch.dsl import cute_online_softmax_layout

    torch.manual_seed(seed)
    Q = torch.randn((M_QUERIES, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((kv_len, HEAD_DIM), device="cuda", dtype=torch.bfloat16)

    m_ref, l_ref = _reference(Q, K)
    m, l = cute_online_softmax_layout(Q, K)

    assert m.dtype is torch.float32
    assert l.dtype is torch.float32
    assert m.shape == (M_QUERIES,)
    assert l.shape == (M_QUERIES,)
    torch.testing.assert_close(m, m_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(l, l_ref, rtol=1e-5, atol=1e-5)


def test_cute_online_softmax_layout_reconstructs_softmax() -> None:
    from einf.executors.torch.dsl import cute_online_softmax_layout

    torch.manual_seed(1234)
    Q = torch.randn((M_QUERIES, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((4 * N_TILE, HEAD_DIM), device="cuda", dtype=torch.bfloat16)

    m, l = cute_online_softmax_layout(Q, K)
    scores = Q.float() @ K.float().transpose(0, 1)
    actual = torch.exp(scores - m[:, None]) / l[:, None]
    expected = torch.softmax(scores, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cute_online_softmax_layout_is_shift_stable() -> None:
    from einf.executors.torch.dsl import cute_online_softmax_layout

    Q = torch.full((M_QUERIES, HEAD_DIM), 40.0, device="cuda", dtype=torch.bfloat16)
    K = torch.full((2 * N_TILE, HEAD_DIM), 40.0, device="cuda", dtype=torch.bfloat16)

    m, l = cute_online_softmax_layout(Q, K)

    assert torch.isfinite(m).all(), "m contains inf/NaN"
    assert torch.isfinite(l).all(), "l contains inf/NaN"
    torch.testing.assert_close(
        l,
        torch.full_like(l, 2 * N_TILE),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.parametrize(
    ("shape_q", "shape_k", "dtype", "match"),
    [
        ((16, HEAD_DIM), (2 * N_TILE, HEAD_DIM), torch.bfloat16, "Q must have shape"),
        ((M_QUERIES, HEAD_DIM), (2 * N_TILE, 8), torch.bfloat16, "K must have shape"),
        (
            (M_QUERIES, HEAD_DIM),
            (N_TILE + N_TILE // 2, HEAD_DIM),
            torch.bfloat16,
            "multiple of 16",
        ),
        ((M_QUERIES, HEAD_DIM), (0, HEAD_DIM), torch.bfloat16, "multiple of 16"),
        ((M_QUERIES, HEAD_DIM), (2 * N_TILE, HEAD_DIM), torch.float32, "requires BF16"),
    ],
)
def test_cute_online_softmax_layout_rejects_unsupported_inputs(
    shape_q: tuple[int, int],
    shape_k: tuple[int, int],
    dtype: torch.dtype,
    match: str,
) -> None:
    from einf.executors.torch.dsl import cute_online_softmax_layout

    Q = torch.zeros(shape_q, device="cuda", dtype=dtype)
    K = torch.zeros(shape_k, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=match):
        cute_online_softmax_layout(Q, K)
