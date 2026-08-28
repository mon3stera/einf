"""第 2 级 cute_online_softmax（CuTe DSL）的测试。

设计说明：这些测试**不断言脚手架未实现**。第 1 级犯过那个错 —— 测试断言
``pytest.raises(RuntimeError, match="learning scaffold")``，于是 kernel 写完之后
测试依然是绿的，验证的却是「还没实现」，属于最危险的假阳性。

练习期间用的是 ``xfail(raises=NotImplementedError, strict=False)``，三种状态各得其所：

    未实现        -> 抛 NotImplementedError，匹配 raises，记为 XFAIL（不算失败）
    实现且正确    -> 通过，记为 XPASS（不算失败，-ra 摘要里能看到）
    实现但算错    -> 抛 AssertionError，**不**匹配 raises，记为真正的 FAIL

即 ``raises=`` 是关键：它让「写错」和「没写」区分开。

**kernel 已实现，标记已摘除。** 这一步是必须的：``strict=False`` 的 xfail 遇到 XPASS
不算失败，所以标记留着的话，日后实现被改坏会退回 XFAIL 而套件**依然是绿的** ——
练习期间它是保护，练习做完它就变成漏网。现在这些测试是真正的断言。
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.dsl import dsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not dsl_available(),
    reason="CuTe DSL (nvidia-cutlass-dsl) and a CUDA device are required",
)

M_QUERIES = 16
HEAD_DIM = 16


def _reference(Q: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 参考实现：S = Q @ K^T 的逐行 max 与 exp(S - m) 的逐行和。"""
    S = Q.float() @ K.float().transpose(0, 1)
    m = S.max(dim=-1).values
    l = torch.exp(S - m[:, None]).sum(dim=-1)
    return m, l


@pytest.mark.parametrize("kv_len", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("seed", [0, 71])
def test_cute_online_softmax_matches_reference(kv_len: int, seed: int) -> None:
    from einf.executors.torch.dsl import cute_online_softmax

    torch.manual_seed(seed)
    Q = torch.randn((M_QUERIES, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((kv_len, HEAD_DIM), device="cuda", dtype=torch.bfloat16)

    m_ref, l_ref = _reference(Q, K)
    m, l = cute_online_softmax(Q, K)

    assert m.dtype is torch.float32
    assert l.dtype is torch.float32
    assert m.shape == (M_QUERIES,)
    assert l.shape == (M_QUERIES,)

    # m 是逐元素取 max，无累加误差，但 MMA 与 torch 的 K 方向累加顺序不同，仍留 f32 余量
    torch.testing.assert_close(m, m_ref, rtol=1e-5, atol=1e-5)
    # l 含 exp 与求和，同样按 f32 量级
    torch.testing.assert_close(l, l_ref, rtol=1e-5, atol=1e-5)


def test_cute_online_softmax_reconstructs_softmax() -> None:
    """(m, l) 必须足以还原 softmax：exp(S - m) / l == softmax(S)。"""
    from einf.executors.torch.dsl import cute_online_softmax

    torch.manual_seed(1234)
    Q = torch.randn((M_QUERIES, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((64, HEAD_DIM), device="cuda", dtype=torch.bfloat16)

    m, l = cute_online_softmax(Q, K)

    S = Q.float() @ K.float().transpose(0, 1)
    actual = torch.exp(S - m[:, None]) / l[:, None]
    expected = torch.softmax(S, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_cute_online_softmax_is_shift_invariant() -> None:
    """给 Q 加一个大常数偏移不应导致溢出 —— 这正是 online softmax 的意义。

    K 全为正、Q 加大偏移会把 score 推到 exp 溢出区；只要减了 m 就不会 inf/NaN。
    """
    from einf.executors.torch.dsl import cute_online_softmax

    torch.manual_seed(5)
    Q = torch.full((M_QUERIES, HEAD_DIM), 40.0, device="cuda", dtype=torch.bfloat16)
    K = torch.full((32, HEAD_DIM), 40.0, device="cuda", dtype=torch.bfloat16)

    m, l = cute_online_softmax(Q, K)

    assert torch.isfinite(m).all(), "m 出现 inf/NaN"
    assert torch.isfinite(l).all(), "l 出现 inf/NaN"
    # 所有 score 相同 -> 每行 l 恰好等于列数
    torch.testing.assert_close(l, torch.full_like(l, 32.0), rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    ("shape_q", "shape_k", "dtype", "match"),
    [
        ((8, HEAD_DIM), (32, HEAD_DIM), torch.bfloat16, "Q must have shape"),
        ((M_QUERIES, HEAD_DIM), (32, 8), torch.bfloat16, "K must have shape"),
        ((M_QUERIES, HEAD_DIM), (12, HEAD_DIM), torch.bfloat16, "multiple of 8"),
        ((M_QUERIES, HEAD_DIM), (0, HEAD_DIM), torch.bfloat16, "multiple of 8"),
        ((M_QUERIES, HEAD_DIM), (32, HEAD_DIM), torch.float32, "requires BF16"),
    ],
)
def test_cute_online_softmax_rejects_unsupported_inputs(
    shape_q: tuple[int, int],
    shape_k: tuple[int, int],
    dtype: torch.dtype,
    match: str,
) -> None:
    """入参校验在 host 侧，与 kernel 是否实现无关，所以这一项现在就该是绿的。"""
    from einf.executors.torch.dsl import cute_online_softmax

    Q = torch.zeros(shape_q, device="cuda", dtype=dtype)
    K = torch.zeros(shape_k, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=match):
        cute_online_softmax(Q, K)
