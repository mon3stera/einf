"""第 3.2 级 four-warp Split-Q CuTe FlashAttention 长期回归测试。

Public wrapper 使用 ``BLOCK_M=64``、``NUM_WARPS=4``、``STAGES=2``；每个 warp
独占连续的 16 行 Q。Direct-launch tests 还验证 constexpr tile/stage
specialization、四个 warp 的无重叠写回，以及非法 Split-Q 几何在编译阶段被拒绝。
"""

from __future__ import annotations

import math

import pytest
import torch

from einf.executors.torch.dsl import dsl_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not dsl_available(),
    reason="CuTe DSL (nvidia-cutlass-dsl) and a CUDA device are required",
)

BLOCK_M = 64
BLOCK_N = 16
HEAD_DIM = 64
NUM_WARPS = 4
STAGES = 2
M_PER_WARP = BLOCK_M // NUM_WARPS
NEG_INF = -1.0e30
GMEM_COPY_BYTES = 16


def _from_dlpack_128bit(*tensors: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    assert all(tensor.data_ptr() % GMEM_COPY_BYTES == 0 for tensor in tensors)
    return tuple(
        from_dlpack(tensor, assumed_align=GMEM_COPY_BYTES)
        for tensor in tensors
    )


def _online_bf16_causal_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    start_pos: int,
    scale: float,
    block_n: int = BLOCK_N,
) -> torch.Tensor:
    """Match tile-local BF16-P quantization and FP32 online-rescale."""
    q_len, num_attention_heads, head_dim = Q.shape
    num_kv_heads = K.shape[1]
    group_size = num_attention_heads // num_kv_heads
    repeated_K = K.repeat_interleave(group_size, dim=1)
    repeated_V = V.repeat_interleave(group_size, dim=1)

    m = torch.full(
        (q_len, num_attention_heads),
        NEG_INF,
        dtype=torch.float32,
        device=Q.device,
    )
    l = torch.zeros_like(m)
    numerator = torch.zeros(
        (q_len, num_attention_heads, head_dim),
        dtype=torch.float32,
        device=Q.device,
    )
    query_positions = start_pos + torch.arange(q_len, device=Q.device)

    for key_start in range(0, K.shape[0], block_n):
        K_tile = repeated_K[key_start : key_start + block_n]
        V_tile = repeated_V[key_start : key_start + block_n]
        scores = torch.einsum("qhd,khd->qhk", Q.float(), K_tile.float()) * scale
        key_positions = torch.arange(
            key_start,
            key_start + block_n,
            device=Q.device,
        )
        causal = key_positions[None, :] <= query_positions[:, None]
        scores = scores.masked_fill(~causal[:, None, :], NEG_INF)

        m_new = torch.maximum(m, scores.max(dim=-1).values)
        alpha = torch.exp(m - m_new)
        probabilities = torch.exp(scores - m_new[:, :, None])
        numerator = (
            numerator * alpha[:, :, None]
            + torch.einsum(
                "qhk,khd->qhd",
                probabilities.to(torch.bfloat16).float(),
                V_tile.float(),
            )
        )
        l = l * alpha + probabilities.sum(dim=-1)
        m = m_new

    return (numerator / l[:, :, None]).to(torch.bfloat16)


def _random_inputs(
    *,
    q_len: int,
    kv_len: int,
    num_attention_heads: int,
    num_kv_heads: int,
    seed: int,
    head_dim: int = HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    Q = torch.randn(
        (q_len, num_attention_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    K = torch.randn(
        (kv_len, num_kv_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    V = torch.randn_like(K)
    return Q, K, V


@pytest.mark.parametrize(
    ("q_len", "kv_len", "start_pos", "num_attention_heads", "num_kv_heads"),
    [
        (BLOCK_M, 4 * BLOCK_N, 0, 1, 1),
        (BLOCK_M, 6 * BLOCK_N, 2 * BLOCK_N, 4, 2),
        (2 * BLOCK_M, 8 * BLOCK_N, 0, 4, 2),
        (2 * BLOCK_M, 10 * BLOCK_N, 2 * BLOCK_N, 8, 2),
    ],
)
@pytest.mark.parametrize("seed", [0, 71])
def test_cute_flash_attention_matches_online_bf16_causal_reference(
    q_len: int,
    kv_len: int,
    start_pos: int,
    num_attention_heads: int,
    num_kv_heads: int,
    seed: int,
) -> None:
    from einf.executors.torch.dsl import cute_flash_attention

    Q, K, V = _random_inputs(
        q_len=q_len,
        kv_len=kv_len,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        seed=seed,
    )
    scale = HEAD_DIM**-0.5

    actual = cute_flash_attention(Q, K, V, start_pos, scale)
    expected = _online_bf16_causal_reference(
        Q,
        K,
        V,
        start_pos=start_pos,
        scale=scale,
    )

    assert actual.dtype is torch.bfloat16
    assert actual.shape == Q.shape
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("head_dim", [128, 256])
def test_cute_flash_attention_supports_mainstream_large_head_dims(
    head_dim: int,
) -> None:
    from einf.executors.torch.dsl import cute_flash_attention

    Q, K, V = _random_inputs(
        q_len=BLOCK_M,
        kv_len=2 * BLOCK_M,
        num_attention_heads=4,
        num_kv_heads=2,
        seed=2000 + head_dim,
        head_dim=head_dim,
    )
    start_pos = BLOCK_M
    scale = head_dim**-0.5

    actual = cute_flash_attention(Q, K, V, start_pos, scale)
    expected = _online_bf16_causal_reference(
        Q,
        K,
        V,
        start_pos=start_pos,
        scale=scale,
    )

    assert actual.shape == Q.shape
    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("block_m", "block_n", "head_dim", "num_warps", "stages"),
    [
        (64, 16, 64, 4, 2),
        (64, 16, 64, 4, 3),
        (64, 32, 64, 4, 2),
        (64, 64, 64, 4, 2),
        (128, 16, 64, 4, 2),
        (64, 16, 128, 4, 2),
        (64, 16, 256, 4, 2),
    ],
)
def test_flash_attention_launch_specializes_split_q_tiles_and_stages(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_warps: int,
    stages: int,
) -> None:
    import cutlass.cute as cute
    from einf.executors.torch.dsl.flash_attention import _flash_attention_launch

    assert block_m % num_warps == 0
    assert (block_m // num_warps) % 16 == 0
    torch.manual_seed(
        1000 + block_m * 10 + block_n + head_dim + num_warps + stages
    )
    q_len = 2 * block_m
    kv_len = max(q_len, 4 * block_n)
    start_pos = kv_len - q_len
    Q = torch.randn((q_len, 4, head_dim), device="cuda", dtype=torch.bfloat16)
    K = torch.randn((kv_len, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    V = torch.randn_like(K)
    O = torch.empty_like(Q)
    mQ, mK, mV, mO = _from_dlpack_128bit(Q, K, V, O)
    scale = head_dim**-0.5

    compiled = cute.compile(
        _flash_attention_launch,
        mQ,
        mK,
        mV,
        mO,
        start_pos,
        scale,
        block_m,
        block_n,
        head_dim,
        num_warps,
        stages,
    )
    compiled(mQ, mK, mV, mO, start_pos, scale)
    torch.cuda.synchronize()

    expected = _online_bf16_causal_reference(
        Q,
        K,
        V,
        start_pos=start_pos,
        scale=scale,
        block_n=block_n,
    )
    torch.testing.assert_close(O, expected, rtol=2e-2, atol=2e-2)


def test_four_warps_write_distinct_split_q_rows() -> None:
    """Each warp must write its own 16-row slice of the 64-row CTA tile."""
    import cutlass.cute as cute
    from einf.executors.torch.dsl.flash_attention import _flash_attention_launch

    assert M_PER_WARP == 16
    q_len = kv_len = BLOCK_M
    Q = torch.zeros((q_len, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros_like(Q)
    token_values = torch.arange(kv_len, device="cuda", dtype=torch.float32)
    V = token_values[:, None, None].expand_as(Q).to(torch.bfloat16)
    O = torch.full_like(Q, float("nan"))
    mQ, mK, mV, mO = _from_dlpack_128bit(Q, K, V, O)

    compiled = cute.compile(
        _flash_attention_launch,
        mQ,
        mK,
        mV,
        mO,
        0,
        1.0,
        BLOCK_M,
        BLOCK_N,
        HEAD_DIM,
        NUM_WARPS,
        STAGES,
    )
    compiled(mQ, mK, mV, mO, 0, 1.0)
    torch.cuda.synchronize()

    prefix_mean = token_values.cumsum(0) / torch.arange(
        1,
        kv_len + 1,
        device="cuda",
    )
    expected = prefix_mean[:, None, None].expand_as(Q).to(torch.bfloat16)
    for warp_id in range(NUM_WARPS):
        row_start = warp_id * M_PER_WARP
        row_end = row_start + M_PER_WARP
        assert torch.isfinite(O[row_start:row_end]).all()
        torch.testing.assert_close(
            O[row_start:row_end],
            expected[row_start:row_end],
            rtol=1e-2,
            atol=1e-2,
        )


def test_cute_flash_attention_grid_maps_q_tiles_and_causal_rows() -> None:
    """Every blockIdx.x tile must write its own rows with the right causal prefix."""
    from einf.executors.torch.dsl import cute_flash_attention

    q_len = kv_len = 2 * BLOCK_M
    Q = torch.zeros((q_len, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((kv_len, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    token_values = torch.arange(kv_len, device="cuda", dtype=torch.float32)
    V = token_values[:, None, None].expand(kv_len, 1, HEAD_DIM).to(torch.bfloat16)

    actual = cute_flash_attention(Q, K, V, 0, 1.0)
    prefix_mean = token_values.cumsum(0) / torch.arange(
        1,
        kv_len + 1,
        device="cuda",
    )
    expected = prefix_mean[:, None, None].expand_as(Q).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_cute_flash_attention_grid_y_maps_gqa_heads() -> None:
    """blockIdx.y query heads must select the correct grouped KV head."""
    from einf.executors.torch.dsl import cute_flash_attention

    q_len, kv_len, start_pos = BLOCK_M, 6 * BLOCK_N, 2 * BLOCK_N
    Q = torch.zeros((q_len, 4, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((kv_len, 2, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.empty_like(K)
    V[:, 0, :].fill_(1.0)
    V[:, 1, :].fill_(5.0)

    actual = cute_flash_attention(Q, K, V, start_pos, 1.0)
    expected_per_head = torch.tensor(
        [1.0, 1.0, 5.0, 5.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = expected_per_head[None, :, None].expand_as(Q)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_cute_flash_attention_rescales_old_output_when_later_max_grows() -> None:
    """Catch implementations that update m/l but forget ``O_old *= alpha``."""
    from einf.executors.torch.dsl import cute_flash_attention

    q_len, kv_len, start_pos = BLOCK_M, 6 * BLOCK_N, 2 * BLOCK_N
    Q = torch.ones((q_len, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((kv_len, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K[2 * BLOCK_N :].fill_(1.0)
    V = torch.full_like(K, -7.0)
    V[2 * BLOCK_N :].fill_(3.0)
    scale = HEAD_DIM**-0.5

    actual = cute_flash_attention(Q, K, V, start_pos, scale)
    expected = _online_bf16_causal_reference(
        Q,
        K,
        V,
        start_pos=start_pos,
        scale=scale,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert torch.all(actual[-1] > 2.99), "later dominant tile did not suppress old O"


def test_cute_flash_attention_is_score_shift_stable_across_tiles() -> None:
    """Large equal scores over several causal tiles must remain finite."""
    from einf.executors.torch.dsl import cute_flash_attention

    q_len, kv_len, start_pos = BLOCK_M, 6 * BLOCK_N, 2 * BLOCK_N
    Q = torch.full(
        (q_len, 2, HEAD_DIM),
        40.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    K = torch.full(
        (kv_len, 1, HEAD_DIM),
        40.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    V = torch.randn((kv_len, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    scale = HEAD_DIM**-0.5

    actual = cute_flash_attention(Q, K, V, start_pos, scale)
    expected = _online_bf16_causal_reference(
        Q,
        K,
        V,
        start_pos=start_pos,
        scale=scale,
    )
    assert torch.isfinite(actual).all(), "O contains inf/NaN"
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("shape_q", "shape_k", "shape_v", "dtype", "match"),
    [
        (
            (BLOCK_M, HEAD_DIM),
            (BLOCK_M, 1, HEAD_DIM),
            (BLOCK_M, 1, HEAD_DIM),
            torch.bfloat16,
            "Q must have shape",
        ),
        (
            (BLOCK_M, 2, HEAD_DIM),
            (BLOCK_M, HEAD_DIM),
            (BLOCK_M, 1, HEAD_DIM),
            torch.bfloat16,
            "K must have shape",
        ),
        (
            (BLOCK_M, 2, HEAD_DIM),
            (BLOCK_M, 1, HEAD_DIM),
            (BLOCK_M, 2, HEAD_DIM),
            torch.bfloat16,
            "V must match K shape",
        ),
        (
            (BLOCK_M, 2, HEAD_DIM),
            (BLOCK_M, 1, HEAD_DIM),
            (BLOCK_M, 1, HEAD_DIM),
            torch.float32,
            "requires BF16",
        ),
        (
            (BLOCK_M, 2, 32),
            (BLOCK_M, 1, 32),
            (BLOCK_M, 1, 32),
            torch.bfloat16,
            "head_dim must be at least 64",
        ),
        (
            (BLOCK_M, 2, 64),
            (BLOCK_M, 1, 128),
            (BLOCK_M, 1, 128),
            torch.bfloat16,
            "head_dim must match",
        ),
    ],
)
def test_cute_flash_attention_rejects_unsupported_inputs(
    shape_q: tuple[int, ...],
    shape_k: tuple[int, ...],
    shape_v: tuple[int, ...],
    dtype: torch.dtype,
    match: str,
) -> None:
    from einf.executors.torch.dsl import cute_flash_attention

    Q = torch.zeros(shape_q, device="cuda", dtype=dtype)
    K = torch.zeros(shape_k, device="cuda", dtype=dtype)
    V = torch.zeros(shape_v, device="cuda", dtype=dtype)
    with pytest.raises(ValueError, match=match):
        cute_flash_attention(Q, K, V, 0, 1.0)


@pytest.mark.parametrize(
    ("block_m", "block_n", "head_dim", "num_warps", "stages", "match"),
    [
        (8, 16, 64, 4, 2, "block_m must be a positive multiple of 16"),
        (64, 8, 64, 4, 2, "block_n must be a positive multiple of 16"),
        (64, 16, 16, 4, 2, "head_dim must be at least 64"),
        (64, 16, 32, 4, 2, "head_dim must be at least 64"),
        (48, 16, 64, 4, 2, "block_m / num_warps must be a multiple of 16"),
        (64, 16, 64, 3, 2, "num_warps must be one of 1, 2, 4, or 8"),
        (
            128,
            16,
            64,
            8,
            2,
            "K/V tile must divide evenly across CTA threads",
        ),
    ],
)
def test_flash_attention_launch_rejects_invalid_split_q_specializations(
    block_m: int,
    block_n: int,
    head_dim: int,
    num_warps: int,
    stages: int,
    match: str,
) -> None:
    import cutlass.cute as cute
    from einf.executors.torch.dsl.flash_attention import _flash_attention_launch

    test_block_m = max(BLOCK_M, block_m)
    test_block_n = max(BLOCK_N, block_n)
    test_head_dim = max(HEAD_DIM, head_dim)
    Q = torch.zeros(
        (test_block_m, 1, test_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    K = torch.zeros(
        (test_block_n, 1, test_head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    V = torch.zeros_like(K)
    O = torch.empty_like(Q)
    mQ, mK, mV, mO = _from_dlpack_128bit(Q, K, V, O)

    with pytest.raises((AssertionError, RuntimeError), match=match):
        cute.compile(
            _flash_attention_launch,
            mQ,
            mK,
            mV,
            mO,
            0,
            0.125,
            block_m,
            block_n,
            head_dim,
            num_warps,
            stages,
        )


@pytest.mark.parametrize(
    ("q_len", "kv_len", "start_pos", "match"),
    [
        (0, BLOCK_N, BLOCK_N, "q_len must be positive"),
        (BLOCK_M, 0, -BLOCK_M, "kv_len must be positive"),
        (BLOCK_M // 2, BLOCK_M, BLOCK_M // 2, "q_len to be a multiple of 64"),
        (BLOCK_M, BLOCK_M + 8, 8, "kv_len to be a multiple of 16"),
        (BLOCK_M, BLOCK_M, -1, "start_pos must be non-negative"),
        (BLOCK_M, 2 * BLOCK_M, 0, r"start_pos \+ q_len must equal kv_len"),
    ],
)
def test_cute_flash_attention_rejects_invalid_lengths(
    q_len: int,
    kv_len: int,
    start_pos: int,
    match: str,
) -> None:
    from einf.executors.torch.dsl import cute_flash_attention

    Q = torch.zeros((q_len, 2, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((kv_len, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.zeros_like(K)
    with pytest.raises(ValueError, match=match):
        cute_flash_attention(Q, K, V, start_pos, 1.0)


def test_cute_flash_attention_rejects_invalid_gqa_geometry() -> None:
    from einf.executors.torch.dsl import cute_flash_attention

    Q = torch.zeros((BLOCK_M, 3, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((BLOCK_M, 2, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.zeros_like(K)
    with pytest.raises(ValueError, match="num_attention_heads must be divisible"):
        cute_flash_attention(Q, K, V, 0, 1.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_cute_flash_attention_rejects_invalid_scale(scale: float) -> None:
    from einf.executors.torch.dsl import cute_flash_attention

    Q = torch.zeros((BLOCK_M, 2, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((BLOCK_M, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.zeros_like(K)
    with pytest.raises(ValueError, match="scale must be finite and positive"):
        cute_flash_attention(Q, K, V, 0, scale)


def test_cute_flash_attention_rejects_noncontiguous_inputs() -> None:
    from einf.executors.torch.dsl import cute_flash_attention

    Q = torch.zeros(
        (BLOCK_M, 2, 2 * HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )[:, :, ::2]
    K = torch.zeros((BLOCK_M, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.zeros_like(K)
    assert not Q.is_contiguous()
    with pytest.raises(ValueError, match="must be contiguous"):
        cute_flash_attention(Q, K, V, 0, 1.0)


def test_cute_flash_attention_rejects_device_mismatch() -> None:
    """Host validation must run before the intentional scaffold gate."""
    from einf.executors.torch.dsl import cute_flash_attention

    Q = torch.zeros((BLOCK_M, 2, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    K = torch.zeros((BLOCK_M, 1, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    V = torch.zeros((BLOCK_M, 1, HEAD_DIM), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must be CUDA tensors"):
        cute_flash_attention(Q, K, V, 0, 1.0)
