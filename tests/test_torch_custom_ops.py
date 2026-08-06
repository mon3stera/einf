import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from einf.executors.torch.ops import (
    contiguous_attention,
    custom_ops_available,
    flash_attention,
    gather_context,
    load_custom_ops,
    paged_decode_attention,
    paged_decode_attention_split_kv,
    write_slots_,
)


def _contiguous_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    start_pos: int,
    scale: float,
) -> torch.Tensor:
    num_attention_heads = Q.shape[1]
    num_kv_heads = K.shape[1]
    group_size = num_attention_heads // num_kv_heads
    repeated_K = K.repeat_interleave(group_size, dim=1).float()
    repeated_V = V.repeat_interleave(group_size, dim=1).float()

    scores = torch.einsum("qhd,khd->qhk", Q.float(), repeated_K) * scale
    query_positions = start_pos + torch.arange(Q.shape[0], device=Q.device)
    key_positions = torch.arange(K.shape[0], device=K.device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    probabilities = torch.softmax(
        scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf")),
        dim=-1,
    )
    output = torch.einsum("qhk,khd->qhd", probabilities, repeated_V)
    return output.to(Q.dtype)


def _paged_decode_attention_reference(
    Q: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    context_len: int,
    scale: float,
) -> torch.Tensor:
    block_len = K_cache.shape[1]
    positions = torch.arange(context_len, dtype=torch.long, device=Q.device)
    logical_blocks = positions // block_len
    block_offsets = positions % block_len
    physical_blocks = block_table[logical_blocks]
    slots = physical_blocks * block_len + block_offsets
    K = K_cache.view(-1, K_cache.shape[2], K_cache.shape[3]).index_select(
        0, slots
    )
    V = V_cache.view(-1, V_cache.shape[2], V_cache.shape[3]).index_select(
        0, slots
    )
    return _contiguous_attention_reference(
        Q.unsqueeze(0),
        K,
        V,
        start_pos=context_len - 1,
        scale=scale,
    ).squeeze(0)


@pytest.mark.skipif(CUDA_HOME is None, reason="CUDA toolkit is unavailable")
def test_custom_ops_extension_registers_write_slots_schema() -> None:
    load_custom_ops()

    assert custom_ops_available()
    assert hasattr(torch.ops.einf, "write_slots_")
    assert hasattr(torch.ops.einf, "gather_context")
    assert hasattr(torch.ops.einf, "contiguous_attention")
    assert hasattr(torch.ops.einf, "flash_attention")
    assert hasattr(torch.ops.einf, "paged_decode_attention")
    assert hasattr(torch.ops.einf, "paged_decode_attention_split_kv")


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_splits", [1, 2, 4])
def test_paged_decode_attention_split_kv_matches_pytorch(
    dtype: torch.dtype,
    num_splits: int,
) -> None:
    torch.manual_seed(41)
    Q = torch.randn((4, 64), device="cuda", dtype=dtype)
    K_cache = torch.randn((6, 4, 2, 64), device="cuda", dtype=dtype)
    V_cache = torch.randn_like(K_cache)
    block_table = torch.tensor([3, 0, 5, 1], device="cuda", dtype=torch.long)
    context_len = 16
    scale = Q.shape[-1] ** -0.5

    expected = _paged_decode_attention_reference(
        Q,
        K_cache,
        V_cache,
        block_table,
        context_len=context_len,
        scale=scale,
    )
    actual = paged_decode_attention_split_kv(
        Q,
        K_cache,
        V_cache,
        block_table,
        context_len=context_len,
        num_splits=num_splits,
        scale=scale,
    )

    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_contiguous_attention_matches_pytorch(dtype: torch.dtype) -> None:
    torch.manual_seed(29)
    Q = torch.randn((3, 4, 8), device="cuda", dtype=dtype)
    K = torch.randn((5, 2, 8), device="cuda", dtype=dtype)
    V = torch.randn((5, 2, 8), device="cuda", dtype=dtype)
    start_pos = 2
    scale = Q.shape[-1] ** -0.5

    expected = _contiguous_attention_reference(
        Q,
        K,
        V,
        start_pos=start_pos,
        scale=scale,
    )
    actual = contiguous_attention(Q, K, V, start_pos, scale)

    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("q_len", "kv_len", "start_pos"),
    [(19, 19, 0), (6, 21, 15)],
)
def test_flash_attention_matches_pytorch(
    dtype: torch.dtype,
    q_len: int,
    kv_len: int,
    start_pos: int,
) -> None:
    torch.manual_seed(31)
    Q = torch.randn((q_len, 4, 64), device="cuda", dtype=dtype)
    K = torch.randn((kv_len, 2, 64), device="cuda", dtype=dtype)
    V = torch.randn((kv_len, 2, 64), device="cuda", dtype=dtype)
    scale = Q.shape[-1] ** -0.5

    expected = _contiguous_attention_reference(
        Q,
        K,
        V,
        start_pos=start_pos,
        scale=scale,
    )
    actual = flash_attention(Q, K, V, start_pos, scale)

    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("context_len", "physical_blocks"),
    [(1, [3]), (8, [3, 0]), (10, [3, 0, 5])],
)
def test_paged_decode_attention_matches_pytorch(
    dtype: torch.dtype,
    context_len: int,
    physical_blocks: list[int],
) -> None:
    torch.manual_seed(37)
    Q = torch.randn((4, 64), device="cuda", dtype=dtype)
    K_cache = torch.randn((6, 4, 2, 64), device="cuda", dtype=dtype)
    V_cache = torch.randn_like(K_cache)
    block_table = torch.tensor(
        physical_blocks,
        device="cuda",
        dtype=torch.long,
    )
    scale = Q.shape[-1] ** -0.5

    expected = _paged_decode_attention_reference(
        Q,
        K_cache,
        V_cache,
        block_table,
        context_len=context_len,
        scale=scale,
    )
    actual = paged_decode_attention(
        Q,
        K_cache,
        V_cache,
        block_table,
        context_len,
        scale,
    )

    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_write_slots_cuda_matches_pytorch_reference(dtype: torch.dtype) -> None:
    load_custom_ops()

    K_cache = torch.full((4, 2, 2, 4), -1, dtype=dtype, device="cuda")
    V_cache = torch.full_like(K_cache, -2)
    K_reference = K_cache.clone()
    V_reference = V_cache.clone()

    K = torch.arange(3 * 2 * 4, dtype=torch.float32, device="cuda").reshape(3, 2, 4).to(dtype)
    V = (K.to(torch.float32) + 100).to(dtype)
    slot_mapping = torch.tensor([5, 0, 7], dtype=torch.long, device="cuda")

    K_reference.view(-1, 2, 4).index_copy_(0, slot_mapping, K)
    V_reference.view(-1, 2, 4).index_copy_(0, slot_mapping, V)

    write_slots_(K_cache, V_cache, slot_mapping, K, V)

    torch.testing.assert_close(K_cache, K_reference, rtol=0, atol=0)
    torch.testing.assert_close(V_cache, V_reference, rtol=0, atol=0)


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
def test_write_slots_cuda_accepts_empty_input() -> None:
    load_custom_ops()

    K_cache = torch.ones((2, 2, 1, 4), device="cuda")
    V_cache = torch.ones_like(K_cache)
    K_before = K_cache.clone()
    V_before = V_cache.clone()

    write_slots_(
        K_cache,
        V_cache,
        torch.empty(0, dtype=torch.long, device="cuda"),
        torch.empty((0, 1, 4), device="cuda"),
        torch.empty((0, 1, 4), device="cuda"),
    )

    torch.testing.assert_close(K_cache, K_before)
    torch.testing.assert_close(V_cache, V_before)


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_context_cuda_matches_pytorch_reference(dtype: torch.dtype) -> None:
    load_custom_ops()

    K_cache = torch.arange(4 * 2 * 2 * 4, dtype=torch.float32, device="cuda")
    K_cache = K_cache.reshape(4, 2, 2, 4).to(dtype)
    V_cache = (K_cache.to(torch.float32) + 1000).to(dtype)
    block_table = torch.tensor([2, 0, 3], dtype=torch.long, device="cuda")
    context_len = 5

    positions = torch.arange(context_len, dtype=torch.long, device="cuda")
    logical_blocks = positions // K_cache.size(1)
    block_offsets = positions % K_cache.size(1)
    physical_blocks = block_table[logical_blocks]
    slots = physical_blocks * K_cache.size(1) + block_offsets
    K_reference = K_cache.view(-1, 2, 4).index_select(0, slots)
    V_reference = V_cache.view(-1, 2, 4).index_select(0, slots)

    K, V = gather_context(K_cache, V_cache, block_table, context_len)

    torch.testing.assert_close(K, K_reference, rtol=0, atol=0)
    torch.testing.assert_close(V, V_reference, rtol=0, atol=0)


@pytest.mark.skipif(
    CUDA_HOME is None or not torch.cuda.is_available(),
    reason="CUDA build/runtime is unavailable",
)
def test_gather_context_cuda_accepts_empty_context() -> None:
    load_custom_ops()

    cache = torch.empty((2, 2, 1, 4), device="cuda")
    block_table = torch.empty(0, dtype=torch.long, device="cuda")

    K, V = gather_context(cache, cache.clone(), block_table, 0)

    assert K.shape == (0, 1, 4)
    assert V.shape == (0, 1, 4)
