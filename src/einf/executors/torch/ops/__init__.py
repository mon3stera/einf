import torch

from einf.executors.torch.ops.loader import (
    custom_ops_available,
    load_custom_ops,
)


def write_slots_(
    K_cache,
    V_cache,
    slot_mapping,
    K,
    V,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Load the extension and invoke the in-place KV-cache write operator.

    ``k_scale``/``v_scale`` multiply the values before they are quantized into
    an FP8 cache; they are no-ops for a floating-point cache.
    """
    load_custom_ops()
    torch.ops.einf.write_slots_(
        K_cache,
        V_cache,
        slot_mapping,
        K,
        V,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def gather_context(K_cache, V_cache, block_table, context_len: int):
    """Load the extension and gather one request's contiguous KV context."""
    load_custom_ops()
    return torch.ops.einf.gather_context(
        K_cache,
        V_cache,
        block_table,
        context_len,
    )


def marlin_gemm(A, b_qweight, scales, out, workspace, *, group_size: int, max_par: int = 8) -> None:
    """Load the extension and run the vendored Marlin W4A16 GEMM into ``out``."""
    load_custom_ops()
    torch.ops.einf.marlin_gemm(
        A,
        b_qweight,
        scales,
        out,
        workspace,
        group_size=group_size,
        max_par=max_par,
    )


def contiguous_attention(Q, K, V, start_pos: int, scale: float):
    """Run the correctness-first contiguous causal Attention operator."""
    load_custom_ops()
    return torch.ops.einf.contiguous_attention(
        Q,
        K,
        V,
        start_pos,
        scale,
    )


def cute_copy(input):
    """Run the first CuTe Global-to-Register-to-Global copy exercise."""
    load_custom_ops()
    return torch.ops.einf.cute_copy(input)


def cute_elementwise_add(X, Y):
    """Run the arbitrary-length CuTe element-wise addition exercise."""
    load_custom_ops()
    return torch.ops.einf.cute_elementwise_add(X, Y)


def cute_gemm(A, B):
    """Run the arbitrary-shape FP32 CuTe SIMT GEMM exercise."""
    load_custom_ops()
    return torch.ops.einf.cute_gemm(A, B)


def cute_mma_qk(Q, K):
    """Run the one-Warp CuTe BF16 m16n8k16 QK learning exercise."""
    load_custom_ops()
    return torch.ops.einf.cute_mma_qk(Q, K)


def cute_reduce_sum(input):
    """Run the arbitrary-length CuTe FP32 reduction exercise."""
    load_custom_ops()
    return torch.ops.einf.cute_reduce_sum(input)


def cute_shared_copy(input):
    """Run the CuTe Global-to-Shared-to-Global copy exercise."""
    load_custom_ops()
    return torch.ops.einf.cute_shared_copy(input)


def cute_transpose(input):
    """Run the CuTe Shared-Memory tiled transpose exercise."""
    load_custom_ops()
    return torch.ops.einf.cute_transpose(input)


def flash_attention(Q, K, V, start_pos: int, scale: float):
    """Run the correctness-first FlashAttention-style forward operator."""
    load_custom_ops()
    return torch.ops.einf.flash_attention(
        Q,
        K,
        V,
        start_pos,
        scale,
    )


def tensor_core_qk(Q, K):
    """Run the BF16 Tensor Core QK learning operator."""
    load_custom_ops()
    return torch.ops.einf.tensor_core_qk(Q, K)


def paged_decode_attention(
    Q,
    K_cache,
    V_cache,
    block_table,
    context_len: int,
    scale: float,
):
    """Run one request's q_len=1 Paged Decode Attention operator."""
    load_custom_ops()
    return torch.ops.einf.paged_decode_attention(
        Q,
        K_cache,
        V_cache,
        block_table,
        context_len,
        scale,
    )


def paged_decode_attention_split_kv(
    Q,
    K_cache,
    V_cache,
    block_table,
    context_len: int,
    num_splits: int,
    scale: float,
):
    """Run the two-stage Split-KV Paged Decode learning operator."""
    load_custom_ops()
    return torch.ops.einf.paged_decode_attention_split_kv(
        Q,
        K_cache,
        V_cache,
        block_table,
        context_len,
        num_splits,
        scale,
    )


def paged_decode_attention_batched(
    query,
    K_cache,
    V_cache,
    block_tables,
    context_lens,
    query_start_loc,
    single_query_request_indices,
    scale: float,
):
    """Run the packed mixed-batch Paged Decode learning operator."""
    load_custom_ops()
    return torch.ops.einf.paged_decode_attention_batched(
        query,
        K_cache,
        V_cache,
        block_tables,
        context_lens,
        query_start_loc,
        single_query_request_indices,
        scale,
    )


__all__ = [
    "contiguous_attention",
    "cute_copy",
    "cute_elementwise_add",
    "cute_gemm",
    "cute_mma_qk",
    "cute_reduce_sum",
    "cute_shared_copy",
    "cute_transpose",
    "custom_ops_available",
    "flash_attention",
    "gather_context",
    "load_custom_ops",
    "marlin_gemm",
    "paged_decode_attention_batched",
    "paged_decode_attention",
    "paged_decode_attention_split_kv",
    "tensor_core_qk",
    "write_slots_",
]
