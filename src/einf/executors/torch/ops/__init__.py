import torch

from einf.executors.torch.ops.loader import (
    custom_ops_available,
    load_custom_ops,
)


def write_slots_(K_cache, V_cache, slot_mapping, K, V) -> None:
    """Load the extension and invoke the in-place KV-cache write operator."""
    load_custom_ops()
    torch.ops.einf.write_slots_(K_cache, V_cache, slot_mapping, K, V)


def gather_context(K_cache, V_cache, block_table, context_len: int):
    """Load the extension and gather one request's contiguous KV context."""
    load_custom_ops()
    return torch.ops.einf.gather_context(
        K_cache,
        V_cache,
        block_table,
        context_len,
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


__all__ = [
    "contiguous_attention",
    "custom_ops_available",
    "flash_attention",
    "gather_context",
    "load_custom_ops",
    "paged_decode_attention",
    "paged_decode_attention_split_kv",
    "write_slots_",
]
