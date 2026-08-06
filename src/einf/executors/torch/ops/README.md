# Torch Custom Ops

This package is the boundary between einf's Python control plane and its
C++/CUDA data plane.

## Current scope

The extension registers five CUDA-focused operators:

```text
einf::write_slots_(
    K_cache,       # [num_blocks, block_len, num_kv_heads, head_dim]
    V_cache,       # [num_blocks, block_len, num_kv_heads, head_dim]
    slot_mapping,  # [num_tokens], int64
    K,             # [num_tokens, num_kv_heads, head_dim]
    V,             # [num_tokens, num_kv_heads, head_dim]
) -> None
```

`K_cache` and `V_cache` are mutated in place. Python selects the layer before
calling the operator, so the native code only owns physical-slot addressing.

```text
einf::gather_context(
    K_cache,       # [num_blocks, block_len, num_kv_heads, head_dim]
    V_cache,       # [num_blocks, block_len, num_kv_heads, head_dim]
    block_table,   # [num_logical_blocks], int64
    context_len,
) -> (K, V)        # [context_len, num_kv_heads, head_dim]
```

`gather_context` directly converts logical context positions through the block
table into physical Cache reads. A separate low-level `read_slots` operator is
not needed until another consumer requires arbitrary precomputed slots.

```text
einf::contiguous_attention(
    Q,          # [q_len, num_attention_heads, head_dim]
    K,          # [kv_len, num_kv_heads, head_dim]
    V,          # [kv_len, num_kv_heads, head_dim]
    start_pos,  # kv_len - q_len
    scale,
) -> output     # [q_len, num_attention_heads, head_dim]
```

`contiguous_attention` is the Gate 5.0 learning/reference operator. Its first
implementation intentionally materializes an FP32 score Tensor and launches
separate QK, stable-softmax, and PV kernels. It supports full Prefill, chunked
Prefill, and Decode through the same causal `start_pos` contract.

```text
einf::flash_attention(
    Q,          # [q_len, num_attention_heads, head_dim]
    K,          # [kv_len, num_kv_heads, head_dim]
    V,          # [kv_len, num_kv_heads, head_dim]
    start_pos,  # kv_len - q_len
    scale,
) -> output     # [q_len, num_attention_heads, head_dim]
```

`flash_attention` preserves the contiguous operator contract but must not
materialize `[q_len,Hq,kv_len]` scores. One CUDA program owns a Query tile/head,
iterates K/V tiles, and merges row-wise FP32 max, denominator, and output
accumulators before normalizing once.

```text
einf::paged_decode_attention(
    Q,             # [num_attention_heads, head_dim]
    K_cache,       # [num_blocks, block_len, num_kv_heads, head_dim]
    V_cache,       # [num_blocks, block_len, num_kv_heads, head_dim]
    block_table,   # [num_logical_blocks], int64
    context_len,
    scale,
) -> output        # [num_attention_heads, head_dim]
```

The initial Gate 5 contract handles one request with `q_len=1`. It performs GQA
head mapping and directly reads block-based K/V without materializing a
contiguous Context Tensor.

The separate two-stage learning operator preserves that completed baseline:

```text
einf::paged_decode_attention_split_kv(
    Q,
    K_cache,
    V_cache,
    block_table,
    context_len,
    num_splits,
    scale,
) -> output
```

Stage 1 launches a two-dimensional `[Hq,num_splits]` Grid and writes one FP32
`(m,l,acc[D])` state per `(query_head,Context split)`. Stage 2 launches one CTA
per Query head with one Warp, merges those states, and normalizes once. Both
learner-authored CUDA stages are implemented and parity-tested.

## Files

- `loader.py`: explicit JIT build/load entry point.
- `csrc/kv_cache.h`: native function declarations.
- `csrc/kv_cache.cpp`: operator schema, dispatcher registration, validation,
  and an intentionally unsupported CPU implementation.
- `csrc/kv_cache_cuda.cu`: CUDA launcher and write kernel.
- `csrc/gather_context.cpp`: gather schema, validation, and an intentionally
  unsupported CPU implementation.
- `csrc/gather_context_cuda.cu`: CUDA gather launcher and kernel.
- `csrc/contiguous_attention.cpp`: Gate 5.0 schema, validation, and unsupported
  CPU dispatcher.
- `csrc/contiguous_attention_cuda.cu`: working learner-authored QK, in-place
  stable softmax, and PV kernels with a CUDA launcher.
- `csrc/flash_attention.cpp`: FlashAttention schema, shared contiguous-input
  validation, and unsupported CPU dispatcher.
- `csrc/flash_attention_cuda.cu`: working learner-authored tiled forward kernel
  plus CUDA launcher. Version 0 uses four Query rows per block, sixteen K/V rows
  per tile, `head_dim=64`, FP32 online-softmax/output accumulation, and no global
  score Tensor.
- `csrc/paged_attention.cpp`: Gate 5 schema, validation, and unsupported CPU
  dispatcher.
- `csrc/paged_attention_cuda.cu`: completed four-Warp Vec2 Paged Decode baseline.
- `csrc/paged_attention_split_kv.cpp`: separate Flash-Decoding-style schema,
  Split-count validation, and unsupported CPU dispatcher.
- `csrc/paged_attention_split_kv_cuda.cu`: completed learner-authored two-stage
  Split-KV kernels, FP32 workspace allocation, and launch plumbing.

## Build the scaffold

```bash
python -m einf.executors.torch.ops --verbose
```

The first invocation compiles into PyTorch's extension cache. Later invocations
reuse the compiled library until the source changes.

## CUDA contract

The CUDA implementation preserves these invariants:

```text
flat_cache[slot_mapping[token], head, dim] = current[token, head, dim]
```

Parity tests compare the custom op with PyTorch `index_copy_` for
non-contiguous slots, block-boundary writes, multiple KV heads, FP32, and BF16.

The CPU dispatcher entry deliberately raises an unsupported error. Gate 4 is
focused on the CUDA data path; add a CPU kernel only when a concrete use case
requires it.

## Completed Gate 5.0 reference

`contiguous_attention` implements:

```text
QK kernel
→ FP32 score Tensor [q_len, Hq, kv_len]
→ in-place stable row softmax
→ PV kernel
→ output [q_len, Hq, D]
```

CUDA parity tests cover FP32/BF16, causal chunked Prefill, and GQA. This
deliberately simple implementation is retained as a native oracle and will not
be optimized.

## Completed Gate 5.1 forward

`flash_attention_forward_kernel` now uses tiled Q/K/V and row-wise FP32
online-softmax merge under the same Q/K/V/start_pos contract as
`contiguous_attention`. FP32/BF16 parity tests cover full and chunked causal
Prefill, partial tiles, and GQA. Version 0 remains single-request, forward-only,
contiguous, restricted to `head_dim=64`, and has no dropout/backward contract.
It can optionally replace the eager body through
`QwenModelRunner(..., use_flash_attention=True)`; eager remains the default.

## Completed Gate 5.2 correctness baseline

`paged_decode_attention_kernel` now implements:

```text
query head → KV head group
→ iterate logical context positions
→ map through block_table
→ QK dot product with scale
→ FP32 online softmax
→ weighted V accumulation
```

The CUDA launcher uses four Warps per Query head. Each Warp processes a
contiguous logical-block range, writes partial FP32 `(m,l,acc)` state to Shared
Memory, and Warp 0 merges the states before normalization. FP32/FP16/BF16 tests
cover GQA, non-consecutive physical blocks, exact block boundaries, and partial
final blocks. The existing `gather_context + Attention` paths remain numerical
and performance oracles for batched/model integration.

## Gate 5.2 Split-KV correctness implementation

The completed single-CTA operator remains unchanged as a correctness and
performance baseline. `paged_decode_attention_split_kv` adds no K/V copy: each
Stage-1 CTA maps `split_idx` to a contiguous logical-block interval. Its FP32
workspace has shapes:

```text
partial_m   [Hq,num_splits]
partial_l   [Hq,num_splits]
partial_acc [Hq,num_splits,D]
```

The first learning version keeps four Warps per Stage-1 CTA and requires
`1 <= num_splits <= num_logical_blocks`. Adaptive split selection, GQA KV reuse,
and tiled/Tensor-Core execution remain later experiments. FP32/FP16/BF16 parity
tests cover `num_splits=1/2/4`, four non-consecutive logical blocks, GQA, and
the complete Stage-1 workspace to Stage-2 merge path.
