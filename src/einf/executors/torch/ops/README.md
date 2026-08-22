# Torch Custom Ops

This package is the boundary between einf's Python control plane and its
C++/CUDA data plane.

## Current scope

The extension registers seven CUDA-focused operators:

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

The separate Tensor Core learning operator isolates only the QK matrix
multiplication:

```text
einf::tensor_core_qk(
    Q,  # BF16 [q_len,Hq,64]
    K,  # BF16 [kv_len,Hkv,64]
) -> scores  # FP32 [q_len,Hq,kv_len]
```

Version 0 requires `q_len` and `kv_len` to be positive multiples of 16. It uses
`BLOCK_M=128`, `BLOCK_N=16`, and eight Warps: each Warp owns 16 Query rows and
computes one `[16,16]` score subtile through four `m16n16k16` BF16 WMMA
operations with FP32 accumulation.

The first CuTe exercise is registered separately:

```text
einf::cute_copy(
    input,  # FP32 CUDA [rows,cols]
) -> output
```

Version 0 requires `rows % 128 == 0` and `cols % 64 == 0`. Each CTA owns one
`[128,64]` row-major tile and uses a row-major `(8,32)` CuTe thread layout so
adjacent lanes traverse adjacent columns. The completed first exercise uses
`local_partition()`, a Register fragment, and Global-to-Register-to-Global
`cute::copy()`.

The second CuTe exercise is:

```text
einf::cute_shared_copy(
    input,  # FP32 CUDA [rows,cols]
) -> output
```

It preserves the same shape contract and CTA/thread geometry while adding a
row-major `[128,64]` Shared Tensor. The completed second exercise uses
Global-to-Shared partition/copy, a CTA barrier, and Shared-to-Global
partition/copy. Tails and vectorized `Copy_Atom` remain deferred.

The third CuTe exercise is:

```text
einf::cute_transpose(
    input,  # FP32 CUDA [rows,cols]
) -> output  # FP32 CUDA [cols,rows]
```

Version 0 requires both dimensions to be positive multiples of 64. Each CTA
owns one `[64,64]` source tile, writes it through a row-major Shared view, and
reads the same storage through a transposed Shared view before writing the
swapped output tile. The completed version pads the physical Shared row stride
from 64 to 65: the write view uses `[65,1]`, the transposed read view uses
`[1,65]`, and the allocation contains `64*65` FP32 elements. This changes the
transposed Warp bank mapping from one shared bank to all 32 banks.

The next three CuTe exercises introduce arbitrary logical extents and keep both
the Kernel body and CUDA launch learner-owned:

```text
einf::cute_elementwise_add(X, Y) -> output
einf::cute_reduce_sum(input) -> scalar
einf::cute_gemm(A, B) -> C
```

`cute_elementwise_add` is correctness-complete. It flattens equal-shape FP32
CUDA inputs, uses 256-element CTA tiles and a 256-thread one-dimensional layout,
partitions an identity-coordinate Tensor beside the data, extracts the rank-one
coordinate with `get<0>()`, and predicates the final partial tile. The launcher
uses ceil-div Grid construction, skips zero-element launches, uses the current
PyTorch CUDA stream, and checks the launch. Exact parity covers scalar, empty,
boundary, multi-CTA, multidimensional, and zero-dimension shapes.

`cute_reduce_sum` is correctness-complete for its one-CTA contract. It treats
the whole flattened input as one logical CTA tile, partitions it over a
256-thread layout so each thread owns a strided Tensor, performs thread-local
FP32 accumulation, Warp-shuffle reduction, eight Shared Warp partials, and a
final Warp-0 reduction. Empty input returns the pre-zeroed scalar without a
launch. This is arbitrary-length but intentionally not a scalable multi-CTA
Reduction; Atomic and two-stage variants remain later exercises.

`cute_gemm` is correctness-complete for FP32 `A[M,K]` and `B[K,N]` with
arbitrary nonnegative `M/N/K`, returning `C[M,N]`. Version 0 uses a
`UniversalFMA<float>` TiledMMA over SIMT `[16,16,16]` tiles and 256 threads.
It stages A/B through Shared Memory, zero-fills M/N/K tail loads, preserves both
CTA barriers, predicates C stores, skips empty-output launches, and handles
`K=0` by writing zeros.

The following fixed-shape Tensor Core exercise is:

```text
einf::cute_mma_qk(
    Q,  # BF16 CUDA [16,16]
    K,  # BF16 CUDA [8,16]
) -> scores  # FP32 CUDA [16,8]
```

It launches one Warp and instantiates one
`SM80_16x8x16_F32BF16BF16F32_TN` CuTe MMA atom. The mathematical contract is
`scores = Q.float() @ K.float().T`. Global Tensor construction, fixed-shape
validation, architecture checking, FP32 output allocation, the MMA atom, and
the per-Lane `thread_mma` slice are scaffolded. Partitioning A/B/C, Register
fragments, `clear`, `gemm`, and the score store remain learner-authored.

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

The packed mixed-batch learning scaffold adds:

```text
einf::paged_decode_attention_batched(
    Q,                            # [T,Hq,D]
    K_cache,                      # [num_blocks,block_len,Hkv,D]
    V_cache,
    block_tables,                 # [R,max_num_blocks]
    context_lens,                 # [R]
    query_start_loc,              # [R+1]
    single_query_request_indices, # [B_decode]
    scale,
) -> decode_output                # [B_decode,Hq,D]
```

The launch Grid is `[Hq,B_decode]`: each CTA maps one compact single-Query slot
back to its packed request and Query row, then reuses the completed four-Warp
Vec2 single-request body. The first contract returns compact Decode rows; Python
will scatter them once into the packed output while Prefill fills the disjoint
multi-Query intervals. The CUDA body is intentionally left as a learning
scaffold.

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
- `csrc/cute_copy.cpp`: schema and first-exercise shape/dtype validation.
- `csrc/cute_copy_cuda.cu`: completed CuTe Global-to-Register-to-Global copy
  using `[128,64]` CTA tiles and a row-major `(8,32)` thread layout.
- `csrc/cute_shared_copy.cpp`: second-exercise schema and shape/dtype validation.
- `csrc/cute_shared_copy_cuda.cu`: completed CuTe Global-to-Shared-to-Global
  copy and CTA synchronization using a row-major `[128,64]` Shared Tensor.
- `csrc/cute_transpose.cpp`: third-exercise transpose schema and validation.
- `csrc/cute_transpose_cuda.cu`: completed tiled transpose using paired
  row-major/transposed CuTe views over one `[64,64]` Shared allocation.
- `csrc/cute_elementwise_add.{cpp,cu}`: correctness-complete arbitrary-numel
  FP32 element-wise addition with flattened Global/coordinate Tensors,
  final-tile predication, empty-launch handling, and current-stream launch.
- `csrc/cute_reduce_sum.{cpp,cu}`: correctness-complete arbitrary-numel FP32
  one-CTA scalar sum with CuTe whole-input/thread partitions, Warp shuffles,
  Shared Warp partials, and empty-input handling.
- `csrc/cute_gemm.{cpp,cu}`: correctness-complete arbitrary-shape FP32 SIMT
  GEMM using runtime CuTe Global layouts, `[16,16,16]` Shared tiling,
  `UniversalFMA<float>` TiledMMA, predicated tails, and current-stream launch.
- `csrc/cute_mma_qk.cpp`: fixed BF16 `[16,16]` Q and `[8,16]` K contract plus
  CUDA-only validation.
- `csrc/cute_mma_qk_cuda.cu`: one-Warp `m16n8k16` CuTe MMA scaffold with
  row-major Global tensors, FP32 output, and learner-owned fragments/GEMM.
- `csrc/flash_attention.cpp`: FlashAttention schema, shared contiguous-input
  validation, and unsupported CPU dispatcher.
- `csrc/flash_attention_cuda.cu`: working learner-authored tiled forward kernel
  plus CUDA launcher. Version 0 uses four Query rows per block, sixteen K/V rows
  per tile, `head_dim=64`, FP32 online-softmax/output accumulation, and no global
  score Tensor.
- `csrc/tensor_core_qk.cpp`: BF16-only QK schema and first-version shape/GQA
  validation.
- `csrc/tensor_core_qk_cuda.cu`: WMMA includes, `[128,16]` CTA and 16-Query-row
  Warp ownership, cooperative BF16 Shared loads, four D=16 WMMA steps, FP32
  score stores, and launch plumbing.
- `csrc/paged_attention.cpp`: Gate 5 schema, validation, and unsupported CPU
  dispatcher.
- `csrc/paged_attention_cuda.cu`: completed four-Warp Vec2 Paged Decode baseline.
- `csrc/paged_attention_split_kv.cpp`: separate Flash-Decoding-style schema,
  Split-count validation, and unsupported CPU dispatcher.
- `csrc/paged_attention_split_kv_cuda.cu`: completed learner-authored two-stage
  Split-KV kernels, FP32 workspace allocation, and launch plumbing.
- `csrc/paged_attention_batched.cpp`: packed mixed-batch schema, metadata
  validation, and unsupported CPU dispatcher.
- `csrc/paged_attention_batched_cuda.cu`: `[Hq,B_decode]` launcher plumbing and
  the learner-owned packed request/query mapping plus reused Vec2 kernel body.

## Build the scaffold

The repository pins CUTLASS/CuTe as a Git submodule. Initialize it once:

```bash
git submodule update --init third_party/cutlass
```

The JIT loader adds `third_party/cutlass/include` to NVCC's include path and
fails with an actionable error when `cute/tensor.hpp` is missing.

```bash
python -m einf.executors.torch.ops --verbose
```

The first invocation compiles into PyTorch's extension cache. Later invocations
reuse the compiled library until the source changes.

## CuTe learning sequence

The CuTe path remains separate from the completed WMMA QK operator and the
working SIMT FlashAttention oracle. Exercises advance in this order:

```text
cute_copy
→ cute_shared_copy
→ cute_transpose
→ cute_elementwise_add
→ cute_reduce_sum
→ cute_gemm
→ cute_mma_qk
→ D64 QK
→ register row softmax
→ online-softmax QK
→ Tensor Core PV / fused FlashAttention
```

`cute_copy`, `cute_shared_copy`, and `cute_transpose` have exact multi-tile
parity coverage. An isolated RTX 4090 CUDA-event A/B measured unpadded versus
stride-65 transpose at 3.615/3.478 us for 64 square, 3.864/3.352 us for 512,
19.282/10.369 us for 2048, and 146.668/145.601 us for 4096. Padding therefore
helps the cache-resident middle sizes substantially, while the 4096 case is
effectively unchanged and likely dominated by a different bottleneck.
`cute_elementwise_add` is correctness-complete with exact parity over `numel`
0/1/31/32/33/255/256/257/1000, scalar input, multidimensional flattening, and
zero-sized dimensions. The compiled RTX 4090 kernel uses 12 registers per thread
with no Shared, Local, or Stack storage. `cute_reduce_sum` is also
correctness-complete for a one-CTA arbitrary-length contract; parity covers
scalar, empty, Warp/CTA boundaries, one million elements, multidimensional
flattening, and zero-sized dimensions. It uses 44 registers per thread, 32 bytes
of Shared Memory, and no Local or Stack storage. `cute_gemm` is
correctness-complete for single- and multi-tile M/N/K shapes, partial M/N/K
tails, and zero-sized dimensions. Its RTX 4090 kernel uses 40 registers per
thread, 2,048 bytes of Shared Memory, and no Local or Stack storage.
`cute_mma_qk` is the next active exercise.

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

## Completed Tensor Core QK learning operator

The first Tensor Core step intentionally materializes FP32 scores instead of
combining WMMA, Online Softmax, and PV at once. Each active Warp owns exactly 16
Query rows, reuses one CTA-wide BF16 K tile, and accumulates D=64 through four
K-dimension fragments. Parity against `torch.einsum("qhd,khd->qhk", ...)`
covers a minimal tile, a full 128-row CTA, a partial final CTA, multiple KV
tiles, GQA, and multiple Query CTAs. Maximum absolute error is at most
`1.15e-5`. SASS contains `HMMA.16816.F32.BF16`; the compiled kernel uses 40
registers per thread, 18,432 bytes of static Shared Memory, and no Local Memory.
Tensor Core PV, larger K tiles, score-fragment Softmax, causal masking, and tail
support remain subsequent steps.

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

## Gate 5.3 packed Batched Decode scaffold

The next throughput step keeps Prefill and Decode as separate kernels over one
packed ModelInput. `single_query_request_indices` compacts only `q_len == 1`
requests, so the launch avoids empty Prefill CTAs without constructing padded Q.
The metadata should be built once on CPU in `ModelInput.from_batch()` and reused
by every model layer. Initial execution will use one four-Warp CTA per
`(single_query_request,query_head)`; existing Split-KV remains the small-batch,
long-Context path until a benchmark justifies variable per-request Batched
Split-KV.
