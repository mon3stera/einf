# Gate 5: Attention Kernels and Paged Decode Attention

Status: current as of 2026-08-05.

Progress: Gate 5.0A naive contiguous Attention and Gate 5.1
FlashAttention-style contiguous forward are correctness-complete. Both match
PyTorch for FP32/BF16 causal Attention with GQA. Paged Decode Attention is the
next step. The FlashAttention operator is available as an optional
`QwenModelRunner` path while eager Attention remains the default.

## Goal

Learn and verify Attention in isolated stages before combining online softmax
with non-contiguous paged KV reads.

```text
5.0A naive contiguous Attention: QK → stable softmax → PV
5.0B fused contiguous q_len=1 Decode Attention
5.1 FlashAttention-style contiguous causal forward
5.2 Paged Decode Attention
5.3 batching, model integration, benchmarks, and profiling
```

The existing Paged Attention scaffold remains in the tree but is intentionally
paused until the contiguous kernels are understood and correct.

## Gate 5.0A operator

```text
einf::contiguous_attention(
    Q,          # [q_len, num_attention_heads, head_dim]
    K,          # [kv_len, num_kv_heads, head_dim]
    V,          # [kv_len, num_kv_heads, head_dim]
    start_pos,  # kv_len - q_len
    scale,
) -> output     # [q_len, num_attention_heads, head_dim]
```

The first implementation uses three kernels and one materialized FP32 score
Tensor:

```text
QK with causal visibility and GQA mapping
→ in-place row-wise stable softmax
→ probability × V
```

The causal rule is:

```text
absolute_query_position = start_pos + query_index
key_index <= absolute_query_position
```

The exact contract `start_pos + q_len == kv_len` covers full Prefill, chunked
Prefill, and Decode.

### Gate 5.0A learning order

1. FP32, `Hq=Hkv=1`, small `D`, `q_len=1`.
2. Multiple keys and stable softmax.
3. `q_len>1` with causal masking.
4. Multiple heads and GQA.
5. BF16 storage with FP32 QK/softmax/PV accumulation.

Each stage must match a PyTorch oracle before adding the next dimension. These
stages are now complete; `contiguous_attention` is frozen as a correctness
oracle rather than a performance path.

## Completed Gate 5.1: FlashAttention-style forward

The implementation keeps the same token-major Q/K/V and causal
`start_pos` contract, but tiles Q/K/V and merges row-wise FP32 max, denominator,
and output accumulators without materializing `[q_len,Hq,kv_len]` scores. The
first version remains single-request, forward-only, contiguous, causal, and has
no dropout or backward contract.

Operator:

```text
einf::flash_attention(
    Q,          # [q_len, Hq, D]
    K,          # [kv_len, Hkv, D]
    V,          # [kv_len, Hkv, D]
    start_pos,
    scale,
) -> output     # [q_len, Hq, D]
```

Scaffold:

```text
src/einf/executors/torch/ops/csrc/flash_attention.h
src/einf/executors/torch/ops/csrc/flash_attention.cpp
src/einf/executors/torch/ops/csrc/flash_attention_cuda.cu
```

Version 0 uses `BLOCK_M=4`, `BLOCK_N=16`, `head_dim=64`, and one CUDA block per
Query tile/query head. Four warps own four Query rows, cooperatively load FP32
Q/K/V tiles, maintain independent FP32 `m`, `l`, and distributed output
accumulators, apply the absolute-position causal mask, and normalize once after
all K/V tiles. It does not allocate or write a global score/probability Tensor.

Parity tests cover full causal Prefill (`q_len=kv_len=19`), chunked Prefill
(`q_len=6`, `kv_len=21`, `start_pos=15`), partial Query/KV tiles, GQA, FP32, and
BF16. An FP32 Qwen runner regression verifies the optional model path, and a
real Qwen2.5-0.5B BF16 chunked-Prefill/Decode run with
`--flash-attention` generates `[12095, 13, 1084, 374]`, exactly matching Hugging
Face. Broad head-dimension support and optimization remain deferred.

The RTX 4090 BF16 microbenchmark shows the expected correctness-first profile:
the fused kernel is 1.54–1.92x faster than the existing eager body for
Decode-128, chunk-16/128, and Prefill-128, but 2.0–4.9x slower for Decode-512,
chunk-64/512, and Prefill-512. Its serial per-warp Key loop and lack of Tensor
Cores dominate as the workload grows; these results are learning evidence, not
a production-performance claim.

## Completed Gate 5.2: Paged Decode correctness

Replace the transitional sequence:

```text
block table
→ gather contiguous K/V
→ PyTorch Attention
```

with a CUDA operator that reads physical KV blocks directly. The control plane,
`ModelInput`, block ownership, and KV write path remain unchanged.

## Paged correctness oracle

Gate 4 freezes the following path as the oracle:

```text
TorchKVCacheStorage.gather_context()
→ GQA repeat
→ scaled QK
→ causal visibility for q_len=1
→ FP32 softmax
→ weighted V
```

The Paged Attention result must match this path before performance work begins.

## Paged Decode operator

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

Version 0 intentionally supports only:

- CUDA;
- one request;
- Decode with `q_len=1`;
- contiguous Q and Cache tensors;
- FP32/FP16/BF16 storage with FP32 score/softmax accumulation;
- GQA where `num_attention_heads % num_kv_heads == 0`.
- `head_dim <= 256` and divisible by 32.

The newest token's K/V must already have been written, so `context_len` includes
the current Decode token. No causal mask is needed: every logical context
position from `0` through `context_len - 1` is visible.

## Core mapping

For logical context position `position`:

```text
logical_block = position / block_len
block_offset = position % block_len
physical_block = block_table[logical_block]
physical_slot = physical_block * block_len + block_offset
```

For query head `q_head`, GQA selects:

```text
kv_group_size = num_attention_heads / num_kv_heads
kv_head = q_head / kv_group_size
```

The kernel then performs a scaled QK reduction and FP32 online softmax while
accumulating V without creating a contiguous Context Tensor.

Version 2 uses four CUDA Warps per Query head. Each Warp owns a contiguous range
of logical KV blocks and maintains local FP32 `(m,l,acc)` state. The states are
written to Shared Memory and merged by Warp 0 before final normalization.

## Paged work packages

### 5.2 Single-request correctness — complete

- direct paged reads and online softmax are implemented;
- FP32/FP16/BF16 parity covers `context_len=1`, an exact block boundary, and a
  partial multi-block Context with non-contiguous physical blocks;
- GQA (`Hq=4`, `Hkv=2`) is covered;
- output matches the independent PyTorch gather/reference path.

The original single-Warp baseline remains the conceptual oracle. The four-Warp
mapping preserves parity while reducing Context-scan latency by 3.58–4.01x.

### 5.3 Split-KV Decode — correctness and benchmark complete

- Stage 1 launches `[Hq,num_splits]` CTAs; each CTA uses four Warps and writes
  one FP32 `(m,l,acc[D])` state;
- Stage 2 launches one Warp/CTA per Query head and merges all split states;
- FP32/FP16/BF16 parity covers `num_splits=1/2/4`, GQA, and nonconsecutive
  physical blocks;
- the benchmark sweeps `num_splits=1/2/4/8/16/32/64/128` and skips counts larger
  than the logical-block count.

RTX 4090 BF16 best results with 100 warmup and 1000 measured iterations:

```text
context   single-CTA   best-s   Split-KV   single/split   Flash/Split
128          12.467        2       9.482          1.31x          1.73x
512          38.712        8       9.462          4.09x          2.08x
2048        148.385       32      13.540         10.96x          1.45x
8192        588.717       64      24.946         23.60x          0.79x
```

The first candidate policy for this exact geometry is approximately one Split
per four logical blocks, capped at 64. It remains benchmark evidence rather than
a general dispatch contract.

An extended Context sweep shows that the cap is not universal:

```text
context   best-s   Split-KV   gather+Flash   Flash/Split
16384         64      35.955          25.447          0.71
32768         64      58.044          45.613          0.79
65536         64     102.634          66.507          0.65
131072        64     193.253         220.624          1.14
262144       256     448.920         460.890          1.03
524288       256     849.480         925.870          1.09
```

The first four rows use 100/1000 warmup/measured iterations; the final two are
exploratory 50/500 and 20/200 runs. Gathered Flash wins through 64K, while
Split-KV is roughly tied or slightly faster from 128K in these repeated-input
microbenchmarks. Nsight evidence is required before attributing the crossover to
cache residency, backend thresholds, or another mechanism.

### 5.4 Batched Decode — next

- accept packed Decode Q, padded block tables, and context lengths;
- preserve request isolation and packed output order;
- compare batched output with separate version 0 calls.

### 5.5 Model integration

- route Decode requests through Paged Attention;
- keep Prefill/chunked Prefill on the reference path initially;
- run real Qwen2.5-0.5B generation parity;
- remove `gather_context` from the production Decode path only after parity.

### 5.6 Performance evidence

- `benchmarks/benchmark_paged_decode_attention.py` benchmarks gathered Qwen,
  forced Flash SDPA, the native oracle, single-CTA Paged Decode, and Split-KV;
- RTX 4090 BF16 single-CTA baseline (`Hq=14`, `Hkv=2`, `D=64`,
  `block_len=16`) reports:

  ```text
  context   gather+qwen   gather+flash   gather+naive   single-CTA
  128          72.423         16.436          25.835        12.467 us
  512          72.398         19.661          73.838        38.712 us
  2048         74.035         19.673         297.469       148.385 us
  8192         72.611         19.789        1184.951       588.717 us
  ```

- benchmark against identical data and record the launch/workload conditions;
- record context length, head counts, head dimension, block length, dtype,
  warmup, repeats, latency, and numerical error;
- use Nsight Systems/Compute only after correctness and stable workloads;
- optimize only when the measurements identify a concrete bottleneck.

## Deferred

- Paged Prefill Attention;
- multi-GPU execution;
- CPU kernel;
- quantization;
- production-grade invalid block-table recovery;
- speculative decoding and prefix sharing.
