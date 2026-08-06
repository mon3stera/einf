# einf Checkpoint

Updated: 2026-08-05

## Current phase

Gate 2, Gate 3, and Gate 4 are happy-path / functional complete. Gate 5.0A
naive contiguous Attention and Gate 5.1 FlashAttention-style forward are
correctness-complete. Gate 5.2 single-request Paged Decode is also
correctness-complete, including four-Warp Context partitioning and partial
online-softmax-state merge; batched/model integration is next.

The repository has accumulated uncommitted Gate 4 and Gate 5-entry changes; do
not reset or overwrite the working tree. No commit has been requested yet.

## Gate 4 completion evidence

- token-budget Continuous Batching, chunked prefill, persistent FCFS and
  DecodeFirst, preemption, deterministic recompute, and OOM termination;
- preallocated Torch K/V tensors indexed by physical block ID;
- packed `ModelInput` with positions, slot mapping, query boundaries, block
  tables, and context lengths;
- CUDA `einf::write_slots_` and `einf::gather_context` custom ops integrated
  into `TorchKVCacheStorage`;
- explicit PyTorch reference path through `use_custom_ops=False`;
- `ReferenceModelRunner` and packed `QwenModelRunner`;
- local Qwen2.5-0.5B checkpoint loading from
  `/home/wyg/python/models/Qwen2.5-0.5B`;
- full LLMServer→Scheduler→TorchExecutor chunked Prefill→Decode smoke test;
- generated tokens `[12095, 13, 1084, 374]`, exactly matching Hugging Face;
- Gate 5 Extension builds successfully and the full suite is at 78 passing
  tests;
- KV write/gather benchmark retained under `benchmarks/`.

## Gate 4 deferred work

- formal wheel/build packaging for the JIT Torch Extension;
- CPU custom kernels;
- broader malformed-metadata and asynchronous CUDA error hardening;
- complete capacity/fragmentation and high-concurrency Qwen benchmarks;
- optimization of `gather_context`—intentionally skipped because it is a
  transitional oracle removed by Paged Attention.

These items do not block Gate 5.

## Gate 5 entry point

Design document:

```text
docs/gate5-paged-attention.md
```

Scaffold:

```text
src/einf/executors/torch/ops/csrc/contiguous_attention.h
src/einf/executors/torch/ops/csrc/contiguous_attention.cpp
src/einf/executors/torch/ops/csrc/contiguous_attention_cuda.cu
src/einf/executors/torch/ops/csrc/flash_attention.h
src/einf/executors/torch/ops/csrc/flash_attention.cpp
src/einf/executors/torch/ops/csrc/flash_attention_cuda.cu
src/einf/executors/torch/ops/csrc/paged_attention.h
src/einf/executors/torch/ops/csrc/paged_attention.cpp
src/einf/executors/torch/ops/csrc/paged_attention_cuda.cu
```

Completed Gate 5.0A operator:

```text
einf::contiguous_attention(Q, K, V, start_pos, scale) -> output
```

It uses token-major Q `[q_len,Hq,D]`, K/V `[kv_len,Hkv,D]`, and output
`[q_len,Hq,D]`. The implementation deliberately separates QK, in-place stable
softmax, and PV with an FP32 score workspace. FP32 and BF16 parity tests cover
causal chunked Prefill and GQA. It is frozen as the native correctness oracle;
no performance optimization is planned.

Completed Gate 5.1 operator:

```text
einf::flash_attention(Q, K, V, start_pos, scale) -> output
```

Version 0 uses `BLOCK_M=4`, `BLOCK_N=16`, four warps per Query tile/head,
cooperative FP32 Q/K/V shared-memory loads, causal GQA score computation, and
per-row FP32 online-softmax/output state without a global score Tensor. It is
currently restricted to `head_dim=64`. FP32/BF16 parity covers full causal
Prefill (`19/19`) and chunked Prefill (`q_len=6`, `kv_len=21`, `start_pos=15`),
including partial Query/KV tiles and multiple K/V tiles. It is optionally
integrated through `QwenModelRunner(..., use_flash_attention=True)` and
`scripts/run_qwen.py --flash-attention`; eager Attention remains the default.
The real Qwen2.5-0.5B BF16 path still generates `[12095, 13, 1084, 374]`, exactly
matching Hugging Face.

`benchmarks/benchmark_attention.py` compares the Qwen eager body, the native
three-kernel oracle, and FlashAttention. On the RTX 4090 BF16 baseline,
FlashAttention wins small workloads (1.54–1.92x versus eager for Decode-128,
chunk-16/128, and Prefill-128) but loses larger workloads (2.0–4.9x slower for
Decode-512, chunk-64/512, and Prefill-512). The kernel is correctness-first,
serializes Keys inside each warp, and has no Tensor-Core path.

Completed Gate 5.2 Paged Decode operator:

```text
einf::paged_decode_attention(
    Q, K_cache, V_cache, block_table, context_len, scale
) -> output
```

Version 0 is CUDA-only, one request, and `q_len=1`. Its output shape matches Q:
`[num_attention_heads, head_dim]`. It directly reads physical KV blocks,
performs GQA head mapping and FP32 online softmax, and matches the independent
PyTorch gather/reference oracle. The launcher supports FP32/FP16/BF16 storage
and enforces `head_dim <= 256` divisible by 32.

Parity tests cover `context_len=1`, an exact two-block boundary, and a partial
three-block Context using non-consecutive physical blocks, with `Hq=4`,
`Hkv=2`, `D=64`, and FP32/FP16/BF16. The full suite is now 87 passing.

The original single-Warp V1 RTX 4090 BF16 baseline was:

```text
context   gather+qwen   gather+naive   paged-v1
128          71.658         25.959   88.709 us
512          70.391         82.483  348.645 us
2048         72.612        333.445 1250.120 us
8192         71.242       1185.225 4998.332 us
```

Paged Decode V2 uses four Warps per Query head. Each Warp processes a contiguous
logical-block range, stores local `(m,l,acc)` in Shared Memory, and Warp 0 merges
the states. `HEAD_DIM` is specialized at compile time for multiples of 32
through 256, dimension loops are unrolled, and partial Shared Memory is sized to
the selected dimension. Relative to the single-Warp V1, the initial four-Warp
version improved latency by 3.58–4.01x:

```text
context   V1 one-Warp   V2 four-Warp   V1/V2
128          88.709         24.461       3.63x
512         348.645         89.262       3.91x
2048       1250.120        349.112       3.58x
8192       4998.332       1247.478       4.01x
```

Compile-time `HEAD_DIM=64` specialization and merge cleanup add another
1.67–1.73x. A longer RTX 4090 BF16 run also forces PyTorch's Flash SDPA backend
for the gathered SDPA comparison:

```text
context   gather+qwen  gather+flash  gather+native   paged-v2
128          69.721        16.683          25.739      14.634 us
512          70.628        19.887          74.581      46.953 us
2048         71.424        19.857         298.684     181.196 us
8192         69.792        19.669        1190.035     721.886 us
```

Direct Paged Decode wins at Context 128, but forced Flash SDPA including gather
is 2.36x, 9.13x, and 36.70x faster at Context 512, 2048, and 8192. The remaining
gap is therefore not primarily transitional gather overhead. GQA-aware KV reuse,
tiled/vectorized execution, and eventually Tensor-Core use are more important
than another scalar address-mapping tweak. The operator is not yet integrated
into QwenAttention.

The current Vec2 version uses dtype-specific pair loads and FP32 `float2`
accumulators. Its RTX 4090 BF16 comparison is:

```text
context   gather+qwen  gather+flash  gather+native   paged-vec2
128          71.977        16.424          25.948        12.511 us
512          71.373        19.753          82.156        43.263 us
2048         72.437        19.610         333.720       165.588 us
8192         70.533        19.854        1210.692       588.666 us
```

Vec2 adds roughly 1.09–1.23x over the preceding template-specialized version.
Long-Context performance remains limited by the small number of Query-head CTAs
and repeated GQA KV reads rather than scalar-load width alone.

A separate two-stage learning operator is now correctness-complete without
changing the completed single-CTA baseline:

```text
einf::paged_decode_attention_split_kv(
    Q, K_cache, V_cache, block_table,
    context_len, num_splits, scale
) -> output
```

Stage 1 uses Grid `[Hq,num_splits]`, four Warps per CTA, balanced logical-block
partitioning at both CTA and Warp levels, and writes FP32 partial `(m,l,acc)`
states. Stage 2 uses one Warp/CTA per Query head to merge the Context splits and
normalize once. FP32/FP16/BF16 tests cover `num_splits=1/2/4`, four
non-consecutive logical blocks, GQA, and the complete two-stage path. The custom
op suite has 31 passing tests and the full suite has 96 passing tests.

The RTX 4090 BF16 Split-KV sweep uses 100 warmup and 1000 measured iterations:

```text
context   single-CTA   best-s   Split-KV   single/split   Flash/Split
128          12.467        2       9.482          1.31x          1.73x
512          38.712        8       9.462          4.09x          2.08x
2048        148.385       32      13.540         10.96x          1.45x
8192        588.717       64      24.946         23.60x          0.79x
```

Split-KV beats gathered forced Flash SDPA through Context 2048 and reaches 79%
of its throughput at Context 8192. The observed choices match approximately
`num_splits = min(64, num_logical_blocks / 4)` for these four power-of-two
Contexts, but this is only a first single-request/Qwen-geometry policy candidate.

An extended RTX 4090 BF16 sweep added configurable `--context-lengths` and found:

```text
context   best-s   Split-KV   gather+Flash   Flash/Split
16384         64      35.955          25.447          0.71
32768         64      58.044          45.613          0.79
65536         64     102.634          66.507          0.65
131072        64     193.253         220.624          1.14
262144       256     448.920         460.890          1.03
524288       256     849.480         925.870          1.09
```

The first four rows use 100/1000 warmup/measured iterations; 262K uses 50/500
and 524K uses 20/200. The Flash/Split gap is not monotonic: gathered Flash wins
through 64K, then Split-KV is roughly tied or slightly faster. The best Split
count eventually rises to 256, invalidating a universal 64-Split cap.

The learner owns the QK, softmax, PV, fused Decode, FlashAttention, and Paged
Attention kernel bodies, thread/block mappings, online-softmax state, and later
optimization decisions. AI may maintain schemas, launch scaffolding, parity
tests, benchmark fixtures, and code review.

## Commands

```bash
cd ~/python/aiinfra/projects/einf
source ~/python/.venv311/bin/activate
python -m pip install -e '.[dev,qwen]'
python -m einf.executors.torch.ops --verbose
python -m pytest
python scripts/run_qwen.py --max-new-len 4 --max-prefill-chunk-len 4 --compare-hf
python scripts/run_qwen.py --flash-attention --max-new-len 4 --max-prefill-chunk-len 4 --compare-hf
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_paged_decode_attention.py
```

## Next implementation step

Use profiler evidence to separate HBM/L2 behavior, Stage-2 workspace overhead,
and forced-SDPA backend thresholds, then test an adaptive Split-count dispatch
across other head geometries/Batch sizes before QwenAttention integration.
GQA-aware KV reuse remains the next larger redesign after that system boundary.
