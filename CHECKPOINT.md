# einf Checkpoint

Updated: 2026-08-05

## Current phase

Gate 2, Gate 3, and Gate 4 are happy-path / functional complete. Gate 5.0A
naive contiguous Attention and Gate 5.1 FlashAttention-style forward are
correctness-complete. Gate 5.2 single-request Paged Decode is also
correctness-complete, including four-Warp Context partitioning and partial
online-softmax-state merge. Single-request Paged/Split-KV is integrated into
Qwen Decode; packed Batched Paged Decode is the current throughput step.

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
src/einf/executors/torch/ops/csrc/tensor_core_qk.h
src/einf/executors/torch/ops/csrc/tensor_core_qk.cpp
src/einf/executors/torch/ops/csrc/tensor_core_qk_cuda.cu
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

`benchmarks/benchmark_attention.py` compares the Qwen eager body, PyTorch forced
Flash SDPA with lower-right causal bias, and einf's native FlashAttention over
realistic full and chunked Prefill shapes. The current Sliced-Q experiment uses
`BLOCK_M=16`, `BLOCK_N=16`, eight Warps per CTA, two Query rows per Warp, and
`__launch_bounds__(256, 2)`. The CTA terminates its K/V loop after the final Key
visible to its last valid Query row, removing fully future causal tiles. Q/K/V
Shared storage is now static and dtype-specific, while QK/PV accumulation,
Online Softmax, and output accumulation remain FP32. The BF16 specialization
uses 6,144 bytes of static Shared Memory, 71 registers per thread, and no Local
Memory; FP32 storage uses 12,288 Shared bytes. A controlled static-Shared A/B is
effectively neutral for full Prefill: FP32-Shared
29.495/232.622/2244.137/8203.738 us versus BF16-Shared
29.225/233.527/2248.530/8177.222 us at 128/512/2048/4096. BF16 Shared improves
the fixed-`q_len=128` chunked cases by about 1.1%:
100.680/395.991/1576.100/6280.049 us becomes
99.604/391.455/1557.593/6207.220 us at Context 512/2048/8192/32768. All 99 tests
pass. Earlier Nsight occupancy figures were collected before static Shared
storage and now serve only as historical launch-bound evidence; a fresh report
is required for the current resource profile. Serial-Key CUDA-core QK/PV and the
missing matrix/Tensor-Core path remain the dominant limitations.

The next Prefill learning operator is registered as:

```text
einf::tensor_core_qk(Q, K) -> scores
```

It isolates BF16 Tensor Core QK before changing the fused Attention dataflow.
Q uses `[q_len,Hq,64]`, K uses `[kv_len,Hkv,64]`, and the result is FP32
`[q_len,Hq,kv_len]`. Version 0 requires Query and KV lengths divisible by 16 and
uses a `[128,16]` CTA score tile with eight Warps. Each Warp owns 16 Query rows
and accumulates four `m16n16k16` WMMA operations across D=64; the CTA shares the
GQA-mapped 16-row K tile. Cooperative BF16 Shared loads, GQA mapping, partial
final Query CTAs, FP32 score addressing, and WMMA stores are implemented.
Parity against an independent FP32 `torch.einsum` reference covers
`(q_len,kv_len,Hq,Hkv)` values `(16,16,4,2)`, `(128,32,4,2)`,
`(144,64,14,2)`, and `(256,16,8,8)`, with maximum absolute error at most
`1.15e-5`. `cuobjdump` confirms `HMMA.16816.F32.BF16`; the kernel uses 40
registers per thread, 18,432 static Shared bytes, and no Local Memory. The full
suite is 103 passing. Tensor Core PV and fused Online Softmax integration remain
future steps.

CuTe learning now starts from a separate `einf::cute_copy(input) -> output`
operator rather than rewriting the completed WMMA QK path. CUTLASS is pinned as
the `third_party/cutlass` Git submodule at commit
`c506e16788cb08416a4a57e11a9067beeee29420`, matching the CUTLASS revision used
by PyTorch 2.10's pinned FlashAttention source. The JIT loader supplies its
`include/` directory to NVCC. The first CUDA scaffold constructs row-major CuTe
Global Tensors, divides them into `[128,64]` CTA tiles, and defines a row-major
`(8,32)` thread layout so adjacent lanes traverse adjacent columns. The first
exercise is correctness-complete: the learner-authored kernel uses
`local_partition`, a Register fragment from `make_tensor_like`, and two
`cute::copy` calls for Global-to-Register-to-Global movement. Exact parity covers
single and multi-tile row/column grids at `(128,64)`, `(256,64)`, `(128,128)`,
and `(256,128)`. The full suite is 107 passing. Global-to-Shared tiled copy is
the next completed CuTe exercise.

The second exercise is correctness-complete as
`einf::cute_shared_copy(input) -> output` with the same FP32 CUDA shape contract,
`[128,64]` CTA tile, and row-major `(8,32)` thread layout. Its CUDA scaffold
constructs a row-major `[128,64]` Shared Tensor with 32 KiB of static Shared
Memory. The learner-authored body partitions Global and Shared tiles, copies
Global-to-Shared, synchronizes the CTA, and copies Shared-to-Global. Exact parity
covers `(128,64)`, `(256,64)`, `(128,128)`, and `(256,128)`.

The third exercise is scaffolded as
`einf::cute_transpose(input) -> output` for FP32 CUDA matrices whose dimensions
are multiples of 64. Each CTA owns a `[64,64]` source tile and the swapped output
tile. The correctness baseline used two views over one 16 KiB allocation:
row-major stride `[64,1]` for Global-to-Shared writes and transposed stride
`[1,64]` for Shared-to-Global reads. This changes ownership across the CTA
barrier and makes synchronization required. The learner then completed the
isolated bank-conflict optimization: the two views now share 16.25 KiB
(`64*65` FP32 elements), with
write stride `[65,1]` and transposed-read stride `[1,65]`. For Warp lane `l`, the
transposed bank changes from `(base + l*64) % 32 = base` to
`(base + l*65) % 32 = (base + l) % 32`. The learner-authored kernel partitions the
source and row-major Shared view, copies Global-to-Shared, synchronizes the CTA,
then partitions the transposed Shared view and swapped destination tile for the
Shared-to-Global copy. Exact parity against `input.T.contiguous()` covers
`(64,64)`, `(128,64)`, `(64,128)`, and `(128,192)`. The full suite is 115
passing. RTX 4090 CUDA-event unpadded/padded timings were 3.615/3.478 us at 64,
3.864/3.352 us at 512, 19.282/10.369 us at 2048, and 146.668/145.601 us at
4096; the 4096 case is effectively unchanged despite the conflict removal.

The fourth CuTe exercise is scaffolded as
`einf::cute_mma_qk(Q, K) -> scores`. It accepts fixed BF16 CUDA tensors
`Q[16,16]` and `K[8,16]`, returns FP32 `scores[16,8]`, and has the mathematical
contract `Q.float() @ K.float().T`. The CUDA path uses one CTA with one Warp and
instantiates `MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>`, `make_tiled_mma`, and
the current Lane's `thread_mma` slice. Global row-major Q/K and score tensors,
device/architecture checks, output allocation, schema, Python/JIT wiring, and
the expected learning guard are complete. The learner owns A/B/C partitioning,
Register fragments, BF16 Global loads, FP32 accumulator clearing, the single
`cute::gemm`, and the Global score store. The extension compiles against the
pinned CUTLASS revision and the full suite is 116 passing.

Before completing the fixed-shape MMA exercise, the CuTe curriculum now inserts
three arbitrary-extent operators. `einf::cute_elementwise_add(X, Y)` accepts
equal-shape contiguous FP32 CUDA inputs with any `numel`, including zero. Its
CUDA scaffold constructs flattened Global Tensors and an identity-coordinate
Tensor; 256-element tiling, final-tile predication, ThreadLayout, ceil-div Grid,
zero-work handling, Kernel launch, and launch checking remain learner-owned.

`cute_elementwise_add` is now correctness-complete. Its 256-thread CTA owns one
256-element tile; matching data and identity-coordinate partitions expose each
logical index, `get<0>()` extracts the rank-one coordinate, and the predicate
suppresses invalid final-tile loads/stores. The launcher uses
`ceil_div(numel,256)`, skips `numel == 0`, launches on the current PyTorch CUDA
stream, and checks the launch. Exact parity covers scalar, empty, 1/31/32/33,
255/256/257/1000, multidimensional, and zero-dimension inputs. On RTX 4090 the
compiled kernel uses 12 registers per thread and no Shared, Local, or Stack
storage.

`einf::cute_reduce_sum(input)` is now correctness-complete for a one-CTA
arbitrary-length contract. It treats the entire flattened input as one CuTe CTA
tile and partitions it across a 256-thread layout, giving each thread a strided
Tensor of elements. Thread-local FP32 sums are merged with Warp shuffles, eight
Shared Warp partials, one CTA barrier, and a Warp-0 final reduction. Empty input
returns the zero-initialized scalar without launch. Parity covers scalar, empty,
1/31/32/33, 255/256/257/1000, one million elements, multidimensional input, and
zero-sized dimensions. The RTX 4090 kernel uses 44 registers per thread, 32
bytes of Shared Memory, and no Local or Stack storage. Multi-CTA Atomic and
two-stage reductions remain deferred.

`einf::cute_gemm(A, B)` accepts contiguous FP32 `A[M,K]` and `B[K,N]` with
`M`, `K`, and `N` divisible by four, including zero dimensions, and returns
`C[M,N]`. The current 256-thread SIMT kernel uses a `64x64x32` CTA tile, a
two-stage 32-KiB Shared pipeline, and 128-bit
`SM80_CP_ASYNC_CACHEALWAYS_ZFILL` copies. Logical Shared layouts retain Stage as
an extra mode while explicit strides keep each A/B stage physically contiguous;
the B MMA view reinterprets the same storage as logical `[N,K,Stage]`.
Identity-coordinate partitions predicate complete four-FP32 vectors, and final
C stores remain scalar-predicated. The Prologue loads Stage 0, each steady-state
iteration waits for the current stage, issues the next Global-to-Shared copy,
then overlaps that copy with Shared-to-register movement and the current
`UniversalFMA<float>` GEMM. Parity covers exact and partial CTA tiles, one and
multiple K tiles, `68x20x76`, zero K, and empty M/N outputs; misaligned partial
vectors are rejected explicitly. The existing `cute_mma_qk` scaffold remains
paused while this GEMM advances through measured pipeline and Tensor Core
experiments. The RTX 4090 build uses 64 registers per thread, 32,768 bytes of
Shared Memory, and no Local or Stack storage; SASS contains 128-bit `LDGSTS`
instructions for the asynchronous copies.

`benchmarks/benchmark_cute_gemm.py` now establishes pinned CUTLASS as the GEMM
performance golden. It CUDA-event-times the current row-major FP32 `cute_gemm`,
then invokes `cutlass_profiler` on the memory-equivalent transposed NN
column-major problem and selects the fastest compiled kernel per shape. CUTLASS
FP32 SIMT is the arithmetic-matched golden; TF32 Tensor Core is labeled
separately as a non-IEEE-FP32 performance ceiling. Both paths reuse one
workspace. `scripts/build_cutlass_profiler.sh` builds only the required SM89
kernel families into `/tmp/einf-cutlass-build` from the pinned CUTLASS submodule.

The current RTX 4090 asynchronous-pipeline run used 20 warmups and 100 measured
iterations. At square 128/256/512/1024/2048, `cute_gemm` measured
7.065/12.268/22.695/80.919/604.627 us and
0.59/2.74/11.83/26.54/28.41 TFLOP/s. The best CUTLASS FP32 SIMT kernels measured
7.752/13.179/24.095/50.678/273.203 us, so the current kernel reaches
109.7%/107.4%/106.2%/62.6%/45.2% of the matched golden. Relative to the prior
16x16 educational baseline, the larger pipelined kernel regresses 128/256 by
1.69x/1.41x because its small CTA grid exposes launch and occupancy costs, but
improves 512/1024/2048 by 2.02x/4.11x/4.01x. The non-square
128x512x512/512x2048x512/512x512x2048 cases reach
113.4%/64.7%/95.0% of CUTLASS SIMT, and the partial-CTA 508x508x516 case reaches
100.7%. CUTLASS TF32 Tensor Core reaches 29.42/72.22/88.70 TFLOP/s at square
512/1024/2048, while remaining a different numerical contract.

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

A minimal end-to-end Qwen2.5-0.5B benchmark now measures one request through
`Scheduler -> TorchExecutor -> QwenModelRunner`. On RTX 4090 BF16 with full
Prefill and 32 generated tokens, gathered eager Attention sustains about
104–106 Decode tokens/s with 9.4–9.6 ms TPOT across prompt lengths 16–2048.
TTFT is 10.309/11.060/11.715/70.552 ms for prompts 16/128/512/2048. The native
learning Flash path reaches about 153 tokens/s at prompts 16–128, then falls to
137 tokens/s at 512 and 64 tokens/s at 2048 because of serial Key traversal.
Using 128-token Prefill chunks raises eager TTFT to 44.205 ms at prompt 512 and
176.317 ms at prompt 2048. Model startup is about 2.3 seconds and allocated CUDA
memory is about 976 MiB for the measured 2048+32-token cache geometry.

QwenAttention now optionally routes `q_len == 1` directly through Paged Decode.
The adaptive path uses the single-CTA kernel for one Split and the two-stage
Split-KV kernel otherwise, capped at 64 Splits; `q_len > 1` retains the gathered
Prefill backend. FP32 and BF16 regression coverage compares cached eager and
Paged Decode logits.

On RTX 4090 BF16 with full Prefill and 32 outputs, Split-KV sustains about
156–159 Decode tokens/s and 6.29–6.39 ms TPOT for prompts 16–2048, versus
104–106 tokens/s and 9.4–9.6 ms for gathered eager Decode. With 128-token
Prefill chunks and eight outputs, prompt lengths 4096/8192/16384/32768 produce
Split-KV TPOT 6.423/6.484/6.573/6.561 ms, versus gathered eager
10.169/10.122/10.192/14.089 ms. The corresponding Decode speedups are
1.58x/1.56x/1.55x/2.15x. TTFT remains dominated by the unchanged Prefill path.

The checkpoint model declares `max_position_embeddings=32768`; a synthetic
131K Prompt is not a representative end-to-end request. Long-cache operator
behavior remains covered by the standalone Paged benchmark.

Exact BF16 greedy-token parity is not guaranteed across eager and Paged
Attention. A real-model smoke test diverged on a near tie where the Paged path
gave two candidate logits 16.75; final-logit maximum absolute difference was
0.249. FP32/BF16 regression tolerances cover the numerical contract.

A closed-loop steady-state benchmark now maintains fixed concurrency with
deterministic random Prompt tokens, Prompt lengths 128/512/2048, output lengths
32/64/128, chunked Prefill 128, and batch token budget 512. At concurrency 8,
Paged/Split-KV produces 425.99 output tokens/s and 5.640 requests/s versus
252.46 tokens/s and 3.343 requests/s for gathered eager Decode, both over the
same 3021-token/402-step seeded workload. Paged inter-token p50/p95 is
15.466/22.969 ms versus eager 29.448/31.124 ms; TTFT p50/p95 is
78.197/373.467 ms versus 119.683/498.620 ms. Mixed Prefill/Decode batches are
44.0% of measured steps.

Paged throughput at concurrency 1/4/8/16 is 140.26/343.67/425.99/514.70 output
tokens/s, while inter-token p50 grows 6.429/10.138/15.466/29.247 ms. The current
Qwen path launches one Paged operator per Decode request per layer, so request
parallelism improves throughput but accumulates Python/operator-launch overhead.

vLLM 0.26.0 is installed separately at `~/python/.venv-vllm` as the
production-grade golden baseline. It uses its bundled PyTorch 2.11.0+cu130 and
keeps einf's PyTorch 2.10/CUDA 12.8 environment unchanged. The server requires
`VLLM_USE_FLASHINFER_SAMPLER=0` on this machine because the optional FlashInfer
sampling JIT is incompatible with the available toolkit headers; vLLM still
selects its FlashAttention-2 Attention backend.

A matched RTX 4090 BF16 workload fixes Prompt/output lengths at 512/64, ignores
EOS, uses greedy sampling, sets the batch token budget to 512, and sweeps
concurrency 1/4/8/16. einf Paged/Split-KV produces
139.86/363.06/473.37/571.57 output tokens/s, while vLLM produces
483.63/1858.98/3519.50/5301.21 tokens/s. The vLLM/einf throughput ratio is
3.46x/5.12x/7.44x/9.27x. Full commands and latency results are in
`benchmarks/vllm_golden.md`.

A separate packed Batched Paged Decode learning scaffold is registered as:

```text
einf::paged_decode_attention_batched(
    Q, K_cache, V_cache, block_tables, context_lens,
    query_start_loc, single_query_request_indices, scale
) -> decode_output
```

It accepts packed Q `[T,Hq,D]`, complete request metadata, and a compact list of
the requests whose `q_len == 1`; output is compact `[B_decode,Hq,D]`. The CUDA
launcher uses Grid `[Hq,B_decode]` and four Warps per CTA. Schema, validation,
Python wrapper, JIT source wiring, launcher dispatch for `HEAD_DIM=32..256`, and
registration/scaffold coverage are complete. The learner-owned kernel body must
map `blockIdx.y -> request_idx -> query_token_idx`, select the request's block
table and Context length, reuse the current Vec2 Paged Decode body, and write the
compact output row.

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
python scripts/run_qwen.py --paged-decode-attention --max-new-len 4
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_paged_decode_attention.py
python benchmarks/benchmark_qwen_e2e.py --paged-decode-attention
python benchmarks/benchmark_qwen_serving.py --decode-backend paged
```

## Next implementation step

Implement the packed request/query address mapping and reuse the completed
four-Warp Vec2 Paged Decode body in `paged_attention_batched_cuda.cu`. Then add
`single_query_request_indices` to `ModelInput`, scatter compact Decode output
once per layer, retain gathered eager Prefill for `q_len > 1`, and rerun the
concurrency sweep against the vLLM golden baseline.
