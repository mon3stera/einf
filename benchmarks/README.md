# Benchmarks

Benchmark code, workload definitions, environment metadata, and raw-sample
locations belong here. Benchmarks are added after the relevant control-plane
interface and workload stabilize; they are not production-performance claims.

## KV Cache custom ops

Compare the CUDA custom operators with the current PyTorch reference paths:

```bash
python benchmarks/benchmark_kv_cache_ops.py
```

The gather benchmark reports both the complete PyTorch block-table mapping path
and a precomputed-slot `index_select` path. Their difference estimates mapping
overhead separately from KV data movement.

## CuTe GEMM

Build the pinned CUTLASS profiler with only the FP32 SIMT and TF32 Tensor Core
kernel families needed by the GEMM benchmark:

```bash
scripts/build_cutlass_profiler.sh
```

Then compare the current FP32 `einf::cute_gemm` kernel with CUTLASS:

```bash
python benchmarks/benchmark_cute_gemm.py
```

`cute_gemm` consumes row-major `A[M,K]` and `B[K,N]`, requires `M`, `N`, and
`K` to be divisible by four, and produces row-major `C[M,N]`. CUTLASS
Profiler's generated kernels use column-major C, so the
benchmark runs the memory-equivalent transposed problem
`C^T[N,M] = B^T[N,K] @ A^T[K,M]` through CUTLASS NN column-major kernels. Both
paths reuse one input workspace so CUDA-event timings represent warm-cache
kernel execution rather than allocation or process startup.

CUTLASS FP32 SIMT is the arithmetic-matched golden: both paths use IEEE FP32
multiplication and accumulation. CUTLASS TF32 Tensor Core is reported separately
as an aspirational performance ceiling because its FP32 inputs are rounded to
TF32 before multiplication and are therefore not numerically equivalent to the
current kernel. The script profiles every compiled candidate in each family and
reports the fastest kernel per shape. It reads the selected CSV row's explicit
`cta_m/n/k`, `stages`, Warp shape, instruction shape, raster order, and swizzle
metadata instead of inferring the configuration only from the procedural kernel
name. It also reads the current `cute_gemm_cuda.cu` constants and groups shapes
by CUTLASS's selected FP32 SIMT configuration. This makes the output directly
usable when choosing new custom tile variants; matching the CTA tile alone does
not match CUTLASS's Warp decomposition, copy schedule, epilogue, or raster order.
Use `--quick` while iterating or `--shapes
128x512x512,512x2048x512` for a targeted MxNxK sweep.

### RTX 4090 FP32 two-stage `cp.async` baseline

Environment: PyTorch 2.10.0+cu128, CUDA 12.8, pinned CUTLASS
`c506e16788cb08416a4a57e11a9067beeee29420`, 20 warmup iterations, 100 measured
iterations, and one reused workspace:

```text
          MxNxK   cute_us  cute_TF  simt_us  simt_TF  simt_%  tf32_us  tf32_TF  tf32_%
    128x128x128     7.065     0.59    7.752     0.54   109.7%    3.779      1.11    53.5%
    256x256x256    12.268     2.74   13.179     2.55   107.4%    5.458      6.15    44.5%
    512x512x512    22.695    11.83   24.095    11.14   106.2%    9.124     29.42    40.2%
 1024x1024x1024    80.919    26.54   50.678    42.38    62.6%   29.737     72.22    36.7%
 2048x2048x2048   604.627    28.41  273.203    62.88    45.2%  193.679     88.70    32.0%
    128x512x512    21.179     3.17   24.013     2.79   113.4%    9.083      7.39    42.9%
   512x2048x512    41.842    25.66   27.075    39.66    64.7%   16.722     64.21    40.0%
   512x512x2048    84.353    12.73   80.108    13.40    95.0%   30.341     35.39    36.0%
    508x508x516    25.324    10.52   25.498    10.44   100.7%    9.779     27.23    38.6%
```

`simt_%` and `tf32_%` are current-kernel throughput as a percentage of the
corresponding CUTLASS result: `100 * cutlass_latency / cute_latency`. The
64x64x32 two-stage kernel matches or slightly exceeds the selected CUTLASS FP32
SIMT family through square 512 and on the partial-CTA 508x508x516 case. It reaches
62.6% and 45.2% of CUTLASS SIMT at square 1024 and 2048, substantially improving
the prior 15.2% and 11.4%. The larger CTA regresses the old 128/256 results by
1.69x/1.41x because those grids expose launch and occupancy costs, while square
512/1024/2048 improve by 2.02x/4.11x/4.01x. These gains combine the larger CTA,
per-thread register micro-tiles, vectorized movement, and asynchronous pipeline;
they should not be attributed to `cp.async` alone. The TF32 ceiling remains a
different numerical contract and reaches 88.70 TFLOP/s at square 2048 versus
28.41 TFLOP/s for the current IEEE-FP32 CUDA-Core kernel.

## Attention kernels

Compare the current Qwen eager Attention body, PyTorch's forced Flash SDPA
backend, and einf's correctness-first native FlashAttention kernel:

```bash
python benchmarks/benchmark_attention.py
```

The benchmark defaults to Qwen2.5-0.5B geometry (`Hq=14`, `Hkv=2`, `D=64`)
and BF16. Its Qwen eager measurement includes `repeat_kv` and causal-mask
construction because those operations are part of the current
`QwenAttention` implementation. PyTorch SDPA is forced through
`SDPBackend.FLASH_ATTENTION` and receives `causal_lower_right(q_len,kv_len)`, so
chunked Prefill uses the same tail-aligned causality as einf. Use `--quick`
while iterating on kernels and `--include-naive` only when the expensive
three-kernel oracle timing is needed.

### CuTe DSL versus production Flash SDPA

Use the dedicated repeated paired benchmark while developing the CuTe DSL
FlashAttention kernel:

```bash
python benchmarks/benchmark_cute_flash_attention.py --quick
```

The comparison target is PyTorch's production Flash Attention backend, forced with
`sdpa_kernel(SDPBackend.FLASH_ATTENTION)`. Inputs use `causal_lower_right` and
`enable_gqa=True`, matching the CuTe kernel's lower-right-causal GQA contract. A
profiler verification on the RTX environment reported
`aten::_scaled_dot_product_flash_attention`, `aten::_flash_attention_forward`, and
`pytorch_flash::flash_fwd_splitkv_*` kernels rather than a math fallback.

The default matrix covers aligned full-Prefill and chunked-Prefill shapes. Use a
targeted matrix with, for example, `--cases 64x512,128x2048`. The current CuTe
contract requires `q_len % 64 == 0`, `kv_len % 16 == 0`, BF16, and `D=64`.
The default geometry is `Hq=14,Hkv=2`, `block_m=64`, `block_n=16`, and four
Split-Q warps, so each warp owns 16 consecutive Q rows.

The benchmark compiles `_flash_attention_launch` explicitly with `cute.compile`,
passing `(block_m, block_n, head_dim, num_warps)` as `cutlass.Constexpr`
specialization arguments. The public wrapper currently selects the verified
default `(64, 16, 64, 4)`; direct launch callers can compile other
SM80-compatible configurations without editing the kernel. The m16n8k16
back-to-back GEMM path requires positive `block_m % 16 == 0`,
`block_n % 16 == 0`, and `head_dim % 16 == 0`. Split-Q additionally requires
`num_warps` in `{1,2,4,8}` and `(block_m / num_warps) % 16 == 0`. Each new
combination still requires correctness and resource/performance validation. The
benchmark exposes `--block-m`, `--block-n`, `--head-dim`, and `--num-warps`, and
records both `num_warps` and `m_per_warp` in CSV. It reuses one CuTe output
allocation and times only the compiled callable. Calling
the learning-oriented CuTe Python wrapper directly performs compile/dispatch work
on every invocation and is not a kernel latency measurement. CuTe compile/JIT,
correctness checks, and warmup are excluded. Each shape uses 20 interleaved warmups
followed by nine order-alternating paired rounds of 100 calls. The report includes
medians, ranges, every raw sample, optional CSV output, and
`paired_x = sdpa_us / cute_us`: greater than one favors CuTe and less than one
favors Flash SDPA. Store generated CSV files under the ignored
`benchmark-results/` directory, for example
`--csv benchmark-results/cute-fa-vs-sdpa-pre-splitq.csv`.

#### RTX 4090 pre-Split-Q baseline

Environment: PyTorch 2.10.0+cu128, CuTe DSL 4.4.1, CUDA 12.1 toolkit selected for
the process, RTX 4090, BF16, `Hq=14`, `Hkv=2`, `D=64`, 20 warmups, 100 calls per
sample, and nine same-session paired rounds. CuTe source SHA256 was
`18a71d1fec6562d7f62370833107ff11bd83cf2be5a9deb455c99ee68417ed7d`.

```text
              case   cute_us  sdpa_us  paired_x  cute_TF  sdpa_TF
        prefill-32     6.714    8.806     1.216     0.28     0.21
       prefill-128    14.394    8.966     0.594     2.06     3.30
       prefill-512    47.739   17.448     0.365     9.86    26.98
      prefill-2048   286.280  110.702     0.387    26.27    67.93
      prefill-4096   894.321  299.981     0.336    33.63   100.25
      chunk-32/512    41.433   14.706     0.306     1.37     3.87
     chunk-32/2048   153.999   14.428     0.094     1.51    16.16
     chunk-32/8192   604.232   26.489     0.044     1.55    35.40
    chunk-128/2048   153.836   16.835     0.109     5.92    54.08
```

Latency columns are medians. Effective TFLOP/s count QK and P@V over
mathematically visible lower-right-causal pairs; softmax work and tile padding are
excluded. All nine correctness comparisons passed before timing; the largest
observed absolute CuTe/SDPA output difference was `0.00390625`. CuTe wins only the
smallest 32-token Prefill case. The largest gaps are short-Q/long-KV workloads,
where production SDPA can dispatch Split-KV while the pre-Split-Q CuTe kernel gave
one serial KV scan to each `(Q tile, query head)` warp. This table remains the
historical pre-Split-Q baseline; rerun the same paired protocol for the four-warp
implementation rather than comparing timings across sessions.

### RTX 4090 BF16 baseline

Environment: PyTorch 2.10.0+cu128, CUDA 12.8, `Hq=14`, `Hkv=2`, `D=64`, 20
warmup iterations, and 100 measured iterations.

```text
              case    eager_us     sdpa_us   native_us   native_%
       prefill-128      66.724      12.134      29.225       41.5%
       prefill-512      75.284      17.713     233.527        7.6%
      prefill-2048    2052.710     108.788    2248.530        4.8%
      prefill-4096    8581.143     291.000    8177.222        3.6%
     chunk-128/512      66.918      19.081      99.604       19.2%
    chunk-128/2048      79.677      18.780     391.455        4.8%
    chunk-128/8192     439.478      43.477    1557.593        2.8%
   chunk-128/32768    2351.306     119.364    6207.220        1.9%
```

`native_%` is native throughput as a percentage of forced Flash SDPA throughput:
`100 * sdpa_latency / native_latency`. This version uses `BLOCK_M=16`,
`BLOCK_N=16`, eight Warps per CTA, two Query rows per Warp, and
`__launch_bounds__(256, 2)`. The earlier dynamic-FP32-Shared experiment reduced
registers from 142 to 128 per thread without local-memory spills. Relative to the
unconstrained Sliced-Q build, full Prefill 512/2048/4096 improved by
1.30x/1.34x/1.36x, while Prefill 128 and the chunked-Prefill cases regressed by
roughly 3–5%. The kernel then added CTA-wide causal K/V-tile skipping, improving
full Prefill 512/2048/4096 by another 1.38x/1.73x/1.88x while fixed-`q_len=128`
chunked Prefill changed by only about 2%.

Q/K/V Shared storage is now static and dtype-specific: BF16/FP16 instantiate
6,144 bytes per CTA and FP32 instantiates 12,288 bytes. Arithmetic, Online
Softmax statistics, and output accumulation remain FP32. `cuobjdump` reports 71
registers and no local-memory use for the BF16 specialization. A controlled
static-Shared A/B changed full Prefill 128/512/2048/4096 from
29.495/232.622/2244.137/8203.738 us with FP32 Shared to
29.225/233.527/2248.530/8177.222 us with BF16 Shared: effectively neutral. The
chunked cases improved consistently by about 1.1%, from
100.680/395.991/1576.100/6280.049 us to
99.604/391.455/1557.593/6207.220 us. The previous Nsight occupancy report
predates static Shared storage and should not be treated as the current resource
profile. Medium/large native Prefill still reaches only 3.6–7.6% of Flash SDPA
throughput; the next large step requires matrix/Tensor-Core execution rather
than additional scalar Shared-storage tuning.

### Nsight Compute

`profile_flash_attention.py` isolates one native FlashAttention shape so Nsight
Compute does not need to filter eager and SDPA kernels from the full benchmark:
The extension is built with CUDA `-lineinfo` so the Source page can map sampled
instructions back to the kernel.

```bash
python benchmarks/profile_flash_attention.py \
  --q-len 128 \
  --kv-len 32768 \
  --warmup 10
```

Profile the launch after the ten warmups:

```bash
ncu \
  --set basic \
  --kernel-name-base function \
  --kernel-name 'regex:.*flash_attention_forward_kernel.*' \
  --launch-skip 10 \
  --launch-count 1 \
  --export /tmp/einf-flash-basic \
  --force-overwrite \
  python benchmarks/profile_flash_attention.py \
    --q-len 128 \
    --kv-len 32768 \
    --warmup 10
```

After the basic pass, replace `--set basic` with `--set full` for scheduler,
memory, occupancy, instruction, and Warp-stall sections. GPU performance
counters must be enabled by the system administrator; otherwise `ncu` reports
`ERR_NVGPUCTRPERM`.

## Paged Decode Attention

Compare direct physical KV-block reads with the transitional gathered paths:

```bash
python benchmarks/benchmark_paged_decode_attention.py
```

The benchmark reports `gather_context + Qwen eager`, forced PyTorch Flash SDPA
for FP16/BF16, `gather_context +` the three-kernel native oracle, direct
single-CTA Paged Decode, and a Split-KV sweep. The default Split counts are
`1,2,4,8,16,32,64,128`; counts exceeding the current logical-block count are
skipped. It uses nonconsecutive physical blocks and defaults to the
Qwen2.5-0.5B BF16 geometry. FP32 uses automatic SDPA backend selection because
the fused Flash backend requires FP16 or BF16 inputs.

Use `--context-lengths` for longer sweeps, for example:

```bash
python benchmarks/benchmark_paged_decode_attention.py \
  --context-lengths 16384,32768,65536,131072 \
  --splits 16,32,64,128,256,512
```

### RTX 4090 BF16 baseline

Environment: PyTorch 2.10.0+cu128, CUDA 12.8, `Hq=14`, `Hkv=2`, `D=64`,
`block_len=16`, 100 warmup iterations, and 1000 measured iterations.

```text
 context   gather+qwen  gather+flash   gather+naive       paged   sdpa/paged
     128        72.423        16.436         25.835      12.467         1.32
     512        72.398        19.661         73.838      38.712         0.51
    2048        74.035        19.673        297.469     148.385         0.13
    8192        72.611        19.789       1184.951     588.717         0.03
```

The same run measured the complete two-stage Split-KV operator:

```text
context   paged   best-s   best-split   paged/best   flash/best
128      12.467       2        9.482        1.31         1.73
512      38.712       8        9.462        4.09         2.08
2048    148.385      32       13.540       10.96         1.45
8192    588.717      64       24.946       23.60         0.79
```

Split-KV removes the single-request CTA shortage and beats gathered forced
Flash SDPA through Context 2048. At Context 8192 it reaches about 79% of the
Flash path's throughput while retaining direct paged reads. The observed best
counts through Context 8192 suggest an initial candidate policy of roughly one
Split per four logical blocks, capped near 64 for this geometry.

### Extended-Context sweep

The same geometry with 100 warmup and 1000 measured iterations gives:

```text
context   best-s   Split-KV   gather+Flash   Flash/Split
16384         64      35.955          25.447          0.71
32768         64      58.044          45.613          0.79
65536         64     102.634          66.507          0.65
131072        64     193.253         220.624          1.14
```

Exploratory larger runs used 50/500 and 20/200 warmup/measured iterations:

```text
context   best-s   Split-KV   gather+Flash   Flash/Split
262144       256     448.920         460.890          1.03
524288       256     849.480         925.870          1.09
```

The gap does not widen monotonically. Gathered Flash remains 1.27–1.54x faster
through 64K, while Split-KV becomes roughly tied or slightly faster at 128K and
beyond. The optimal Split count also eventually moves from 64 to 256, so the
earlier cap is not a universal policy. Cache residency, kernel/backend thresholds,
workspace reduction, and GQA read reuse need profiler evidence before assigning
the crossover to one cause. Batch size, head geometry, and GPU architecture must
also participate in the eventual dispatch rule.

## Qwen2.5 end-to-end latency

Measure one request through `Scheduler -> TorchExecutor -> QwenModelRunner`,
excluding model loading and one warmup request per prompt shape:

```bash
python benchmarks/benchmark_qwen_e2e.py \
  --prompt-lens 16,128,512,2048 \
  --max-new-len 32 \
  --max-prefill-chunk-len 2048 \
  --max-batch-len 2048 \
  --paged-decode-attention
```

`TTFT` starts immediately before submit and includes scheduling, Prefill, and
first-token sampling. `TPOT` and Decode tokens/s cover the remaining generated
tokens. By default EOS is ignored so every repetition produces the requested
number of tokens. This is a single-request latency benchmark, not a concurrent
serving-throughput benchmark.

RTX 4090 BF16, Qwen2.5-0.5B, 32 output tokens, one warmup and three measured
requests per shape:

```text
backend         prompt   TTFT ms   TPOT ms   Decode tok/s   E2E tok/s
gather+eager        16     10.309      9.398         106.40      106.05
gather+eager       128     11.060      9.571         104.49      103.90
gather+eager       512     11.715      9.484         105.44      104.67
gather+eager      2048     70.552      9.601         104.15       86.90
split-kv             16     10.578      6.291         158.95      155.63
split-kv            128     11.257      6.371         156.97      153.29
split-kv            512     11.855      6.393         156.41      152.40
split-kv           2048     70.519      6.360         157.23      119.54
gather+native       16      6.832      6.529         153.17      152.95
gather+native      128      7.298      6.502         153.79      153.20
gather+native      512     16.304      7.278         137.40      132.26
gather+native     2048    151.909     15.683          63.76       50.11
```

The learning Flash kernel wins on short Contexts through lower launch and score
materialization overhead, then loses as its serial Key traversal dominates.
With eager Attention and a 128-token Prefill chunk, TTFT rises from 11.715 to
44.205 ms at prompt 512 and from 70.552 to 176.317 ms at prompt 2048. That is the
expected latency cost of preserving chunked-Prefill admission opportunities.

With `--paged-decode-attention`, `q_len == 1` reads physical cache blocks
directly. Short Contexts use the single-CTA Paged kernel; longer Contexts use the
adaptive Split-KV path with a maximum of 64 Splits. Prefill continues to use the
selected gathered eager/native backend.

With 128-token Prefill chunks and eight output tokens, the long-Context Decode
comparison is:

```text
prompt   eager TPOT   split TPOT   eager tok/s   split tok/s   speedup
4096         10.169        6.423         98.34        155.70      1.58x
8192         10.122        6.484         98.80        154.23      1.56x
16384        10.192        6.573         98.12        152.14      1.55x
32768        14.089        6.561         70.98        152.41      2.15x
```

TTFT is nearly unchanged because Prefill is intentionally still gathered eager
Attention. The model configuration declares a 32768-token maximum context;
synthetic 128K Prompt tests are therefore not representative model inference.
Very long cache-read behavior remains covered by the standalone Paged Decode
benchmarks. A serving benchmark should instead model repeated request arrivals,
chunked Prefill, and Decode-to-completion interleaving.

FP32 and BF16 logits pass backend-parity tolerances. Exact greedy token sequences
can still diverge when two BF16 logits are effectively tied; one real-model
smoke test differed at a step where the Paged path gave both candidate logits
16.75. This is a numerical backend difference rather than a cache/addressing
failure.

## Qwen2.5 steady-state serving

`benchmark_qwen_serving.py` maintains a closed-loop target concurrency. Requests
use deterministic random token IDs and sampled Prompt/output lengths; completed
requests are immediately replaced. This naturally creates mixed chunked-Prefill
and Decode batches without requiring meaningful model output:

```bash
python benchmarks/benchmark_qwen_serving.py \
  --decode-backend paged \
  --concurrency 8 \
  --prompt-lens 128,512,2048 \
  --output-lens 32,64,128 \
  --max-prefill-chunk-len 128 \
  --max-batch-len 512
```

The default benchmark warms eight completions and then collects latency samples
from 32 newly submitted requests. Aggregate throughput includes every token and
completion during the steady-state measurement window. EOS is disabled so eager
and Paged runs execute exactly the same seeded workload.

RTX 4090 BF16, concurrency 8:

```text
metric                         gathered eager     Paged/Split-KV
output tokens/s                        252.46             425.99
requests/s                               3.343              5.640
TTFT p50 / p95 ms               119.683 / 498.620   78.197 / 373.467
inter-token p50 / p95 ms         29.448 / 31.124    15.466 / 22.969
request latency p50 / p95 ms   1987.140 / 4081.012 1138.793 / 2516.119
step latency p50 / p95 ms         29.574 / 31.064    15.664 / 22.949
mixed Prefill/Decode fraction             0.440              0.440
```

Direct Paged Decode improves aggregate output throughput by 1.69x and request
throughput by 1.69x. Although Prefill remains eager, shorter Decode steps also
reduce queueing around Prefill, improving TTFT and request latency.

The Paged closed-loop throughput curve is:

```text
concurrency   output tok/s   inter-token p50 ms   TTFT p50 ms
1                   140.26                 6.429        45.032
4                   343.67                10.138        38.803
8                   425.99                15.466        78.197
16                  514.70                29.247       118.322
```

Throughput continues to rise through concurrency 16, but per-request latency
also rises. QwenAttention currently launches one Paged operator per Decode
request per layer, so a packed batched Paged Decode operator is the next
system-level throughput boundary.

## Decode-only fixed batch

`benchmark_decode_only.py` drains Prefill first, then holds the same
`concurrency` running requests so every measured step is decode-only. Completed
requests are not replaced. Use this for decode tokens/s; the closed-loop serving
number includes mixed Prefill.

```bash
python benchmarks/benchmark_decode_only.py \
  --decode-backend flashinfer \
  --concurrency 8 \
  --prompt-lens 512 \
  --warmup-steps 80 \
  --steps 200 \
  --max-prefill-chunk-len 128 \
  --max-batch-len 512

# same-host vLLM A/B (isolated subprocesses; prefill/TTFT excluded)
python benchmarks/benchmark_decode_only.py \
  --engine both \
  --decode-backend flashinfer \
  --concurrency 8 \
  --prompt-lens 512 \
  --warmup-steps 80 \
  --steps 200
```

vLLM decode throughput uses `first_token_ts → last_token_ts` from
`RequestOutput.metrics`, so the first generated token (prefill) is not counted.
Do not mix this with HTTP `bench serve` or `vllm_golden.md`.

## vLLM golden baseline

`vllm_golden.md` is the **previous** 4090 (now failed), vLLM 0.26.0. Do not
compare current einf numbers to it. The current host's FlashInfer decode
ladder and same-day vLLM 0.17.0 run are
[`flashinfer-decode-opt-2026-08-29.md`](flashinfer-decode-opt-2026-08-29.md).
Post-migration same-host numbers from 2026-08-23 (older session) remain in
[`post-migration-2026-08-23.md`](post-migration-2026-08-23.md).

The historical golden used an isolated vLLM 0.26.0 environment
and a matched fixed workload of 512 Prompt tokens, 64 output tokens, greedy
sampling, a 512-token batch budget, and concurrency 1/4/8/16.

On the RTX 4090, einf Paged/Split-KV reaches
139.86/363.06/473.37/571.57 output tokens/s, while vLLM reaches
483.63/1858.98/3519.50/5301.21 tokens/s. The gap grows from 3.46x at concurrency
1 to 9.27x at concurrency 16. This is a system-level upper target rather than a
kernel-only comparison: vLLM uses FlashAttention Prefill, batched Paged Decode,
compiled/fused model execution, and CUDA Graphs.
