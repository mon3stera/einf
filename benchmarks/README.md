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

## Attention kernels

Compare the current Qwen eager Attention body, the three-kernel native oracle,
and the correctness-first FlashAttention kernel:

```bash
python benchmarks/benchmark_attention.py
```

The benchmark defaults to Qwen2.5-0.5B geometry (`Hq=14`, `Hkv=2`, `D=64`)
and BF16. Its Qwen eager measurement includes `repeat_kv` and causal-mask
construction because those operations are part of the current
`QwenAttention` implementation. Use `--quick` while iterating on kernels.

### RTX 4090 BF16 baseline

Environment: PyTorch 2.10.0+cu128, CUDA 12.8, `Hq=14`, `Hkv=2`, `D=64`, 20
warmup iterations, and 100 measured iterations.

```text
            case     qwen_us    naive_us    flash_us  qwen/flash  naive/flash
      decode-128      65.708      23.962      34.634        1.90         0.69
      decode-512      65.577      80.346     131.584        0.50         0.61
    chunk-16/128      66.435      58.212      34.550        1.92         1.68
    chunk-64/512      67.256     299.376     137.267        0.49         2.18
     prefill-128      66.621      97.407      43.303        1.54         2.25
     prefill-512      74.967     597.041     369.244        0.20         1.62
```

The fused learner kernel removes the global score Tensor and wins some smaller
workloads by avoiding eager launch/mask/repeat overhead. It becomes much slower
as Context and Query lengths grow because each warp serially processes Keys and
uses ordinary FP32 CUDA cores rather than Tensor Cores. These numbers are not a
production-performance claim.

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
