# FlashInfer decode optimization ladder (2026-08-29)

Same-host record for the Qwen serving path after Gate 6. Do not cite
`vllm_golden.md` as this machine: that file is the previous (failed) 4090,
vLLM 0.26.0. `post-migration-2026-08-23.md` is this host but an older session
(vLLM 3421 tok/s); treat it as historical, not today's A/B.

## Environment

```text
host:        zzx-System-Product-Name, RTX 4090, driver 550.163.01, 24564 MiB
python:      Anaconda 3.12 + /home/zzx/anaconda3
torch:       2.10.0+cu128
CUDA:        toolkit 12.1 on PATH, runtime 12.8
FlashInfer:  0.6.4
einf:        /media/zzx/新加卷2/aiinfra/projects/einf
model:       /media/zzx/新加卷2/models/Qwen2.5-0.5B, BF16
vLLM:        0.17.0 (same Anaconda), FLASH_ATTN v2, VLLM_USE_FLASHINFER_SAMPLER=0
```

## Matched workload

```text
concurrency:           8
prompt / output:       512 / 64
token budget:          512 (einf max_batch_len / vLLM max-num-batched-tokens)
prefill chunk:         128 (einf)
EOS:                   ignored
sampling:              greedy (temperature = 0)
einf:                  8 warmup + 32 measured completions, closed loop, seed 0
vLLM:                  one warmup (8 prompts), then 40 formal requests
```

Harness difference: vLLM is HTTP `bench serve` (open loop); einf is a closed
loop. Directional comparison only.

## vLLM on this host, this day

Command and log:
`benchmark-results/vllm-this-host-20260829-194617.log`.

```text
Output token throughput:  3204.70 tok/s
Median ITL:               1.98 ms
Median TPOT:              2.09 ms
Median TTFT:              23.30 ms
```

Server: `max-num-seqs 8`, `max-num-batched-tokens 512`, `gpu_memory_utilization 0.8`,
`torch.compile` + FULL_AND_PIECEWISE CUDA graphs.

Immediately after stopping vLLM, einf FlashInfer+graph+fused+greedy was
1865 / 1889 tok/s (two runs). Later rungs below were the same day on the same
box, not interleaved with another vLLM run.

## Serving ladder (c=8, 512/64)

Each rung stacks on the previous. tok/s and ITL p50 are
`benchmarks/benchmark_qwen_serving.py --decode-backend flashinfer`.

| rung | tok/s | ITL p50 | vs previous | vs today's vLLM |
| --- | ---: | ---: | ---: | ---: |
| FlashInfer, no graph | 1030 | 6.88 ms | — | 3.11x |
| Whole-step decode CUDA Graph | 1322 | 4.78 ms | +28% | 2.42x |
| Fused RMSNorm / SiLU-mul / RoPE | 1684 | 3.80 ms | +27% | 1.90x |
| All-greedy `argmax` sampler | 1875 | 3.28 ms | +11% | 1.71x |
| Packed QKV GEMM | 1969 | 3.14 ms | +5% | 1.63x |
| `BatchDecode` wrapper (tensor-core) | 2015 | 3.01 ms | +2% | 1.59x |
| Pack `BatchPlan` into graph buffers | **2070** | **2.84 ms** | +3% | **1.55x** |

vLLM on this host today: **3205 tok/s / 1.98 ms ITL**.

What each rung actually changed:

1. **Whole-step CUDA Graph** (`decode_graph.py`). Capture embed → 24 layers →
   lm_head per batch bucket `{1..8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64}`.
   FlashInfer `plan()` stays outside the graph. Mixed prefill/decode does not
   enter the graph (~18% of steps).
2. **FlashInfer fused elementwise** (`fused_ops.py`). `fused_add_rmsnorm`,
   `silu_and_mul`, `apply_rope_pos_ids_inplace`. CUDA FP16/BF16 only; CPU/FP32
   keep the PyTorch fallback. Qwen MLP packs `gate_up_weight`.
3. **All-greedy sampler**. Serving default is temperature=0; skip softmax /
   top-k / Python per-request loop and return `logits.argmax(-1)`.
4. **Packed QKV** (`packed_qkv.py`). One `Linear(H, q+k+v)` then split. HF
   checkpoints still load as `q_proj` / `k_proj` / `v_proj`. Decode GEMMs per
   layer: 6 → 4 (QKV, O, gate_up, down). Profiler CUTLASS: ~144 → ~96 / step.
5. **`BatchDecodeWithPagedKVCacheWrapper`**. Decode graphs no longer borrow
   ragged `BatchPrefill`. `use_cuda_graph=True`, `use_tensor_cores=True`,
   `pos_encoding_mode="NONE"` (RoPE is outside). Mixed still uses Prefill.
6. **Direct pack into graph buffers**. Decode skips `ModelInput.from_plan`
   (13× `torch.tensor` H2D). Python lists go into pinned staging, then five
   `copy_` into the captured tensors. Mixed still uses `from_plan`.

## Decode-only sync breakdown (c=8)

`profile_step_breakdown.py --timing sync --warmup-steps 80 --steps 200`.
165 decode-only steps / 35 mixed.

| phase | after graph (morning) | after pack_plan (now) |
| --- | ---: | ---: |
| input_build | 0.27 ms | **0.13 ms** |
| forward (includes plan) | 3.65 ms | **2.23 ms** |
|   plan (nested) | 0.32 ms | 0.31 ms |
|   graph_replay CPU | 0.01 ms | 0.01 ms |
| sampler | 0.51 ms | **0.05 ms** |
| sampler_meta_h2d | 0.10 ms | 0.16 ms |
| exec_residual | 0.24 ms | 0.19 ms |
| **decode step** | **4.96 ms** | **2.92 ms** |

Mixed steps are still eager: ~7.2 ms, input_build 0.90 ms (13 H2D allocations),
forward ~5.4 ms. That 18% mix is why serving ITL (2.84 ms) is not equal to
decode-only step (2.92 ms p50 / 2.92 mean) — ITL also includes TTFT-adjacent
and mixed interference on the closed loop.

## Remaining gap to this-host vLLM (~1.55x)

Not in ranked-implementation order, just where the milliseconds are:

1. **Decode GPU forward ~2.2 ms vs vLLM whole-step ITL 1.98 ms.** The graph
   already launches once. Inside: ~96 CUTLASS GEMMs, FlashInfer attention,
   leftover elementwise. Packed QKV and fused RMS/SiLU/RoPE already landed.
   Further kernel fusion or a decode-shaped GEMM is the GPU side.
2. **FlashInfer `plan()` ~0.31 ms, every decode step.** Still rebuilds CSR
   (`empty` / `cumsum` / boolean index) then `wrapper.plan()`. Next: write
   CSR into the static `indptr` / `indices` / `last_page_len` the graph
   already owns; skip a full plan when only `last_page_len` changes.
3. **Host leftovers ~0.5 ms.** `input_build` 0.13 ms (Python loops over page
   tables), sampler metadata H2D 0.16 ms, exec_residual 0.19 ms. Persistent
   sampling-parameter buffers and less Python glue.
4. **Mixed prefill, 18% of steps, ~7 ms.** Token-budget `BatchPrefill` CUDA
   graphs (pad packed_len to 128/256/384/512) are the single-GPU option.
   Prefill–Decode disaggregation is a multi-GPU architecture, not a patch
   for this 1×4090 run.

## Commands

```bash
# vLLM (this host)
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.cli.main serve \
  /media/zzx/新加卷2/models/Qwen2.5-0.5B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 \
  --max-num-batched-tokens 512 --max-num-seqs 8 --gpu-memory-utilization 0.8

VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --model /media/zzx/新加卷2/models/Qwen2.5-0.5B \
  --host 127.0.0.1 --port 8000 --dataset-name random --num-prompts 40 \
  --random-input-len 512 --random-output-len 64 --request-rate inf \
  --max-concurrency 8 --ignore-eos --temperature 0 --seed 0

# einf
python benchmarks/benchmark_qwen_serving.py \
  --model-dir /media/zzx/新加卷2/models/Qwen2.5-0.5B \
  --decode-backend flashinfer --concurrency 8 \
  --prompt-lens 512 --output-lens 64 \
  --max-prefill-chunk-len 128 --max-batch-len 512 \
  --warmup-completions 8 --measured-completions 32 --seed 0

python benchmarks/profile_step_breakdown.py \
  --model-dir /media/zzx/新加卷2/models/Qwen2.5-0.5B \
  --decode-backend flashinfer --concurrency 8 \
  --prompt-lens 512 --output-lens 64 \
  --warmup-steps 80 --steps 200 --timing sync
```

Related logs on this host: `benchmark-results/vllm-this-host-20260829-194617.log`,
`benchmark-results/einf-after-vllm-20260829-194657.log`. Morning graph-only
breakdown: `step-breakdown-flashinfer-graph-2026-08-29.md`.
