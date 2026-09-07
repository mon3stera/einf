# Per-step decomposition after FlashInfer + whole-step CUDA Graph

> Snapshot of the **graph-only** rung. The same-day ladder (fused ops, greedy
> sampler, packed QKV, BatchDecode, pack-into-graph) and the vLLM measurement
> **on this host** are in [`flashinfer-decode-opt-2026-08-29.md`](flashinfer-decode-opt-2026-08-29.md).
> Do not use `vllm_golden.md` as this machine.

Closes the question "why is decode still ~2x vLLM after graphs". Tool:
`benchmarks/profile_step_breakdown.py` (now attributes `try_decode_cuda_graph`
to `forward`, and splits `plan` / `graph_replay` inside it).

```bash
python benchmarks/profile_step_breakdown.py --decode-backend flashinfer \
    --concurrency 8 --prompt-lens 512 --output-lens 64 \
    --warmup-steps 80 --steps 200 --profiler-steps 8 --timing wall
# and the same with --timing sync
```

RTX 4090, torch 2.10.0+cu128, Qwen2.5-0.5B BF16, FlashInfer 0.6.4, whole-step
decode CUDA Graph. Raw log:
`benchmark-results/step-breakdown-flashinfer-graph-20260829-184713.log`.

Serving context from the same day: c=8 512/64 FlashInfer+graph is 1322 tok/s,
ITL p50 4.78 ms. Historical vLLM on this host was ~3421 tok/s / ~2 ms ITL.

## Decode-only steps (165 of 200)

Mean milliseconds. `wall` is CPU occupancy (GPU work surfaces in the phase that
blocks). `sync` waits at each coarse phase, so it is the attribution to use.

| phase | wall ms | sync ms | sync share |
| --- | --- | --- | --- |
| sched | 0.041 | 0.041 | 0.8% |
| input_build | 0.266 | 0.269 | 5.4% |
| **forward (graph)** | 0.461 | **3.651** | **73.6%** |
| sampler | 0.470 | 0.511 | 10.3% |
| sampler_meta_h2d | 3.248 | 0.099 | 2.0% |
| d2h | 0.032 | 0.010 | 0.2% |
| exec_residual | 0.241 | 0.242 | 4.9% |
| sync_wait | 0.021 | 0.008 | 0.2% |
| apply_result | 0.119 | 0.118 | 2.4% |
| **step** | **4.908** | **4.958** | 100% |

Inside `forward` (not additive):

| nested | wall ms | sync ms |
| --- | --- | --- |
| FlashInfer `plan()` | 0.351 | 0.322 |
| `CUDAGraph.replay` CPU | 0.011 | 0.011 |

`graph_replay` is one `cudaGraphLaunch` per step. The 3.65 ms in sync-`forward`
is the captured GPU work, not launch overhead. Wall-mode `sampler_meta_h2d`
3.25 ms is the same GPU tail: the graph is async and the next `torch.tensor`
blocks on the stream.

Profiler window (8 steps, all graph-launched): 3.694 ms device time / step,
**75% device-busy** of the 4.91 ms step. Graphs flipped the old paged path
(42% busy, 10+ ms of `cudaStreamSynchronize`) into a device-bound decode step.

## What is inside the 3.65 ms GPU forward

Per step, from the profiler kernel rows:

| work | per step | device ms |
| --- | --- | --- |
| CUTLASS GEMM (QKV / O / MLP) | 121 + 48 launches | ~1.8 |
| unfused elementwise / reduce | several hundred | ~1.3 |
| FlashInfer attention (inside graph) | folded into graph | remainder of 3.65 |

24 layers × 5 GEMMs is 120 CUTLASS launches. They are *inside* the graph, so
they no longer pay Python launch gaps, but they still run as 120 separate
kernels plus a swarm of RMSNorm / residual / SiLU / RoPE elementwise kernels.
vLLM fuses add+RMS, SiLU-mul, and RoPE; that is the remaining GPU gap
(roughly 3.65 ms vs ~1.5–2 ms for a whole vLLM decode step).

## Host leftovers (~1.3 ms even after the graph)

These are additive on top of GPU time (sync mode):

| leftover | ms / step | what it is |
| --- | --- | --- |
| sampler | 0.51 | per-request `.item()` seed loop, top-k |
| plan | 0.32 | rebuild CSR + FlashInfer `plan()` every step |
| input_build | 0.27 | `torch.tensor` packing + H2D |
| exec_residual | 0.24 | Python `ExecutionResult` glue |
| sampler_meta_h2d | 0.10 | six sampling-parameter copies |
| sched + apply | 0.16 | Rust/PyO3, already cheap |

Profiler still sees **34 `aten::_local_scalar_dense` / step** and **42
`cudaStreamSynchronize` / step** — sampler + `is_decode_only` + dummy-page
checks. Not 1200, but not zero.

## Mixed prefill steps (35 of 200)

Not graphed. 9.5 ms/step, 78% forward. This is why serving ITL p95 stays ~10 ms
and why tok/s is worse than decode-only 1000/4.9×8 ≈ 1630. Closed-loop serving
at 1322 tok/s is this mix.

## Ranked remaining gap to vLLM

1. **Unfused decode GPU work (~3.65 ms, ~75% of the decode step).** Graphs
   already did their job (`cudaGraphLaunch` 0.01 ms). Next kernel-side wins
   are fused add-RMS, SiLU-mul, RoPE, and a decode-only FlashInfer
   `BatchDecode` wrapper — not more graphs.
2. **Sampler + metadata (~0.6 ms).** Gate 6.4: batched GPU sampling, no
   `.item()`.
3. **Persistent buffers + plan (~0.6 ms).** Gate 6.3: stop allocating CSR /
   `ModelInput` every step; write into the static tensors the graph already
   has.
4. **Mixed prefill (~18% of steps, 9.5 ms).** Separate problem; decode graphs
   cannot cover it.

Graphs moved the bottleneck from host sync to **the unfused 24-layer decode
GPU work, then sampling/plan packing**. That is the opposite of the 2026-08-23
paged breakdown, where device busy was 42% and `cudaStreamSynchronize` was
10.6 ms.
