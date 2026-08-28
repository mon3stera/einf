# einf Roadmap

Updated: 2026-08-22

This document defines the order of work after the Rust/PyO3 control-plane
migration. It is deliberately opinionated: the next milestone is **closing the
measured system-level performance gap**, not adding features.

## 1. Where the project stands

Functional state:

- Gate 2/3/4 happy-path complete: token-budget continuous batching, chunked
  prefill, FCFS and DecodeFirst policies, preemption, deterministic recompute,
  OOM termination, preallocated block-indexed K/V, packed `ModelInput`.
- Gate 5 correctness-complete: naive contiguous attention, FlashAttention-style
  forward, single-request Paged/Split-KV decode integrated into Qwen decode.
- Control plane migrated to Rust: `crates/einf-control` (pure lib) plus
  `crates/einf-control-py` exposing native `einf._control`. Request lifecycle,
  block pool, KV cache, scheduler policies, prefix cache and sampling now live
  in Rust; Python keeps weights, storage, `ModelInput` and custom-op dispatch.
- Prefix Cache v0 implemented, scoped to sharing among *active* requests only.
- Torch sampler implemented on the Python side.

Kernel state (`src/einf/executors/torch/ops/csrc/`): a CuTe learning ladder
(copy, shared copy, transpose, elementwise add, reduce sum, GEMM, MMA QK,
tensor-core QK), plus `flash_attention`, `contiguous_attention`, paged decode,
`write_slots_` and `gather_context`. The packed batched paged-decode body
remains an intentionally paused scaffold.

Repository state: committed as `df93925` and `1d9b350`, mirrored to
`github.com/mon3stera/einf` (default branch `main`).

## 2. The measurement that sets the priority

Current baseline (`benchmarks/post-migration-2026-08-23.md`), measured after the
migration with both engines on the same host and the same PyTorch build —
prompt 512, output 64, token budget 512, EOS ignored, greedy:

| concurrency | einf tok/s | vLLM 0.17.0 tok/s | ratio | einf TTFT p50 | vLLM TTFT p50 | einf ITL p50 | vLLM ITL p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 125.66 | 430.92 | 3.43x | 38.78 | 11.75 | 6.93 | 2.08 |
| 4 | 302.76 | 1763.50 | 5.83x | 89.60 | 13.98 | 11.26 | 2.00 |
| 8 | 401.70 | 3421.02 | 8.52x | 111.26 | 18.51 | 18.21 | 1.99 |
| 16 | 477.91 | 5018.90 | 10.50x | 165.68 | 43.34 | 29.03 | 2.05 |

The absolute ratio matters less than the **slope**: vLLM's median inter-token
latency is flat over the whole range (2.08 → 2.05 ms) while einf's grows 4.19x
(6.93 → 29.03 ms); equivalently einf scales throughput 3.80x from concurrency 1
to 16 against vLLM's 11.65x. A decode step that costs proportionally more as the
batch grows means per-request work inside the step, which matches the known
structural facts — einf launches one paged-decode operator per request per
layer, rebuilds plan structures every step, samples per request, and captures
nothing in a CUDA Graph. Prefill is not the bottleneck: prefill throughput
scales 1005 → 3935 tok/s over the same range.

The paged decode path is 1.58x the gathered eager path at concurrency 8 (401.70
vs 254.22 tok/s), so the in-house kernel is a real win over the naive baseline;
the gap to vLLM is a batching and launch-overhead problem, not a kernel-quality
problem.

Regression note: einf lost 10-17% versus the pre-migration golden while vLLM
lost 3-11% on the same move, so roughly 10 points of einf's loss are not
explained by the host change. The per-step decomposition
(`benchmarks/step-breakdown-2026-08-23.md`) has since **ruled out** the PyO3
boundary as the cause — `schedule()` plus `apply_result()` is 0.7-1.2% of the
step — so the remaining candidates are the different PyTorch/CUDA build and the
active remote-desktop session. Still unattributed; it is not worth more effort
than a re-run once Gate 6 lands.

Where the step actually goes (`benchmarks/step-breakdown-2026-08-23.md`,
decode-only, concurrency 16): `forward` 25.704 ms of a 28.302 ms step (90.8%),
device busy share only 42.5%, 3079 kernel launches per step, and **1172 blocking
device-to-host scalar reads per step** — about 73 per request, 3 per request per
layer — from the per-request loop in the attention forward. The paged-decode
kernel is launched 384 times per step (24 layers x 16 requests) and averages
16.9 us per launch for a single token, so it is latency-bound, not work-bound.

## 3. Working rule for every Gate below

A Gate is only done when it lands three things together:

1. correctness evidence (tests, or equality against a reference path);
2. a number, measured on the matched workload;
3. one paragraph in `docs/` or `benchmarks/` explaining *why* the number moved.

## 4. Gate 6 — Batched decode, overhead removal, CUDA Graph

Highest priority. No new user-visible features. This is the Gate that produces
the strongest engineering narrative because every step is quantifiable.

- **6.0 Re-baseline. Done (2026-08-23).** Full suite passes on the 4090 host
  post-migration (130 passed in 75.61s); the matched workload is re-measured
  against vLLM on the same stack (`benchmarks/post-migration-2026-08-23.md`);
  and the per-step decomposition tool
  (`benchmarks/profile_step_breakdown.py`, report
  `benchmarks/step-breakdown-2026-08-23.md`) is the scoreboard for everything
  below.
- **6.1 Remove the per-request device-to-host scalar reads — DONE (paged path).**
  `ModelInput` now carries `query_start_loc_host` and `context_lens_host`, built in
  `from_plan` from the Python lists it already assembles, and the per-layer loop in
  `qwen.py` reads those instead of calling `.item()` on device tensors. The device
  tensors stay, because ops and index arithmetic still consume them. No flag, no
  fallback path: the values are identical by construction, and
  `test_host_metadata_matches_device_tensors` pins that invariant.

  Measured at concurrency 16, decode-only:

  | metric | before | after | |
  | --- | --- | --- | --- |
  | blocking DtoH scalar reads per step | 1185 | **19** | 62x |
  | `cudaStreamSynchronize` per step | ~1200 | **48** | 25x |
  | kernel/memcpy launches per step | 3079 | 1927 | |
  | device time per step | 9.492 ms | 8.578 ms | |
  | step | 22.426 ms | **10.726 ms** | **2.09x** |
  | forward | 20.516 ms | 8.702 ms | 2.36x |
  | throughput, n=5 | 604.39 tok/s | **1137.52 tok/s** | **1.88x** |

  The structural counts are the real acceptance criterion because they are immune
  to the host drift documented in `post-migration-2026-08-23.md`; throughput is
  reported with n=5 and its spread (sd 18.88, 1.66%) as a secondary check, and the
  533 tok/s effect dwarfs both the 1.7% within-session noise and the 25% across-
  session drift.

  **The bottleneck has flipped.** Device busy share went from 42% to 80%
  (8.578 ms of a 10.726 ms step), and `forward` is now within 1.5% of its own
  device time — it is device-bound. Every kernel-level item below therefore
  becomes worth doing, which was not true before this landed.

  Still open, deliberately out of scope: `model_runner.py:176-179` has the same
  pattern but only serves `--decode-backend eager`, and `sampler.py:41,81` account
  for the residual 19 reads per step. `executor.py:100` must stay — the scheduler
  needs the sampled tokens on the host.
- **6.2 Packed batched paged decode.** One batched paged-decode launch per layer
  instead of 24 x batch launches per step. The isolated sweep
  (`benchmarks/micro-paged-decode-2026-08-23.md`) shows why this is worth doing
  and what it is worth: the single-request kernel has a fixed device-side floor of
  ~9.5-11.6 us per call plus 6.5 us of host enqueue, so at the 512-token context
  this project benchmarks ~90% of the kernel's device time is fixed cost, not data
  movement. Batching pays that floor 24 times per step instead of 384. The payoff
  shrinks with context (about 5-10x on attention device time at 512-2048, about
  1.3x at 16K), because longer K legitimately does supply parallelism through the
  split dimension.
  Design consequence for the scoreboard: `paged_attention_batched_cuda.cu`
  currently plans `grid = (num_q_heads, num_decode_requests)` with **no split
  dimension**. Keep the split dimension, but make the count a runtime function of
  `(batch, context, num_kv_heads)` rather than a constant, because batch and splits
  are interchangeable as CTA suppliers — `CTAs = num_q_heads x batch x splits`, so
  batch 16 with 1 split is the same 224 CTAs that one request with 16 splits gives
  today. Three things the split dimension still buys that batch cannot:
  1. **Saturation.** 224 CTAs is 7 warps per SM on 128 SMs (3.5 effective, given
     the idle-warp effect below), far short of the ~28 needed to hide the per-token
     dependent chain. Reaching ~1024 CTAs at batch 16 needs about 4 splits. Today
     that is unreachable because the legal maximum is `num_logical_blocks = 34`.
  2. **Bounding the serial per-warp loop.** Without splits each warp walks
     `context / 4` tokens: 136 at context 544, but **4096 at context 16384**, which
     at the measured per-token rate is ~2.4 ms for one layer, or 57 ms per step
     across 24 layers. Splits divide this directly; batch does not touch it,
     because batch adds parallel warps rather than shortening any warp's chain.
  3. **Paying for the GQA merge.** Folding the group into one CTA drops `grid.x`
     from 14 to 2, leaving only 32 CTAs at batch 16, so splits have to make up the
     difference.
  Throughput ceiling for the batched kernel, extrapolated from the saturated
  context-16384 measurement (6.19 warp-token iterations per ns): batch 16 at
  context 544 is `14 x 544 x 16 = 121856` iterations, about **20 us per layer**
  against the 187 us that 16 separate calls cost today, i.e. ~9x on attention
  device time. At batch 16 and short context the optimal split count is plausibly
  2-4 rather than 16 — make it sweepable and measure it, do not hardcode it.
- **6.2a Split-heuristic fix — DONE, and the clearest lesson in this file.**
  `choose_num_splits` in `src/einf/executors/torch/qwen.py` replaced
  `max(1, num_logical_blocks // 4)` with
  `max(min(num_logical_blocks, 16), num_logical_blocks // 4)`, reproducing every
  optimum measured in `benchmarks/micro-paged-decode-2026-08-23.md` and pinned by
  tests in `tests/test_torch_qwen.py`. Isolated gain at context 544 is **1.30x** on
  the kernel call (15.18 → 11.70 us).
  The same patch, measured end to end by the same same-session A/B on the same
  hardware:

  | | before 6.1 | after 6.1 |
  | --- | --- | --- |
  | old, 8 splits | 600.18 tok/s (n=7) | 1014.39 tok/s (n=4) |
  | new, 16 splits | 604.39 tok/s (n=7) | 1152.90 tok/s (n=4) |
  | effect | +0.70%, t = 1.26, not significant | **+13.7%, t = 14.1** |

  Predicted from the kernel measurement: 384 calls x 3.48 us = 1.34 ms of a
  10.726 ms step, i.e. 12.5% against 13.7% measured. Optimisation order is not a
  matter of taste — the same patch is worth 0.7% or 13.7% depending only on what
  runs before it, and a null end-to-end result does not invalidate a correct
  kernel measurement.
- **6.2b Kernel efficiency — must be designed INTO 6.2, not sequenced after it.**
  The phase analysis (`benchmarks/micro-paged-decode-2026-08-23.md`) finds three
  defects, and the first one cannot be fixed on its own:
  (i) **7x GQA read amplification.** `grid.x` runs over the 14 Query heads while
  `head_kv = q_head / 7`, so the seven CTAs sharing a KV head walk the same logical
  block interval and read byte-identical K/V. At context 544 the kernel issues
  1904 KiB to consume 272 KiB. Since the seven reads hit the same addresses they
  are largely served by L2, so the cost is 7x load instructions, 7x L2-to-SM
  bandwidth and 7x independently exposed latency rather than 7x HBM traffic.
  The natural fix — one CTA per `(kv_head, split)` holding all 7 Query heads, so
  each loaded 128 B K/V tile feeds seven dot products — is cheap in registers
  (per lane: one `float2` becomes seven, 14 floats) **but shrinks the grid 7x**.
  At context 544 with 2 KV heads and 34 logical blocks, even the maximum legal
  split count gives only `2 x 34 = 68` CTAs, i.e. 0.53 of a wave on 128 SMs, and
  `head_dim` is 64 so there is nothing else to split. The missing parallelism has
  to come from the batch, which is exactly what 6.2 supplies: `(kv_head, split,
  request)` is 1088 CTAs at batch 16. **So the GQA-group-per-CTA layout belongs in
  the batched kernel from the start; landing it alone would starve the grid.**
  (ii) **The merge kernel is a single-warp serial reduction** — `grid = 14 x 32
  threads`, looping over `num_splits` with a dependent online-softmax merge. It
  costs 3.64 us of an 11.59 us call at 16 splits and 11.4 us at 64, caps useful
  splits near 64, and is most of the 5.6x collapse at the maximum legal split
  count. It gets worse under (i), which needs *more* splits to hold occupancy, so
  it is coupled too: parallelise it over `head_dim` with a tree reduction over
  splits.
  (iii) **Warps idle at the measured optimum.** The 4 warps partition the CTA's
  logical blocks, and `base_warp_blocks = block_this_cta / NUM_WARPS` is 0 whenever
  a CTA holds fewer than 4 blocks. At context 544 with 16 splits, 14 of the 16
  splits get 2 blocks, so warps 2 and 3 do nothing — about half the warps idle in
  the configuration that measured fastest. Partitioning warps over tokens rather
  than whole blocks would recover this.
  Only after those is the per-token dependent chain (~1080 cycles per token per
  active warp: one coalesced 128 B load, 32 FFMA, five shuffles, a serial lane-0
  `expf` update, a broadcast) worth attacking.
  Note for roofline claims: at long context these loads are served from L2, not
  HBM — context 16384 issues 1187 GB/s, i.e. 118% of HBM peak — so compare against
  L2 (~2.5 TB/s reference), where the kernel already sits near 47%.
- **6.3 Persistent buffers.** Preallocate device and pinned buffers; write block
  tables and index arrays directly from Rust into them instead of rebuilding
  Python objects each step. Also folds in the six per-step sampling-metadata
  `torch.tensor` copies (0.34-0.43 ms per step for values that rarely change).
- **6.4 Batched GPU sampling.** One sampling call for the whole batch; no
  per-request Python loop.
- **6.5 CUDA Graph capture** of the decode step, with fixed batch buckets
  (1/2/4/8/16) and a documented fallback for shapes outside the buckets. Comes
  last because graph capture cannot tolerate the synchronisations that 6.1
  removes.

Exit criteria: the per-step decomposition is forward-dominated, and the
concurrency 8/16 ratio against vLLM on the same host is materially reduced.
Target — explicitly a target, not a result — is within roughly 2x.

## 5. Gate 7 — W4A16 quantisation with a fused dequant GEMM

Chosen ahead of speculative decoding because it is self-contained (linear
layers only, no scheduler change), because decode is memory-bound so the win is
directly measurable, and because it adds a substantial *new* kernel rather than
replacing an existing one.

- 7.0 Offline weight packing: group-wise scale/zero, layout chosen for the
  `ldmatrix` access pattern of the consuming kernel.
- 7.1 Dequant-fused GEMV/GEMM kernel: dequantise in registers, keep the
  tensor-core path, avoid materialising fp16 weights.
- 7.2 Integration with the Qwen linear layers plus an accuracy gate against the
  fp16 path (token-level agreement on a fixed prompt set, or perplexity).

Exit criteria: measured reduction in decode weight traffic (expect 3-4x),
measured end-to-end decode speedup, and kernel efficiency reported against
achievable bandwidth.

## 6. Gate 8 — Speculative decoding, n-gram first

Deferred behind Gate 7 because it touches the scheduler and its payoff depends
on acceptance rate.

- **8.0 Prerequisite:** append attention with `query_len > 1`. The existing
  paged attention must accept multi-token queries before verification is
  possible.
- 8.1 n-gram / prompt-lookup proposer, batched verification, KV rollback, and
  acceptance-rate accounting. No draft model, no extra training.
- 8.2 Optional follow-up: small draft model reusing the same mechanism.

Exit criteria: acceptance rate and net speedup measured, and rollback proven
correct by output equality against non-speculative greedy decoding.

## 7. Gate 9 — Tensor parallelism

Scheduled last because the end-to-end run requires renting a second GPU. The
code, however, does not have to wait.

- 9.0 Column/row parallel sharding, all-reduce after attention `out_proj` and
  MLP `down_proj`, sharded weight loader.
- 9.1 Validate logic at `world_size=1` and with the CPU gloo backend — this
  covers almost all of the correctness risk at zero cost.
- 9.2 Rent two GPUs for the scaling measurement only.

## 8. Cross-cutting: presentability

Continuous work, not a final phase. This is what makes the numbers legible to
someone reading the repository for the first time.

- `README.md`: architecture, design decisions, and numbers.
- A benchmark report with TTFT / TPOT / throughput against concurrency, on the
  same workload as the reference engine.
- Correctness suite: greedy token-level agreement with Hugging Face, and
  prefix-cache-on equals prefix-cache-off.
- Observability: prefix cache hit rate, batch composition, preemption count —
  so a benchmark curve can be *explained* rather than only shown.
- Kernel layer presented independently: per-kernel benchmarks against torch,
  cuBLAS, CUTLASS and FlashInfer, with roofline or SASS analysis. The existing
  `cute_gemm` result (76-79% of CUTLASS SIMT against a ~54 TFLOP/s golden, with
  the remaining gap attributed to B-load vectorisation and epilogue stores via
  SASS inspection and a 30-configuration grid search) is the template.

## 9. FlashInfer positioning

FlashInfer is adopted as a **selectable backend**, not a replacement:
`--attn-backend=einf|flashinfer`. Rationale — it provides an authoritative
reference that quantifies the gap of the in-house kernels, and an honest gap
report is more valuable than an unmeasured claim. Scope notes:

- It covers attention (prefill/decode/append, paged and ragged), paged KV
  append, sampling, RMSNorm, activations, RoPE, cascade attention for shared
  prefixes, MLA, and specialised fp8/fp4/MoE GEMM. It does **not** cover
  scheduling, block ownership, prefix reuse policy, or plain dense linear
  layers.
- Its `plan()`/`run()` split maps well onto `BatchPlan`: `plan()` is CPU-side
  and cannot be captured in a CUDA Graph or `torch.compile`, while `run()` can.
- On this hardware (sm89) the FA2 path applies; FA3, trtllm-gen and fp4 paths
  are Hopper and newer.
- Cascade attention is the natural pairing with Prefix Cache, but only after
  Prefix Cache semantics are stable.

## 10. Benchmark environment notes

Facts established on the current 4090 host that affect any measurement:

- The Qwen2.5-0.5B checkpoint now lives on the 1.1 TB volume of the current
  host, with `~/models/Qwen2.5-0.5B` symlinked to it. The scripts no longer
  hardcode the failed machine's path: `benchmark_qwen_serving.py`,
  `benchmark_qwen_e2e.py`, `run_qwen.py` and `profile_step_breakdown.py` resolve
  `$EINF_MODEL_DIR` and fall back to `~/models/Qwen2.5-0.5B`.
- The root filesystem has roughly 15 GB free; large artefacts and model weights
  belong on the 1.1 TB volume, not in `~/.cache`.
- The active venv sets `include-system-site-packages = true` over an Anaconda
  installation owned by another user, so `torch`, `vllm` (0.17.0) and
  `flashinfer` (0.6.4) are all inherited, not ours. Install into the venv only;
  never modify the inherited site-packages.
- The inherited vLLM is older than the 0.26.0 used for the previous golden, so
  a re-baseline must state the version and treat cross-host comparison with the
  old table as indicative only.
- Nsight Compute counters are blocked for this user on this host, so kernel
  analysis relies on `cuobjdump` SASS/resource dumps and end-to-end timing.

## 11. Non-goals for now

HTTP/streaming serving surface, production hardening, MoE, MLA, LoRA, and
kernel micro-optimisation of paths that the profile does not show as dominant.
