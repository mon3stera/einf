# Per-step time decomposition (Gate 6.0, 2026-08-23)

Closes the open half of Gate 6.0. The question was where the einf step actually
spends its time, and specifically whether the Rust/PyO3 boundary explains the
regression seen in `post-migration-2026-08-23.md`. It does not — the answer is
host-device synchronisation inside the attention forward.

Tool: `benchmarks/profile_step_breakdown.py`. It owns the outer step loop
(mirroring `benchmark_qwen_serving.run_step`) and temporarily wraps
`ModelInput.from_batch`, `model_runner.forward`, `Sampler.sample`,
`torch.tensor` and `Tensor.cpu`; no production code is modified.

```bash
python benchmarks/profile_step_breakdown.py --concurrency 16 \
    --prompt-lens 512 --output-lens 64 --steps 200 --warmup-steps 40 \
    --profiler-steps 5 --timing wall
```

Environment as in `post-migration-2026-08-23.md` (RTX 4090, torch 2.10.0+cu128,
Qwen2.5-0.5B BF16, prompt 512 / output 64, paged decode). Raw logs:
`../../artifacts/bench-2026-08-22/breakdown-*.log`.

## Decode-only steps, wall mode

Mean milliseconds per step. `wall` adds no synchronisation, so a phase's time is
CPU-side occupancy plus whatever device work it ends up blocking on.

| phase | c=1 | c=4 | c=8 | c=16 |
| --- | --- | --- | --- | --- |
| sched (Rust) | 0.011 | 0.024 | 0.043 | 0.079 |
| input_build | 0.160 | 0.234 | 0.340 | 0.536 |
| **forward** | **6.375** | **11.028** | **15.775** | **25.704** |
| sampler | 0.369 | 0.493 | 0.536 | 0.838 |
| sampler_meta_h2d | 0.414 | 0.431 | 0.335 | 0.430 |
| d2h | 0.028 | 0.032 | 0.024 | 0.057 |
| exec_residual | 0.145 | 0.215 | 0.272 | 0.372 |
| sync_wait | 0.021 | 0.025 | 0.026 | 0.027 |
| apply_result (Rust) | 0.039 | 0.089 | 0.141 | 0.249 |
| step_residual | 0.009 | 0.011 | 0.011 | 0.012 |
| **step total** | **7.571** | **12.583** | **17.502** | **28.302** |
| forward share | 84.2% | 87.6% | 90.1% | 90.8% |

Device counters from the profiler window (the paged-decode kernel count of
exactly 24 x concurrency confirms the sampled steps were decode-only):

| counter | c=1 | c=4 | c=8 | c=16 |
| --- | --- | --- | --- | --- |
| kernel + memcpy launches / step | 1275 | 1603 | 2095 | 3079 |
| device time / step (ms) | 3.476 | 5.496 | 7.685 | 12.026 |
| **device busy share of step** | **45.9%** | **43.7%** | **43.9%** | **42.5%** |
| Memcpy DtoH / step | 77 | 296 | 588 | 1172 |
| DtoH per request | 77 | 74 | 73.5 | 73.3 |

## Where the host time goes, and why the syncs are expensive

Host-side CPU time per step at concurrency 16 (profiler window, `--steps 60`, so
its own step was 22.147 ms rather than the 28.302 ms of the 200-step run):

| host-side call | per step | CPU ms / step |
| --- | --- | --- |
| `cudaStreamSynchronize` | 1200 | **10.585** |
| `cudaLaunchKernel` | 1848 | 2.987 |
| `cudaMemcpyAsync` | 1219 | 1.695 |
| `aten::select` | 2769 | 1.328 |
| `einf::paged_decode_attention_split_kv` | 384 | 1.287 |
| `aten::_local_scalar_dense` | 1171 | 1.236 |
| `aten::empty_strided` | 948 | 1.211 |
| `aten::empty` | 802 | 0.944 |
| `aten::as_strided` | 3456 | 0.428 |
| `aten::item` | 1171 | 0.422 |

Two readings of this table matter, and they are different claims.

**The direct cost of the scalar reads is modest.** API plus dispatch plus the
copies is roughly 3.4 ms of CPU per step, and the 1171 `Memcpy DtoH` copies
consume 0.916 ms of *device* time. For scale, that is about 80% of the device
time of every dense GEMM in the model combined (1.149 ms/step for the cutlass
wmma kernels) — moving 1171 scalars costs nearly as much device time as all of
the model's matrix multiplies. It is still far less than the 7.27 ms the
paged-decode kernels spend, so the scalar reads are *not* more expensive than
the kernels.

**The damage is the lost overlap, not the direct cost.** The 10.585 ms of
`cudaStreamSynchronize` is not the price of `.item()`; it is device work the CPU
is forced to sit through. With ~16.3 ms of host work and ~12.0 ms of device work
per step, a pipelined loop would cost about `max(16.3, 12.0)` ≈ 16 ms, whereas
the measured step is 28.302 ms ≈ `16.3 + 12.0`. **The synchronisations convert a
max into a sum.** That is also why the harness `sync_wait` phase is always
~0.026 ms: everything has already been waited for inside `forward`.

Caveat on these CPU figures: the listed self-CPU times sum to 24.3 ms while the
same run's unprofiled step is 22.147 ms. The sum exceeding the whole is direct
evidence that profiling inflates host-side time, so read these as proportions,
not absolutes. Device-side numbers are much less affected.

## Findings

**1. The Rust/PyO3 boundary is not the problem.** `sched` + `apply_result` is
0.050 ms at c=1 and 0.328 ms at c=16 — 0.7% to 1.2% of the step. The hypothesis
that per-step PyO3 crossing explains the post-migration regression is **refuted**
by this data. Whatever caused that 10-point delta, it is not the control-plane
boundary, and the regression remains unattributed.

**2. Everything lives inside `forward`: 84-91% of the step.** Nothing else is
individually worth optimising yet. `input_build`, sampling metadata, the result
loop and both Rust calls together account for under 10%.

**3. The GPU is idle for more than half of every step.** Device busy share is
42-46% and *falls* as concurrency rises. The step is not device-throughput
bound; it is bound by the host-side gaps between kernels.

**4. The mechanism: ~73 device-to-host scalar reads per request per step.**
That is 1172 blocking `.item()`-style reads per step at concurrency 16, each one
a `Memcpy DtoH (Device -> Pinned)` plus a `cudaStreamSynchronize` (1200 of those
per step). Roughly 3 per request per layer, and it scales linearly with the batch
— which is exactly the shape of the ITL curve in
`post-migration-2026-08-23.md`. The relevant code is the per-request loop in the
attention forward (`model_runner.py`, `for request_idx in range(...)`). Every one
of those reads drains the pipeline, which is why `sync_wait` at the end of the
step is always ~0.026 ms: by then the device has long since been forced to
finish.

**5. Kernel launches are per-request, not per-batch.** At c=16 the profiler
counts 384 `einf::ops::paged_decode_attention_split_kv` launches per step —
exactly 24 layers x 16 requests — plus 3456 `aten::as_strided` and 2769
`aten::select` host ops per step from the same Python loop. The main paged-decode
kernel averages 16.9 us of device time per launch while serving a single token,
so each launch is latency-bound rather than work-bound.

**6. `sampler_meta_h2d` is small but pure waste.** Six `torch.tensor(...)` H2D
copies per step (temperature, top_k, top_p, min_p, seed, offset), rebuilt every
step regardless of batch size, costing 0.34-0.43 ms — 1.5-5.5% of the step for
values that rarely change.

## What this implies for Gate 6 ordering

Evidence, not a decision — the optimisation call is the author's:

- Removing the per-request scalar reads is the highest ratio of payoff to risk.
  The host-side values needed by the loop are already known to the control plane
  before the step starts, so they should not have to be read back from device
  tensors at all. The concrete sites are `qwen.py:187-190` (paged path, three
  reads per request per layer, so 72 per request per step) and the same three
  lines at `model_runner.py:176-179` for the eager path, plus two per request
  and one per step in `sampler.py:81` and `sampler.py:41`. Arithmetic check at
  concurrency 4: 24 x 3 x 4 + 4 x 2 + 1 = 297 predicted against 296 measured.
- Ceiling estimate if only the syncs go away: host and device work start to
  overlap, so the step approaches `max(host, device)` ≈ `max(16.3, 12.0)` ≈ 16 ms
  at concurrency 16, i.e. about 1.7x rather than the 2.3x an earlier draft of
  this file claimed. Reaching the device-time floor of ~12 ms additionally needs
  6.2 to remove the host cost of the 384-iteration loop (5.5 ms of Python and
  dispatch, 3.0 ms of launch API). **Estimates, not measurements.**
- Going past that needs the launch count down (one batched paged-decode launch
  per layer instead of one per request per layer) and then CUDA Graph capture,
  because at 3079 launches per step the residual host cost stays significant.
- Sampling metadata and `input_build` are worth doing after the above, as part
  of the persistent-buffer work, not before.

## Caveats

- The instrumentation itself costs something: wrappers around five entry points
  plus `perf_counter` calls. The agreement between `wall` (17.502 ms) and `sync`
  (17.765 ms) totals at c=8 suggests the distortion is small, but these numbers
  should be compared with each other, not with `benchmark_qwen_serving.py`
  output.
- The profiler window is 5 steps taken from the live mix; kernel counts confirm
  those steps were decode-only, but they are a small sample.
- A ToDesk remote-desktop session held ~798 MiB and was active during the run.
- `with_prefill` steps are reported separately in the raw logs and are dominated
  by the same `forward` term (88-90%); at c=16 prefill steps run 38.5 ms with
  `input_build` rising to 1.216 ms.

## Outcome: Gate 6.1 removed the reads

`ModelInput` now carries `query_start_loc_host` and `context_lens_host`, built in
`from_plan` from the Python lists it already assembles, and the per-layer loop in
`qwen.py` reads those instead of `.item()` on device tensors. The device tensors
stay for the ops and index arithmetic that consume them. The values are identical
by construction, so no flag or fallback path was added, and
`test_host_metadata_matches_device_tensors` pins the invariant.

| metric, concurrency 16 decode-only | before | after | |
| --- | --- | --- | --- |
| blocking DtoH scalar reads per step | 1185 | **19** | 62x |
| `cudaStreamSynchronize` per step | ~1200 | **48** | 25x |
| kernel/memcpy launches per step | 3079 | 1927 | |
| device time per step | 9.492 ms | 8.578 ms | |
| `forward` | 20.516 ms | 8.702 ms | 2.36x |
| step | 22.426 ms | **10.726 ms** | **2.09x** |
| device busy share | 42.3% | **80.0%** | |
| throughput, n=5 | 604.39 tok/s | **1137.52 tok/s** | **1.88x** |

The structural counts are the acceptance criterion, because they cannot be faked
by the host drift documented in `post-migration-2026-08-23.md`; throughput is a
secondary check reported with n=5 and sd 18.88 (1.66%), and its 533 tok/s effect
is two orders of magnitude past the within-session noise.

The prediction this report made — that the syncs turn `max(host, device)` into
`host + device` — holds up, but the mechanism was worth more than the estimate:
1.7x was predicted from restoring overlap, 2.09x was measured, because removing
the reads also removed their own host-side cost and 1152 memcpy launches per step.

**The bottleneck has flipped from host to device.** `forward` is now within 1.5% of
its own device time, so kernel work on the decode path finally pays. Demonstrated
immediately: the split-heuristic fix that measured +0.70% (t = 1.26) before 6.1
measures **+13.7% (t = 14.1)** after it, with the predicted 12.5% from kernel
arithmetic matching. See `micro-paged-decode-2026-08-23.md`.

Residual reads, deliberately left: `sampler.py:41,81` account for the remaining 19
per step, `model_runner.py:176-179` has the same pattern but only serves
`--decode-backend eager`, and `executor.py:100` must stay because the scheduler
needs the sampled tokens on the host.
