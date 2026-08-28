# Paged decode kernel micro-benchmark (2026-08-23)

Answers a question the end-to-end decomposition cannot: how good is the
single-request Split-KV kernel *by itself*, and does a long context supply enough
parallelism on its own to make a batched operator unnecessary?

Tool: `benchmarks/micro_paged_decode.py`. RTX 4090, bf16, 14 Q heads / 2 KV heads,
head_dim 64, block_len 16, 400 iterations per configuration.

Two method details the numbers depend on:

* The KV cache is 512 MB — far past the 4090's 72 MB L2 — and every timed
  iteration reads a **different** region through a different block table. Timing
  one short context in place would keep the working set in L2 and report a
  bandwidth number the real decode loop never sees.
* Both device time (CUDA events) and host enqueue time (wall time of the enqueue
  loop) are reported, because a flat floor could be either the kernel or the call
  path.

## Split-KV parallelism sweep

`grid = (14 Q heads, num_splits)`; "waves" is grid / 128 SMs. KV GB/s counts only
the unavoidable `context * 2 * num_kv_heads * head_dim * itemsize` traffic.
`num_splits = 1` is served by the non-split op, as production does.

| context | splits | blocks | waves | dev us | host us | KV GB/s | %peak | partial KB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 544 | 1 | 14 | 0.11 | 50.34 | 3.5 | 5.5 | 0.5% | 0 |
| 544 | 4 | 56 | 0.44 | 19.54 | 6.5 | 14.3 | 1.4% | 29 |
| 544 | **8** | 112 | 0.88 | **15.14** | 6.5 | 18.4 | 1.8% | 58 |
| 544 | **16** | 224 | 1.75 | **11.61** | 6.5 | 24.0 | 2.4% | 116 |
| 544 | 32 | 448 | 3.50 | 14.22 | 6.5 | 19.6 | 1.9% | 231 |
| 544 | 34 | 476 | 3.72 | 14.66 | 6.5 | 19.0 | 1.9% | 245 |
| 1024 | 16 | 224 | 1.75 | **11.00** | 6.5 | 47.7 | 4.7% | 116 |
| 2048 | 32 | 448 | 3.50 | **13.86** | 6.5 | 75.7 | 7.5% | 231 |
| 4096 | 64 | 896 | 7.00 | **22.37** | 6.5 | 93.8 | 9.3% | 462 |
| 8192 | 64 | 896 | 7.00 | **30.71** | 6.5 | 136.6 | 13.5% | 462 |
| 16384 | 64 | 896 | 7.00 | **47.23** | 6.5 | 177.6 | 17.6% | 462 |
| 16384 | 128 | 1792 | 14.00 | 57.47 | 6.5 | 146.0 | 14.5% | 924 |
| 16384 | 1024 | 14336 | 112.00 | 262.60 | 6.5 | 31.9 | 3.2% | 7392 |

Shuffling the block table changed nothing anywhere (544: 11.61 vs 11.61; 16384:
47.23 vs 47.27), so paging locality is not a factor at any context length.

## What the sweep says

**1. There is a fixed device-side floor of ~9.5-11.6 us per call.** Device time is
flat from context 544 to 1024 (11.61 → 11.00 us) even though the KV traffic
doubles, and context 128 still costs 9.53 us. Host enqueue is 6.5 us, flat
everywhere, and always *below* the device time — so the queue never drains and
the floor is genuinely the two dependent kernels (partial + reduce) plus their
allocations, not the Python call path.

**2. The kernel body is *not* streaming-inefficient — it is redundant.** An
earlier draft of this file concluded from unique-byte arithmetic that the body
runs at ~253 GB/s, about 25% of peak, and called that a roofline problem. The
phase analysis below (`benchmarks/micro_paged_decode_phases.py`) shows that
reading was wrong: `grid.x` runs over the 14 Query heads while there are only 2
KV heads, so seven CTAs each load the same KV independently. Counting the loads
the hardware actually issues, the marginal rate over the 4096 → 16384 segment is
~1840 GB/s, and at context 16384 the total issued rate is 1187 GB/s — **118% of
HBM peak**, which is only possible because the loads are served from L2. The body
moves data at a respectable rate; six sevenths of what it moves is redundant.

**3. At the serving context this project benchmarks, the kernel is ~90% fixed
cost.** At context 544 the KV traffic is 272 KiB, which at 253 GB/s is ~1.1 us
against a measured 11.61 us.

**4. Longer K does buy real parallelism — the questioner's intuition is correct.**
Efficiency climbs monotonically with context: 2.4% of peak at 544, 4.7% at 1024,
7.5% at 2048, 9.3% at 4096, 13.5% at 8192, 17.6% at 16384, as the split count
that fits grows from 8 to 64.

**5. But splits are not free parallelism.** Past ~64 the collapse is severe:
context 16384 goes from 47 us at 64 splits to 263 us at the maximum legal 1024
splits, a 5.6x regression, because the partial state written and re-read scales
linearly with splits (7392 KB at 1024 splits, comparable to the 8.4 MB of KV
itself) and the reduce kernel loops over every split.

**6. The production split heuristic under-splits at short context.**
`min(64, max(1, num_logical_blocks // 4))` picks 8 where 16 is optimal at context
544 (15.14 vs 11.61 us, **1.31x**) and 2 where 8 is optimal at context 128 (16.73
vs 9.53 us, **1.76x**). At 1024 and above it happens to land on the measured
optimum. A rule that reproduces every measured optimum:

```python
num_splits = min(64, max(min(num_logical_blocks, 16), num_logical_blocks // 4))
```

Measured optima: 128 → 8, 544 → 16, 1024 → 16, 2048 → 32, 4096 → 64,
8192 → 64, 16384 → 64.

## Do we still need a batched operator?

Yes, but the reason is fixed-cost amortisation, not parallelism.

| context | per-call dev us (prod) | loop ms/step, 16 req x 24 layers | KV floor ms/step | ratio |
| --- | --- | --- | --- | --- |
| 128 | 16.73 | 6.43 | 0.025 | 257x |
| 544 | 15.16 | 5.82 | 0.106 | 55x |
| 1024 | 11.00 | 4.22 | 0.200 | 21x |
| 4096 | 22.37 | 8.59 | 0.799 | 11x |
| 16384 | 47.23 | 18.14 | 3.196 | 5.7x |

At context 544 a decode step spends 5.82 ms of device time on attention for
41 MB of real traffic. A batched kernel pays the ~11 us fixed cost 24 times per
step instead of 384, and saves 384 host enqueues of 6.5 us (2.50 ms) — which
independently reconciles with the 2.99 ms of `cudaLaunchKernel` measured in
`step-breakdown-2026-08-23.md`, and 384 x 15.16 us = 5.82 ms reconciles with the
7.27 ms the profiler attributed to the two paged kernels there. Two independent
measurements agreeing is the main reason to trust either.

The fixed-cost share, and therefore batching's payoff, shrinks as context grows:
~90% at 544, ~76% at 4096, ~24% at 16384. So batching is worth roughly 5-10x on
attention device time in the 512-2048 range this project benchmarks, and closer
to 1.3x at 16K context. Both statements are true at once, and the workload
decides which one matters.

Design consequence for the scaffold: `paged_attention_batched_cuda.cu` currently
plans `grid = (num_q_heads, num_decode_requests)` with **no split dimension**,
which would reintroduce the parallelism starvation this sweep measures at low
split counts (224 blocks at batch 16, versus 896 for one request at 64 splits).
The batched kernel needs a split dimension too — `(num_q_heads, requests x
splits)` with a batched reduce — and can then choose *fewer* splits per request
because the batch itself supplies parallelism.

## Why the effective bandwidth is so low

Tool: `benchmarks/micro_paged_decode_phases.py`. The "% of peak" figures above
are misleading in three independent ways, and this section separates them.

Phase split — profiler self device time per kernel against the event-timed total:

| ctx | splits | total us | partial | reduce | gap | not-attention share |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 8 | 10.30 | 6.67 | 2.29 | 1.34 | 35.3% |
| 544 | 8 | 15.70 | 11.81 | 2.28 | 1.61 | 24.8% |
| 544 | 16 | 11.59 | 6.93 | **3.64** | 1.02 | **40.2%** |
| 1024 | 16 | 11.94 | 7.06 | 3.57 | 1.31 | 40.9% |
| 4096 | 64 | 25.59 | 11.15 | **11.53** | 2.90 | **56.4%** |
| 16384 | 64 | 49.47 | 37.08 | 11.40 | 0.99 | 25.0% |

Unique bytes against the loads the hardware actually issues:

| ctx | unique KB | issued KB | unique GB/s | issued GB/s | % HBM peak |
| --- | --- | --- | --- | --- | --- |
| 128 | 64 | 448 | 6.4 | 44.5 | 4.4% |
| 544 | 272 | 1904 | 17.7 | 124.2 | 12.3% |
| 544 (16 splits) | 272 | 1904 | 24.0 | 168.2 | 16.7% |
| 1024 | 512 | 3584 | 43.9 | 307.4 | 30.5% |
| 4096 | 2048 | 14336 | 82.0 | 573.7 | 56.9% |
| 16384 | 8192 | 57344 | 169.6 | **1186.9** | **117.7%** |

Four distinct causes, largest first at the context this project benchmarks:

**(a) The 7x GQA read amplification.** `head_kv = q_head / 7` at line 91, with
`grid.x = num_attention_heads`, so the seven Query heads sharing a KV head each
load that KV independently. The metric counts 272 KiB at context 544 while the
kernel issues 1904 KiB. Correcting only this raises the figure from 2.4% to 16.7%
of HBM peak.

**(b) 40% of the call is not the attention.** At context 544 with 16 splits the
reduce kernel costs 3.64 us against the partial kernel's 6.93 us, plus 1.02 us of
exposed launch gap between two dependent kernels. The reduce kernel is
`grid = 14 blocks x 32 threads` — 14 CTAs of a *single warp* on a 128-SM GPU — and
it loops over `num_splits` serially with a dependent online-softmax merge. Its
cost therefore scales with the split count: 2.28 us at 8 splits, 3.6 us at 16,
11.4 us at 64. **This is the real reason over-splitting collapses**: at the
maximum legal 1024 splits for context 16384 the merge alone accounts for most of
the 263 us measured in the sweep above.

**(c) The per-token loop is latency-bound at ~1080 cycles per token per active
warp** (433 ns at context 544; the figure is roughly constant across context, so
it is structural rather than a bandwidth effect). Line 127 walks one token at a
time, and each iteration is a dependent chain: one coalesced 128 B load, 32 FFMA,
a five-step `__shfl_down_sync` reduction, a serial `expf`-based online-softmax
update on lane 0, then a broadcast. There is no unrolling across tokens and no
double buffering, so a warp has at most one load in flight.

**(c2) Warps idle at the measured optimum — and the split heuristic causes it.**
The 4 warps partition the CTA's *logical blocks* (lines 115-121): an even share
each, then the remainder handed one at a time to the leading warps. That is
correct, and it idles a warp only when a CTA holds fewer than 4 blocks. The catch
is that this is not a corner case but a direct consequence of the split count,
since `blocks_per_CTA ~ num_logical_blocks / num_splits`:

| ctx | num_logical | old `// 4` | blocks/CTA | new rule | blocks/CTA | idle warps |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 8 | 2 | 4 | **8** | **1** | **3 of 4** |
| 544 | 34 | 8 | 4 | **16** | **2** | **2 of 4** |
| 1024 | 64 | 16 | 4 | 16 | 4 | 0 |
| 2048 | 128 | 32 | 4 | 32 | 4 | 0 |
| 4096 | 256 | 64 | 4 | 64 | 4 | 0 |

The original `num_logical_blocks // 4` was exactly "four blocks per CTA, one per
warp", and never idled a warp. The floor of 16 added in 6.2a deliberately breaks
that invariant below context ~1024 — and still wins 1.30x, which is itself a
finding: at these sizes the kernel is starved of CTAs, not of warp throughput, so
trading half the warps for more CTAs pays. A configuration that gets both is
available: partition warps over *tokens* instead of whole blocks. Note the
per-token figure in (c) counts 2 active warps at this configuration; an earlier
draft assumed 4 and reported ~1445 cycles.

**(d) At long context the loads come from L2, not HBM.** One request's KV working
set is 8.4 MB at context 16384, well inside the 4090's 72 MB L2, and the issued
rate of 1187 GB/s exceeds HBM peak, which settles it. So "17.6% of peak" was
comparing against the wrong level: against an L2 reference of roughly 2.5 TB/s
the kernel is at ~47%, and its marginal streaming rate of ~1840 GB/s is ~74%.
There is far less headroom in the body than the HBM comparison implied.

Consequences, in the order the data supports:

1. The GQA-group-per-CTA layout — parallelise over `(kv_head, split)` and let the
   7 Query heads share each loaded K/V tile from registers — removes six sevenths
   of the issued loads and is cheap in registers (per lane, one `float2` becomes
   seven). **But it cannot ship on its own**: it shrinks the grid 7x, and at
   context 544 with 2 KV heads and 34 logical blocks even the maximum legal split
   count yields only `2 x 34 = 68` CTAs, 0.53 of a wave on 128 SMs, with
   `head_dim = 64` leaving nothing else to split. The batch has to supply the
   missing parallelism, so this layout belongs inside the batched operator
   (`(kv_head, split, request)` is 1088 CTAs at batch 16) rather than before it.
2. Fix the merge kernel: 14 single-warp CTAs with a serial loop over splits is what
   makes the split dimension expensive and caps useful splits near 64. It also gets
   worse under (1), which needs more splits to hold occupancy, so the two are
   coupled. Parallelise it over `head_dim` and reduce over splits as a tree.
3. Partition warps over tokens rather than whole logical blocks, which fixes the
   idle-warp effect in (c2).
4. Only then the per-token dependent chain (more tokens per iteration, more loads
   in flight).

None of these can be evaluated end to end before Gate 6.1 removes the host
serialisation — see the follow-up section below for what that changed.

## The split-heuristic fix, and why it does not show up end to end

`choose_num_splits` in `src/einf/executors/torch/qwen.py` now reads
`min(max_splits, max(min(num_logical_blocks, 16), num_logical_blocks // 4))`,
which reproduces every measured optimum above and is pinned by tests in
`tests/test_torch_qwen.py`. In isolation it does exactly what the sweep predicted:
at context 544 the call drops from 15.18 us to 11.70 us, **1.30x**.

End to end it does nothing measurable. Because `--paged-decode-max-splits 8`
makes the new helper return 8 at this context — exactly the old value — both
heuristics can be A/B'd back to back inside one session, which cancels the machine
drift documented in `post-migration-2026-08-23.md`:

| heuristic | n | mean tok/s | median | sd | range |
| --- | --- | --- | --- | --- | --- |
| old, 8 splits | 7 | 600.18 | 601.51 | 7.24 (1.2%) | 585.70 - 606.24 |
| new, 16 splits | 7 | 604.39 | 603.41 | 5.11 (0.9%) | 598.81 - 611.28 |

The difference is +0.70%, standard error 3.35 tok/s, t = 1.26 — not significant.
The sample is large enough to *exclude* the ~5% that 1.34 ms of saved device time
per step would predict (that would give t ~ 9), so this is a real null result and
not an underpowered one.

The reason is the host-bound structure measured in
`step-breakdown-2026-08-23.md`: at concurrency 16 the step spends ~16.3 ms on the
host across 24 layers x 16 requests, which is **~42 us of host work per (layer,
request)** against an 11.7-15.2 us attention kernel. The kernel is already hidden
behind host work, so shortening it moves no critical path — the GPU is waiting on
the CPU, not the reverse.

Keep the fix anyway: it is free, pinned by tests, and it will surface once 6.1
removes the synchronisations. But record its end-to-end value honestly as zero
today. It also settles the ordering question empirically — **no kernel work on
this path can pay off before the host serialisation is removed**, which applies
equally to the batched operator and the GQA fix above.

### Follow-up: after Gate 6.1 the same change is worth 13.7%

Gate 6.1 landed later the same day, removing the per-layer `.item()` reads and
taking the step from 22.426 ms to 10.726 ms. Re-running the identical A/B on the
identical code path:

| heuristic | n | mean tok/s | sd | vs before 6.1 |
| --- | --- | --- | --- | --- |
| old, 8 splits | 4 | 1014.39 | 18.70 | +0.70%, t = 1.26 (not significant) |
| new, 16 splits | 4 | 1152.90 | 6.04 | **+13.7%, t = 14.1** |

The one-line change went from unmeasurable to highly significant with no change
to itself, purely because the bottleneck moved. The magnitude also checks out
against the kernel measurement: 384 calls x 3.48 us saved is 1.34 ms of a 10.726 ms
step, or 12.5% predicted against 13.7% measured — agreement close enough to
confirm that `forward` is now device-bound.

Two lessons worth keeping:

1. Optimisation order is not a matter of taste here. The same patch measured 0.7%
   and 13.7% on the same hardware, and only the second number is the patch's
   actual value.
2. A null end-to-end result does not mean a kernel measurement was wrong. The
   isolated 1.30x was correct both times; what changed was whether anything
   downstream could observe it.

## Caveats

- The `%peak` columns are against 1008 GB/s theoretical; achievable copy bandwidth
  on this card is nearer 900 GB/s, so the real percentages are ~10% higher.
- `loop ms/step` assumes per-request calls serialise on one stream, which they do
  today; it is device time only and excludes the sync stalls measured separately.
- The ~1840 GB/s marginal issued rate is measured only along the 4096 → 16384
  segment at 64 splits, and the ~2.5 TB/s L2 figure it is compared against is a
  scale reference for AD102, not a measured limit on this card.
- Per-kernel splits come from `torch.profiler` self device time, which is reliable
  for kernels but inflates host-side figures; the totals it is compared against
  are CUDA-event timings from an unprofiled loop.
