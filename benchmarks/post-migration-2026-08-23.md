# Post-migration serving baseline (2026-08-23)

Fresh re-measurement after the Rust/PyO3 control-plane migration (`1d9b350`).
This supersedes the environment and numbers in `vllm_golden.md`, which were
measured before the migration on the previous 4090 host that has since suffered
a disk failure.

One methodological improvement over the previous golden: einf and vLLM now run
on the **same host, same Python, same PyTorch build**, so the comparison is
stack-matched rather than cross-stack.

## Environment

```text
host:        RTX 4090, driver 550.163.01, 24564 MiB
python:      3.12.2  (venv with include-system-site-packages over Anaconda)
torch:       2.10.0+cu128, CUDA 12.8   (shared by both engines)
einf:        1d9b350, decode_backend=paged, prefill_backend=eager
vLLM:        0.17.0 (inherited site-packages), VLLM_USE_FLASHINFER_SAMPLER=0
model:       /media/zzx/新加卷1/models/Qwen2.5-0.5B, BF16
confound:    a ToDesk remote-desktop session held ~798 MiB and was active
raw logs:    ../../artifacts/bench-2026-08-22/   (outside the repo)
```

## Matched workload

```text
prompt length:   512 tokens
output length:    64 tokens
token budget:    512 (einf max_batch_len / vLLM max-num-batched-tokens)
prefill chunk:   128 (einf)
EOS:             ignored
sampling:        greedy
concurrency:     1, 4, 8, 16
einf:            8 warmup + 32 measured completions, closed loop, seed 0
vLLM:            one warmup run, then 40 formal requests (48 at concurrency 16)
vLLM server:     --max-num-seqs 8 (16 for the concurrency-16 run),
                 --max-model-len 32768, --gpu-memory-utilization 0.8
```

## Results

| concurrency | einf tok/s | vLLM tok/s | ratio | einf TTFT p50 | vLLM TTFT p50 | einf ITL p50 | vLLM ITL p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 125.66 | 430.92 | 3.43x | 38.78 | 11.75 | 6.93 | 2.08 |
| 4 | 302.76 | 1763.50 | 5.83x | 89.60 | 13.98 | 11.26 | 2.00 |
| 8 | 401.70 | 3421.02 | 8.52x | 111.26 | 18.51 | 18.21 | 1.99 |
| 16 | 477.91 | 5018.90 | 10.50x | 165.68 | 43.34 | 29.03 | 2.05 |

Times are milliseconds. einf figures are `output_tokens_per_second`; vLLM
figures are `Output token throughput`.

einf detail:

| concurrency | req/s | steps | step p50 | avg scheduled tokens | prefill tok/s | peak alloc MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.963 | 2211 | 6.954 | 8.58 | 1005.24 | 1010.9 |
| 4 | 4.731 | 603 | 11.380 | 34.33 | 2422.11 | 1172.1 |
| 8 | 6.250 | 338 | 18.382 | 69.97 | 3299.82 | 1226.9 |
| 16 | 7.574 | 204 | 29.090 | 139.72 | 3935.17 | 1355.2 |

Paged versus eager control at concurrency 8: **401.70 vs 254.22 tok/s**, i.e.
the paged decode path is 1.58x the gathered eager path on the real workload —
the in-house paged attention is doing its job.

Memory is not comparable: einf peaks at 1.0-1.4 GiB while vLLM reserves roughly
19.6 GiB of KV cache at `gpu_memory_utilization=0.8`. Compare latency and
throughput only.

## The diagnostic that matters

**vLLM's median inter-token latency is flat across the whole concurrency range
(2.08 → 2.05 ms). einf's grows 4.19x (6.93 → 29.03 ms).**

Equivalently, from concurrency 1 to 16 einf scales throughput 3.80x while vLLM
scales 11.65x. A decode step that costs proportionally more as the batch grows
is the signature of per-request work inside the step rather than per-batch work:
einf still launches one paged-decode operator per request per layer, rebuilds
plan structures each step, samples per request, and captures nothing in a CUDA
Graph. Prefill is not the problem — einf's prefill throughput scales 1005 →
3935 tok/s over the same range, and TTFT at concurrency 1 actually improved
relative to the pre-migration measurement (45.00 → 38.78 ms).

This is exactly the Gate 6 work item list in `docs/ROADMAP.md`, and it is now
measured on the current stack rather than inferred.

## Comparison with the pre-migration golden

Both engines are slower on this host than the numbers recorded in
`vllm_golden.md`, so the host itself accounts for part of the change:

| concurrency | einf before | einf now | delta | vLLM before (0.26.0) | vLLM now (0.17.0) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 139.86 | 125.66 | -10.2% | 483.63 | 430.92 | -10.9% |
| 4 | 363.06 | 302.76 | -16.6% | 1858.98 | 1763.50 | -5.1% |
| 8 | 473.37 | 401.70 | -15.1% | 3519.50 | 3421.02 | -2.8% |
| 16 | 571.57 | 477.91 | -16.4% | 5301.21 | 5018.90 | -5.3% |

Reading this honestly: vLLM's -3% to -11% is a rough proxy for the host and
version delta, and einf appears to lose roughly 10 percentage points more than
that at concurrency 4-16. **That attribution has since been withdrawn — see the
next section.** The apparent gap is smaller than this host's day-to-day
measurement drift, so the data cannot support it.

## Measurement validity: this host drifts more than the effect being measured

Re-running the concurrency-16 row on 2026-08-23 with the decode path unchanged
(`--paged-decode-max-splits 8` reproduces the split count this table was measured
with) gives **600.18 tok/s, sd 7.24, n = 7**, against the **477.91** recorded
above. That is **+25.6% with no meaningful code change**, and it is larger than
the entire ~10-point regression the previous section tried to attribute.

Two separate magnitudes matter, and they should not be confused:

- *Within* one session, back-to-back runs vary by about 1-2% (sd 0.9-1.2% over
  7 runs per arm), which is tight enough to A/B a change.
- *Across* sessions, the same command drifted 25.6%, which makes absolute
  throughput from different sessions non-comparable.

Recorded differences in host state between the two measurements: the display and
remote-desktop session held ~798 MiB at baseline versus 1873 MiB on re-measurement,
persistence mode is disabled so the SM clock idles at 210 MHz between runs and must
ramp, and no application clock is pinned. No thermal or power throttle flags were
active either time.

Consequences for how this project measures from now on:

1. Treat every absolute number in this file as session-local. The einf-vs-vLLM
   ratios were at least collected back to back in one session, so they are more
   defensible than the cross-session comparison above, but they still need
   re-measurement with repetitions before being quoted as a scoreboard.
2. Report a median plus spread over n >= 5 runs, never a single run.
3. Evaluate any change as a same-session A/B, ideally through a flag that
   switches behaviour inside one process, as was done for the split heuristic in
   `micro-paged-decode-2026-08-23.md`.
4. If the machine's owner permits it, enable persistence mode and pin application
   clocks; both are GPU-wide settings on a borrowed host, so ask first.

The practical conclusion is that the post-migration regression hunt was chasing
measurement noise. The PyO3 boundary was already exonerated directly by the step
decomposition (0.7-1.2% of the step), and this settles the remainder: there is no
established regression to explain.

## Reproduce

```bash
# einf
export CUDA_HOME=/usr/local/cuda-12.1 PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8
python benchmarks/benchmark_qwen_serving.py \
  --model-dir /media/zzx/新加卷1/models/Qwen2.5-0.5B \
  --decode-backend paged --concurrency 8 \
  --prompt-lens 512 --output-lens 64 \
  --warmup-completions 8 --measured-completions 32 \
  --seed 0 --progress-every 0

# vLLM server
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.cli.main serve \
  /media/zzx/新加卷1/models/Qwen2.5-0.5B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 \
  --max-num-batched-tokens 512 --max-num-seqs 8 --gpu-memory-utilization 0.8

# vLLM client
VLLM_USE_FLASHINFER_SAMPLER=0 python -m vllm.entrypoints.cli.main bench serve \
  --backend vllm --model /media/zzx/新加卷1/models/Qwen2.5-0.5B \
  --host 127.0.0.1 --port 8000 --dataset-name random --num-prompts 40 \
  --random-input-len 512 --random-output-len 64 --request-rate inf \
  --max-concurrency 8 --ignore-eos --temperature 0 --seed 0
```

Note: the Qwen2.5-0.5B checkpoint had to be downloaded to this host; the
benchmark script's `DEFAULT_MODEL_DIR` still points at the failed machine's
path, so `--model-dir` is mandatory here.
