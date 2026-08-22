# vLLM golden serving baseline

This benchmark is the production-grade upper reference for einf. It is not a
kernel-only comparison: vLLM contributes FlashAttention Prefill, batched Paged
Decode, fused model kernels, CUDA Graphs, compilation, and a mature serving
runtime.

## Environment

The RTX 4090 server uses an isolated environment so vLLM's bundled PyTorch and
CUDA dependencies do not alter einf's development environment:

```text
environment: ~/python/.venv-vllm
vLLM:        0.26.0
PyTorch:     2.11.0+cu130
CUDA:        13.0
model:       /home/wyg/python/models/Qwen2.5-0.5B
dtype:       BF16
```

The current FlashInfer sampler JIT is incompatible with the server's available
toolkit headers. `VLLM_USE_FLASHINFER_SAMPLER=0` selects vLLM's native sampler;
it does not disable the FlashAttention-2 Attention backend. Greedy sampling is
used for the measured workload.

## Matched workload

The comparison fixes the workload to remove Prompt/output-distribution noise:

```text
Prompt length:             512 tokens
Output length:              64 tokens
Chunk/batch token budget:  512 tokens
EOS:                       ignored
Sampling:                  greedy
Concurrency:               1, 4, 8, 16
```

Start vLLM for the concurrency-8 comparison:

```bash
source ~/python/.venv-vllm/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve \
  ~/python/models/Qwen2.5-0.5B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.8
```

Run the official serving benchmark after one warmup run:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm bench serve \
  --backend vllm \
  --model ~/python/models/Qwen2.5-0.5B \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name random \
  --num-prompts 40 \
  --random-input-len 512 \
  --random-output-len 64 \
  --request-rate inf \
  --max-concurrency 8 \
  --ignore-eos \
  --temperature 0 \
  --seed 0
```

Run the matched einf workload with:

```bash
source ~/python/.venv311/bin/activate
cd ~/python/aiinfra/projects/einf
python benchmarks/benchmark_qwen_serving.py \
  --decode-backend paged \
  --concurrency 8 \
  --prompt-lens 512 \
  --output-lens 64 \
  --warmup-completions 8 \
  --measured-completions 32 \
  --seed 0 \
  --progress-every 0
```

## RTX 4090 results

```text
concurrency   einf tok/s   vLLM tok/s   vLLM/einf   einf TTFT p50   vLLM TTFT p50   einf ITL p50   vLLM TPOT p50
1                 139.86       483.63        3.46x          45.003           6.76           6.502            1.98
4                 363.06      1858.98        5.12x          81.604          15.78           9.976            1.92
8                 473.37      3519.50        7.44x         100.045          20.36          14.685            1.92
16                571.57      5301.21        9.27x         164.931          34.09          24.000            2.28
```

Times are milliseconds. The vLLM concurrency-16 server uses
`--max-num-seqs 16`; lower-concurrency runs use a limit of eight. vLLM's formal
runs process 40 requests for concurrency 1/4/8 and 48 requests for concurrency
16 after separate warmup runs. einf uses its closed-loop warmup and measurement
windows.

The measurements are intentionally a system-level target rather than a claim of
identical software stacks: the isolated vLLM wheel uses PyTorch 2.11/CUDA 13,
while einf uses PyTorch 2.10/CUDA 12.8. vLLM also reserves about 17.45 GiB for KV
cache at `gpu_memory_utilization=0.8`, whereas the measured einf configurations
allocate about 1.0-1.2 GiB. Compare latency and throughput, not memory efficiency,
from this table.

The widening throughput gap identifies the next einf system boundary. einf
still launches one Paged Decode operator per request per layer and uses eager
gathered Prefill. vLLM batches Decode requests into optimized Paged Attention,
uses FlashAttention for Prefill, and captures repeated execution with CUDA
Graphs. A packed batched Paged Decode operator is therefore the highest-value
next comparison point before local kernel micro-optimization.
