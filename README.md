# einf

`einf` is a learning-oriented single-GPU LLM inference engine. The project will
remain in this directory and evolve across the later learning gates instead of
creating one disposable project per gate.

## Current stage

**Gate 5 — Paged Decode Attention — current**

Gate 2 is happy-path complete: the project has a CPU-only deterministic control
plane with Scheduler-owned Request lifecycle, transactional logical KV block
allocation, immutable scheduled batches, and a synchronous fake executor.

Gate 3 is happy-path complete as of 2026-08-04. Gate 4 is happy-path complete
as of 2026-08-05: the engine now has preallocated GPU block KV storage, packed
model metadata, CUDA KV write/gather custom ops, a PyTorch reference backend,
and a real Qwen2.5-0.5B path whose chunked Prefill→Decode output matches
Hugging Face. Gate 5.0A provides a three-kernel contiguous Attention CUDA
oracle, and Gate 5.1 provides a correctness-first FlashAttention-style forward
that can optionally replace the eager QwenAttention body. Direct paged KV reads
are the current step.

## Language boundary

```text
Python control plane
- Request lifecycle and Scheduler
- logical KV block metadata and ownership
- ScheduledBatch / ExecutionPlan
- orchestration, traces, and benchmarks

C++/CUDA data plane
- Torch custom ops
- GPU KV cache movement
- paged decode attention
- fused kernels and CUDA resource RAII
```

## Project organization

Code is organized by stable subsystem rather than by Gate, so later work extends
the same engine instead of duplicating it.

```text
einf/
├── pyproject.toml
├── README.md
├── src/einf/
│   ├── request.py
│   ├── scheduler.py
│   ├── execution.py
│   ├── lib.py
│   ├── executors/
│   │   ├── fake.py
│   │   └── torch/
│   │       ├── model_runner.py
│   │       ├── qwen.py
│   │       └── ops/
│   └── cache/
│       ├── pool.py
│       ├── reservation.py
│       ├── manager.py
│       └── storage.py
├── tests/
├── benchmarks/
│   └── traces/
├── scripts/
├── docs/
└── artifacts/
```

## Implemented mechanisms

- guarded Request lifecycle with explicit admission, completion, cancellation,
  failure, and preemption operations;
- transactional BlockPool reservations and logical KV block tables;
- immutable token-budget ScheduledBatch values;
- deterministic FakeExecutor execution and result application;
- chunked prefill with sampling only on the final chunk;
- persistent FCFS and decode-first policies;
- continuous admission and cache-pressure preemption;
- preallocated block-based Torch KV Cache with CUDA custom write/gather ops;
- packed Reference and Qwen2.5 model runners;
- real Qwen2.5-0.5B chunked Prefill→Decode through the full server stack;
- PyTorch reference paths and CUDA parity/microbenchmark evidence.
- three-kernel contiguous causal Attention with FP32/BF16 and GQA parity.
- tiled FlashAttention-style causal forward with optional Qwen integration.
- direct single-request Paged Decode Attention with physical block-table reads.

The project prioritizes end-to-end happy paths for common inference-engine
mechanisms. Gate 3 fairness aging, randomized stress traces, production error
classification, control-plane stress tests, formal Extension packaging, and
broader Gate 4 hardening remain non-blocking work.

## Gate 5 learning sequence

Gate 5 separates Attention math, fusion, IO-aware tiling, and paged addressing:

```text
    three-kernel contiguous Attention
    → fused contiguous Decode Attention with online softmax
    → FlashAttention-style contiguous causal forward
    → Paged Decode Attention
```

The single-request Paged Decode path now splits each Query head's Context across
four Warps and merges partial FP32 online-softmax states inside the CTA. Batched
Decode and model integration remain next.

`einf::contiguous_attention(Q, K, V, start_pos, scale)` is complete and frozen
as a simple native oracle. It materializes FP32 scores and separates QK, stable
softmax, and PV. `einf::flash_attention` reuses the same contract without a
global score Tensor and is available through `QwenModelRunner(...,
use_flash_attention=True)`. Gate 5.2 now replaces gathered contiguous K/V with
direct block-table reads. See `docs/gate5-paged-attention.md` and `CHECKPOINT.md`.

## Development

```bash
cd ~/python/aiinfra/projects/einf
source ~/python/.venv311/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

## Qwen2.5-0.5B end-to-end smoke test

Install the optional model dependencies and point the smoke test at a local
Hugging Face checkpoint:

```bash
python -m pip install -e '.[dev,qwen]'
python scripts/run_qwen.py \
  --model-dir /home/wyg/python/models/Qwen2.5-0.5B \
  --max-prefill-chunk-len 4 \
  --flash-attention \
  --compare-hf
```

The script exercises `LLMServer -> Scheduler -> TorchExecutor -> packed Qwen
forward -> block KV Cache custom ops` and optionally checks greedy token parity
against Hugging Face. The current Qwen path is inference-only; it does not
define a loss or backward/training contract.

## Collaboration boundary

The learner owns the core state machine, ownership transfers, failure semantics,
scheduling decisions, and benchmark interpretation. AI may prepare scaffolding,
fixtures, repetitive tests, counterexamples, reviews, and result organization.
