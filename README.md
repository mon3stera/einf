# einf

`einf` is a learning-oriented single-GPU LLM inference engine. The project will
remain in this directory and evolve across the later learning gates instead of
creating one disposable project per gate.

## Current stage

**Gate 3 — Continuous batching, chunked prefill, and cache preemption — happy-path complete**

Gate 2 is happy-path complete: the project has a CPU-only deterministic control
plane with Scheduler-owned Request lifecycle, transactional logical KV block
allocation, immutable scheduled batches, and a synchronous fake executor.

Gate 3 is happy-path complete as of 2026-08-04. It adds token-budget
batching, continuous admission, chunked prefill, persistent FCFS and
decode-first policies, cache preemption, deterministic recompute, and
single-request capacity failure without scheduler livelock. Gate 4 is the
current stage.

## Language boundary

```text
Python control plane
- Request lifecycle and Scheduler
- logical KV block metadata and ownership
- ScheduledBatch / ExecutionPlan
- orchestration, traces, and benchmarks

C++/CUDA data plane (later gates)
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
│   └── cache/
│       ├── pool.py
│       ├── reservation.py
│       └── manager.py
├── tests/
├── benchmarks/
│   └── traces/
├── scripts/
├── docs/
└── artifacts/
```

Additional subsystems such as the model runner, GPU cache storage, custom ops,
and attention backend should be added only when their Gate begins.

## Implemented mechanisms

- guarded Request lifecycle with explicit admission, completion, cancellation,
  failure, and preemption operations;
- transactional BlockPool reservations and logical KV block tables;
- immutable token-budget ScheduledBatch values;
- deterministic FakeExecutor execution and result application;
- chunked prefill with sampling only on the final chunk;
- persistent FCFS and decode-first policies;
- continuous admission and cache-pressure preemption.

The project prioritizes end-to-end happy paths for common inference-engine
mechanisms. Gate 3 fairness aging, randomized stress traces, production error
classification, and control-plane benchmarks remain non-blocking hardening
work. Gate 4 begins the GPU block-based KV cache.

## Development

```bash
cd ~/python/aiinfra/projects/einf
source ~/python/.venv311/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

## Collaboration boundary

The learner owns the core state machine, ownership transfers, failure semantics,
scheduling decisions, and benchmark interpretation. AI may prepare scaffolding,
fixtures, repetitive tests, counterexamples, reviews, and result organization.
