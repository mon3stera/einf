# Rust Control-Plane Migration Boundary

The current Python control plane is a correctness baseline. The intended
migration replaces it with one Rust control-plane owner without moving the
PyTorch model executor or CUDA storage.

## Ownership

Rust will own:

- request lifecycle and generated-token state;
- scheduler policy, admission, preemption, and token budgets;
- logical request-to-physical-block tables;
- block allocation transactions and writer/sealed state;
- prefix-cache lookup and eviction;
- validation of execution results and publication of completed state.

Python will own:

- model weights and Qwen/Hugging Face execution;
- `torch.Tensor` objects and `TorchKVCacheStorage`;
- `ModelInput` tensor materialization;
- PyTorch custom-op dispatch and C++/CUDA kernels;
- sampling implementation until it is deliberately moved.

There must be one lifecycle writer. During migration, that writer is the
existing Python `Scheduler`. After the Rust implementation is ready, the
Python scheduler must not run in parallel with it.

## Stable boundary values

`src/einf/execution_plan.py` defines the language-neutral logical contract:

- `BatchPlan`: one immutable execution step;
- `ScheduledRequest`: request ID, input token IDs, work type, absolute start
  position, physical block table, and whether the executor must sample;
- `WorkType`: prefill or decode.

`ScheduledBatch` remains an alias for compatibility with the current Python
baseline. A Rust/PyO3 implementation should expose objects with the same
read-only attributes, or an adapter should convert them once at the boundary.

`src/einf/execution.py` defines the result contract:

- `ExecutionResult.step_id` must identify the submitted plan;
- each `RequestExecutionResult` contains exactly one result for a scheduled
  request;
- `cached_len_delta` is the number of input tokens whose KV was written;
- generated token IDs and EOS are returned, but the executor does not mutate
  Request or block state.

The control plane must validate step ID, request IDs, duplicate/missing
results, and lifecycle state before applying a result.

## Torch-side flow

The intended call sequence is:

```text
plan = control.schedule()
result = torch_executor.execute(plan)
control.apply_result(result)
```

`TorchExecutor` and `ModelInput` consume `BatchPlan`; they do not import or
modify `Request`, `BlockPool`, or `KVCacheManager`. `ModelInput.from_plan()` is
the preferred constructor. `from_batch()` remains a compatibility alias while
old tests and callers are migrated.

The Rust plan must already contain the final logical block-table order:

```text
cached immutable prefix blocks + private writable suffix blocks
```

The Python side must only derive tensor metadata from that plan. It must not
perform prefix lookup, allocate blocks, change `cached_len`, or reorder the
block table.

`slot_mapping` must contain only destinations for input tokens in the current
step. It must never write into an immutable cached block.

## Rust implementation order

1. Reimplement pure Rust `Request`, `BlockPool`, and `Scheduler` with unit and
   property tests; do not connect PyTorch yet.
2. Make the Rust scheduler emit `BatchPlan` values and accept
   `ExecutionResult` values.
3. Add a thin PyO3/maturin wrapper or an explicit Python adapter.
4. Run the Rust control plane with the existing `TorchExecutor` and compare
   every plan/result trace against the Python baseline.
5. Remove the Python scheduler only after parity and ownership tests pass.

Prefix Cache is intentionally not part of this migration baseline. The broken
experimental Python Prefix Cache changes were removed. Add it in Rust only
after the baseline block-table and execution-result protocol is stable.
