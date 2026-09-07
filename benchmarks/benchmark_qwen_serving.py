from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch import QwenConfig, QwenModelRunner, TorchExecutor
from einf.request import RequestSpec, RequestState
from einf.scheduler import Scheduler, WorkType


DEFAULT_MODEL_DIR = Path(
    os.environ.get("EINF_MODEL_DIR", "~/models/Qwen2.5-0.5B")
).expanduser()
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
KV_DTYPES = {
    **DTYPES,
    "fp8_e4m3": torch.float8_e4m3fn,
}


@dataclass(slots=True)
class RequestTrace:
    request_id: str
    prompt_len: int
    output_len: int
    submitted_at: float
    measured: bool
    token_times: list[float] = field(default_factory=list)
    completed_at: float | None = None


@dataclass(slots=True)
class WindowStats:
    output_tokens: int = 0
    completed_requests: int = 0
    steps: int = 0
    scheduled_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    scheduled_requests: int = 0
    prefill_requests: int = 0
    decode_requests: int = 0
    mixed_steps: int = 0
    step_latencies_ms: list[float] = field(default_factory=list)


def parse_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a closed-loop steady-state Qwen serving benchmark"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-lens", default="128,512,2048")
    parser.add_argument("--output-lens", default="32,64,128")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-completions", type=int, default=8)
    parser.add_argument("--measured-completions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=KV_DTYPES,
        default="auto",
        help="KV cache storage dtype; auto follows --dtype. fp8_e4m3 requires "
        "--decode-backend=flashinfer",
    )
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--max-batch-len", type=int, default=512)
    parser.add_argument("--max-prefill-chunk-len", type=int, default=128)
    parser.add_argument(
        "--decode-backend",
        choices=("eager", "paged", "flashinfer"),
        default="paged",
    )
    parser.add_argument("--paged-decode-max-splits", type=int, default=64)
    parser.add_argument("--w4a16", action="store_true")
    parser.add_argument("--progress-every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.warmup_completions < 0 or args.measured_completions <= 0:
        raise SystemExit("completion counts must be warmup >= 0 and measured > 0")
    if args.max_batch_len < args.max_prefill_chunk_len:
        raise SystemExit("--max-batch-len must be >= --max-prefill-chunk-len")

    prompt_lens = parse_int_tuple(args.prompt_lens, name="--prompt-lens")
    output_lens = parse_int_tuple(args.output_lens, name="--output-lens")
    model_dir = args.model_dir.expanduser().resolve()
    config = QwenConfig.from_json(model_dir / "config.json")
    if max(prompt_lens) + max(output_lens) > config.max_position_embeddings:
        raise SystemExit(
            "the maximum prompt+output length exceeds the model context limit "
            f"of {config.max_position_embeddings}"
        )

    blocks_per_request = math.ceil(
        (max(prompt_lens) + max(output_lens)) / args.block_len
    )
    required_blocks = args.concurrency * blocks_per_request
    num_blocks = args.num_blocks or required_blocks + args.concurrency
    if num_blocks < required_blocks:
        raise SystemExit(
            f"--num-blocks must be at least {required_blocks} for this workload"
        )

    dtype = DTYPES[args.dtype]
    kv_dtype = dtype if args.kv_cache_dtype == "auto" else KV_DTYPES[args.kv_cache_dtype]
    if kv_dtype != dtype and args.decode_backend != "flashinfer":
        raise SystemExit("--kv-cache-dtype=fp8_e4m3 requires --decode-backend=flashinfer")
    device = torch.device("cuda")
    load_start = time.perf_counter()
    cache_blocks = num_blocks + int(args.decode_backend == "flashinfer")
    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=cache_blocks,
            block_len=args.block_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=dtype,
        device=device,
        use_custom_ops=True,
        kv_dtype=kv_dtype,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            runner = QwenModelRunner(
                config,
                cache=cache,
                use_paged_decode_attention=args.decode_backend == "paged",
                use_flashinfer_attention=args.decode_backend == "flashinfer",
                paged_decode_max_splits=args.paged_decode_max_splits,
                dtype=dtype,
                w4a16=args.w4a16,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    if args.w4a16:
        # Quantized on CPU masters; the runner moves itself to the device
        # inside load_checkpoint so the fp transient never reaches the GPU.
        runner = runner.eval()
    else:
        runner = runner.to(device=device, dtype=dtype).eval()
    runner.load_checkpoint(model_dir)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    scheduler = Scheduler(
        policy="fcfs",
        num_blocks=num_blocks,
        block_len=args.block_len,
        max_batch_len=args.max_batch_len,
        max_prefill_chunk_len=args.max_prefill_chunk_len,
    )
    executor = TorchExecutor(
        model_runner=runner,
        block_len=args.block_len,
        eos_token_id=-1,
        device=device,
    )
    rng = random.Random(args.seed)
    traces: dict[str, RequestTrace] = {}
    active_request_ids: set[str] = set()
    next_request_index = 0

    def submit_request(*, measured: bool) -> None:
        nonlocal next_request_index
        prompt_len = rng.choice(prompt_lens)
        output_len = rng.choice(output_lens)
        request_id = f"request-{next_request_index}"
        next_request_index += 1
        prompt_token_ids = tuple(
            rng.randrange(3, config.vocab_size)
            for _ in range(prompt_len)
        )
        submitted_at = time.perf_counter()
        scheduler.submit(
            RequestSpec(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                max_new_len=output_len,
            )
        )
        traces[request_id] = RequestTrace(
            request_id=request_id,
            prompt_len=prompt_len,
            output_len=output_len,
            submitted_at=submitted_at,
            measured=measured,
        )
        active_request_ids.add(request_id)

    def run_step(stats: WindowStats | None) -> list[str]:
        step_start = time.perf_counter()
        batch = scheduler.schedule()
        if batch is None:
            raise RuntimeError("scheduler became idle with active requests")
        result = executor.execute(batch)
        torch.cuda.synchronize()
        scheduler.apply_result(result)
        step_end = time.perf_counter()

        if stats is not None:
            prefill_requests = sum(
                request.work_type is WorkType.PREFILL
                for request in batch.requests
            )
            decode_requests = len(batch.requests) - prefill_requests
            prefill_tokens = sum(
                len(request.input_token_ids)
                for request in batch.requests
                if request.work_type is WorkType.PREFILL
            )
            decode_tokens = sum(
                len(request.input_token_ids)
                for request in batch.requests
                if request.work_type is WorkType.DECODE
            )
            stats.steps += 1
            stats.step_latencies_ms.append((step_end - step_start) * 1000)
            stats.scheduled_requests += len(batch.requests)
            stats.prefill_requests += prefill_requests
            stats.decode_requests += decode_requests
            stats.scheduled_tokens += prefill_tokens + decode_tokens
            stats.prefill_tokens += prefill_tokens
            stats.decode_tokens += decode_tokens
            stats.mixed_steps += int(prefill_requests > 0 and decode_requests > 0)

        for request_result in result.request_results:
            if request_result.generated_token_ids:
                traces[request_result.request_id].token_times.append(step_end)
                if stats is not None:
                    stats.output_tokens += len(request_result.generated_token_ids)

        completed = []
        for scheduled_request in batch.requests:
            request_id = scheduled_request.request_id
            request = scheduler.request(request_id)
            if request.state in (
                RequestState.FINISHED,
                RequestState.FAILED,
                RequestState.CANCELLED,
            ):
                trace = traces[request_id]
                trace.completed_at = step_end
                active_request_ids.remove(request_id)
                completed.append(request_id)
                if stats is not None:
                    stats.completed_requests += 1
                if request.state is RequestState.FAILED:
                    raise RuntimeError(request.error)
        return completed

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(f"model: {model_dir}")
    prefill_backend = (
        "flashinfer" if args.decode_backend == "flashinfer" else "sdpa"
    )
    print(f"decode_backend: {args.decode_backend}, prefill_backend: {prefill_backend}")
    print(
        f"concurrency={args.concurrency}, prompt_lens={prompt_lens}, "
        f"output_lens={output_lens}, chunk={args.max_prefill_chunk_len}, "
        f"max_batch_len={args.max_batch_len}, num_blocks={num_blocks}, "
        f"warmup_completions={args.warmup_completions}, "
        f"measured_completions={args.measured_completions}, seed={args.seed}"
    )
    print(f"model_startup_s: {load_seconds:.3f}")
    print(f"cuda_allocated_mib: {torch.cuda.memory_allocated() / 2**20:.1f}")

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        if args.warmup_completions == 0:
            measurement_start = time.perf_counter()
            for _ in range(args.concurrency):
                submit_request(measured=True)
        else:
            for _ in range(args.concurrency):
                submit_request(measured=False)
            warmup_completed = 0
            while warmup_completed < args.warmup_completions:
                completed = run_step(None)
                warmup_completed += len(completed)
                while len(active_request_ids) < args.concurrency:
                    submit_request(measured=False)
            torch.cuda.synchronize()
            measurement_start = time.perf_counter()

        stats = WindowStats()
        measured_completed = 0
        next_progress = args.progress_every
        while measured_completed < args.measured_completions:
            completed = run_step(stats)
            for request_id in completed:
                measured_completed += int(traces[request_id].measured)
            while (
                len(active_request_ids) < args.concurrency
                and measured_completed < args.measured_completions
            ):
                submit_request(measured=True)
            if args.progress_every > 0 and measured_completed >= next_progress:
                print(
                    f"progress: {measured_completed}/"
                    f"{args.measured_completions} measured requests"
                )
                while next_progress <= measured_completed:
                    next_progress += args.progress_every

        torch.cuda.synchronize()
        measurement_end = time.perf_counter()

    measured_traces = [
        trace
        for trace in traces.values()
        if trace.measured and trace.completed_at is not None
    ]
    ttft_ms = [
        (trace.token_times[0] - trace.submitted_at) * 1000
        for trace in measured_traces
    ]
    request_latency_ms = [
        (trace.completed_at - trace.submitted_at) * 1000
        for trace in measured_traces
        if trace.completed_at is not None
    ]
    inter_token_ms = [
        (current - previous) * 1000
        for trace in measured_traces
        for previous, current in zip(trace.token_times, trace.token_times[1:])
    ]
    elapsed_seconds = measurement_end - measurement_start

    print()
    print("steady-state results")
    print(f"measurement_s: {elapsed_seconds:.3f}")
    print(f"output_tokens: {stats.output_tokens}")
    print(f"output_tokens_per_second: {stats.output_tokens / elapsed_seconds:.2f}")
    print(f"completed_requests: {stats.completed_requests}")
    print(f"requests_per_second: {stats.completed_requests / elapsed_seconds:.3f}")
    print(f"measured_latency_samples: {len(measured_traces)}")
    print(
        f"TTFT_ms: p50={percentile(ttft_ms, 0.50):.3f}, "
        f"p95={percentile(ttft_ms, 0.95):.3f}"
    )
    print(
        f"inter_token_ms: p50={percentile(inter_token_ms, 0.50):.3f}, "
        f"p95={percentile(inter_token_ms, 0.95):.3f}"
    )
    print(
        f"request_latency_ms: p50={percentile(request_latency_ms, 0.50):.3f}, "
        f"p95={percentile(request_latency_ms, 0.95):.3f}"
    )
    print(f"steps: {stats.steps}")
    print(f"step_latency_ms_p50: {percentile(stats.step_latencies_ms, 0.50):.3f}")
    print(f"step_latency_ms_p95: {percentile(stats.step_latencies_ms, 0.95):.3f}")
    print(f"average_scheduled_tokens: {stats.scheduled_tokens / stats.steps:.2f}")
    print(f"average_requests_per_step: {stats.scheduled_requests / stats.steps:.2f}")
    print(f"average_prefill_requests_per_step: {stats.prefill_requests / stats.steps:.2f}")
    print(f"average_decode_requests_per_step: {stats.decode_requests / stats.steps:.2f}")
    print(f"mixed_step_fraction: {stats.mixed_steps / stats.steps:.3f}")
    print(f"prefill_tokens_per_second: {stats.prefill_tokens / elapsed_seconds:.2f}")
    print(f"decode_input_tokens_per_second: {stats.decode_tokens / elapsed_seconds:.2f}")
    print(f"cuda_peak_allocated_mib: {torch.cuda.max_memory_allocated() / 2**20:.1f}")
    print()
    print("Prompts are random token IDs and EOS is disabled for reproducibility.")
    print("The closed loop replaces completed requests to maintain concurrency.")


if __name__ == "__main__":
    main()
