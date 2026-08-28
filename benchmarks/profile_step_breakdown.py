"""Per-step time decomposition for the einf serving loop (Gate 6.0 scoreboard).

Splits one scheduler step into its phases without editing production code: this
harness owns the outer loop (mirroring ``benchmark_qwen_serving.run_step``) and
temporarily wraps a small set of Python entry points for the duration of the run.

Phases
    sched             Scheduler.schedule()          Rust + PyO3 boundary
    input_build       ModelInput.from_batch()       Python packing + H2D
      input_h2d       torch.tensor(...) within it   host->device copies
      input_cpu       input_build - input_h2d       pure Python packing
    forward           model_runner.forward()        model execution
    sampler           Sampler.sample()              sampling kernels
    sampler_meta_h2d  torch.tensor(...) in execute  per-request metadata copies
    d2h               Tensor.cpu() in execute       device->host + implicit sync
    exec_residual     execute minus the above       result objects, Python glue
    sync_wait         torch.cuda.synchronize()      GPU tail the harness waits on
    apply_result      Scheduler.apply_result()      Rust + PyO3 boundary
    step_residual     step minus all of the above

``input_h2d`` and ``input_cpu`` are a breakdown *inside* ``input_build`` and are
excluded from the additive total; every other phase is additive.

Timing modes
    wall  (default) no added synchronisation. Phase times are CPU-side
          occupancy, so GPU work overlaps and its cost surfaces in whichever
          phase finally blocks on the device -- normally ``d2h``.
    sync  torch.cuda.synchronize() at the end of each coarse phase, so a phase
          includes its own GPU work. Totals are inflated relative to ``wall``;
          use it to attribute device time, not to quote step latency.

Usage
    python benchmarks/profile_step_breakdown.py --concurrency 16 \
        --prompt-lens 512 --output-lens 64 --steps 200 --profiler-steps 5
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch import QwenConfig, QwenModelRunner, TorchExecutor
from einf.executors.torch.input import ModelInput
from einf.executors.torch.sampler import Sampler
from einf.request import RequestSpec, RequestState
from einf.scheduler import Scheduler, WorkType


DEFAULT_MODEL_DIR = Path(
    os.environ.get("EINF_MODEL_DIR", "~/models/Qwen2.5-0.5B")
).expanduser()
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

ADDITIVE_PHASES = (
    "sched",
    "input_build",
    "forward",
    "sampler",
    "sampler_meta_h2d",
    "d2h",
    "exec_residual",
    "sync_wait",
    "apply_result",
    "step_residual",
)
BREAKDOWN_PHASES = ("input_h2d", "input_cpu")


class Recorder:
    """Accumulates per-phase durations for the step currently being timed."""

    def __init__(self, *, sync: bool) -> None:
        self.sync = sync
        self.stack: list[str] = []
        self.times: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def inside(self, name: str) -> bool:
        return name in self.stack

    def add(self, name: str, ms: float) -> None:
        self.times[name] += ms

    def bump(self, name: str) -> None:
        self.counts[name] += 1

    @contextmanager
    def phase(self, name: str, *, sync: bool | None = None):
        self.stack.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync if sync is None else sync:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.stack.pop()
            self.times[name] += elapsed_ms


@contextmanager
def instrumented(recorder: Recorder, runner) -> None:
    """Install phase wrappers, and restore the originals on the way out."""

    original_from_batch = ModelInput.from_batch
    original_sampler_sample = Sampler.sample
    original_torch_tensor = torch.tensor
    original_tensor_cpu = torch.Tensor.cpu
    original_runner_forward = runner.forward

    def wrapped_from_batch(*args, **kwargs):
        with recorder.phase("input_build"):
            return original_from_batch(*args, **kwargs)

    def wrapped_forward(*args, **kwargs):
        with recorder.phase("forward"):
            return original_runner_forward(*args, **kwargs)

    def wrapped_sample(self, *args, **kwargs):
        with recorder.phase("sampler"):
            return original_sampler_sample(self, *args, **kwargs)

    def wrapped_torch_tensor(*args, **kwargs):
        start = time.perf_counter()
        result = original_torch_tensor(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if recorder.inside("input_build"):
            recorder.add("input_h2d", elapsed_ms)
            recorder.bump("input_h2d_calls")
        elif recorder.inside("sampler") or recorder.inside("forward"):
            pass
        elif recorder.inside("execute"):
            recorder.add("sampler_meta_h2d", elapsed_ms)
            recorder.bump("sampler_meta_calls")
        return result

    def wrapped_tensor_cpu(self, *args, **kwargs):
        if recorder.inside("sampler") or recorder.inside("forward"):
            return original_tensor_cpu(self, *args, **kwargs)
        if not recorder.inside("execute"):
            return original_tensor_cpu(self, *args, **kwargs)
        start = time.perf_counter()
        result = original_tensor_cpu(self, *args, **kwargs)
        recorder.add("d2h", (time.perf_counter() - start) * 1000.0)
        recorder.bump("d2h_calls")
        return result

    ModelInput.from_batch = staticmethod(wrapped_from_batch)
    Sampler.sample = wrapped_sample
    torch.tensor = wrapped_torch_tensor
    torch.Tensor.cpu = wrapped_tensor_cpu
    runner.forward = wrapped_forward
    try:
        yield
    finally:
        ModelInput.from_batch = original_from_batch
        Sampler.sample = original_sampler_sample
        torch.tensor = original_torch_tensor
        torch.Tensor.cpu = original_tensor_cpu
        try:
            del runner.forward
        except AttributeError:
            runner.forward = original_runner_forward


def parse_int_tuple(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise SystemExit(f"{name} must be a comma-separated list of positive ints")
    return values


def percentile(samples: list[float], fraction: float) -> float:
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose einf step latency into its phases"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-lens", default="512")
    parser.add_argument("--output-lens", default="64")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--max-batch-len", type=int, default=512)
    parser.add_argument("--max-prefill-chunk-len", type=int, default=128)
    parser.add_argument("--decode-backend", choices=("eager", "paged"), default="paged")
    parser.add_argument("--paged-decode-max-splits", type=int, default=64)
    parser.add_argument("--timing", choices=("wall", "sync"), default="wall")
    parser.add_argument(
        "--profiler-steps",
        type=int,
        default=0,
        help="run this many extra steps under torch.profiler to count launches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.concurrency <= 0 or args.steps <= 0:
        raise SystemExit("--concurrency and --steps must be positive")

    prompt_lens = parse_int_tuple(args.prompt_lens, name="--prompt-lens")
    output_lens = parse_int_tuple(args.output_lens, name="--output-lens")
    model_dir = args.model_dir.expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise SystemExit(
            f"no config.json under {model_dir} -- pass --model-dir or set EINF_MODEL_DIR"
        )
    config = QwenConfig.from_json(model_dir / "config.json")

    blocks_per_request = math.ceil(
        (max(prompt_lens) + max(output_lens)) / args.block_len
    )
    num_blocks = args.num_blocks or args.concurrency * blocks_per_request + args.concurrency

    dtype = DTYPES[args.dtype]
    device = torch.device("cuda")
    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks,
            block_len=args.block_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=dtype,
        device=device,
        use_custom_ops=True,
    )
    runner = (
        QwenModelRunner(
            config,
            cache=cache,
            use_paged_decode_attention=args.decode_backend == "paged",
            paged_decode_max_splits=args.paged_decode_max_splits,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    runner.load_checkpoint(model_dir / "model.safetensors")
    torch.cuda.synchronize()

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
    active: set[str] = set()
    next_index = 0

    def submit() -> None:
        nonlocal next_index
        request_id = f"request-{next_index}"
        next_index += 1
        prompt_len = rng.choice(prompt_lens)
        scheduler.submit(
            RequestSpec(
                request_id=request_id,
                prompt_token_ids=tuple(
                    rng.randrange(3, config.vocab_size) for _ in range(prompt_len)
                ),
                max_new_len=rng.choice(output_lens),
            )
        )
        active.add(request_id)

    recorder = Recorder(sync=args.timing == "sync")
    per_class: dict[str, dict[str, list[float]]] = {
        "decode_only": defaultdict(list),
        "with_prefill": defaultdict(list),
    }
    per_class_counts: dict[str, dict[str, list[int]]] = {
        "decode_only": defaultdict(list),
        "with_prefill": defaultdict(list),
    }
    batch_shape: dict[str, list[tuple[int, int]]] = {
        "decode_only": [],
        "with_prefill": [],
    }

    def run_step(collect: bool) -> None:
        if collect:
            recorder.reset()
        step_start = time.perf_counter()

        with recorder.phase("sched", sync=False):
            batch = scheduler.schedule()
        if batch is None:
            raise RuntimeError("scheduler became idle with active requests")

        with recorder.phase("execute", sync=False):
            execute_start = time.perf_counter()
            result = executor.execute(batch)
            execute_ms = (time.perf_counter() - execute_start) * 1000.0

        with recorder.phase("sync_wait", sync=False):
            torch.cuda.synchronize()

        with recorder.phase("apply_result", sync=False):
            scheduler.apply_result(result)

        step_ms = (time.perf_counter() - step_start) * 1000.0

        for scheduled in batch.requests:
            request = scheduler.request(scheduled.request_id)
            if request.state in (
                RequestState.FINISHED,
                RequestState.FAILED,
                RequestState.CANCELLED,
            ):
                active.discard(scheduled.request_id)
        while len(active) < args.concurrency:
            submit()

        if not collect:
            return

        prefill_requests = sum(
            scheduled.work_type is WorkType.PREFILL for scheduled in batch.requests
        )
        scheduled_tokens = sum(
            len(scheduled.input_token_ids) for scheduled in batch.requests
        )
        label = "with_prefill" if prefill_requests else "decode_only"
        times = recorder.times

        inner = (
            times["input_build"]
            + times["forward"]
            + times["sampler"]
            + times["sampler_meta_h2d"]
            + times["d2h"]
        )
        times["exec_residual"] = max(0.0, execute_ms - inner)
        times["input_cpu"] = max(0.0, times["input_build"] - times["input_h2d"])
        accounted = sum(times[name] for name in ADDITIVE_PHASES if name != "step_residual")
        times["step_residual"] = max(0.0, step_ms - accounted)

        bucket = per_class[label]
        bucket["step"].append(step_ms)
        for name in ADDITIVE_PHASES + BREAKDOWN_PHASES:
            bucket[name].append(times[name])
        for name, value in recorder.counts.items():
            per_class_counts[label][name].append(value)
        batch_shape[label].append((len(batch.requests), scheduled_tokens))

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(f"model: {model_dir}")
    print(f"decode_backend: {args.decode_backend}, timing_mode: {args.timing}")
    print(
        f"concurrency={args.concurrency}, prompt_lens={prompt_lens}, "
        f"output_lens={output_lens}, chunk={args.max_prefill_chunk_len}, "
        f"max_batch_len={args.max_batch_len}, num_blocks={num_blocks}, "
        f"warmup_steps={args.warmup_steps}, steps={args.steps}, seed={args.seed}"
    )

    for _ in range(args.concurrency):
        submit()

    with torch.inference_mode(), instrumented(recorder, runner):
        for _ in range(args.warmup_steps):
            run_step(collect=False)
        torch.cuda.synchronize()
        for _ in range(args.steps):
            run_step(collect=True)
        torch.cuda.synchronize()

    for label in ("decode_only", "with_prefill"):
        samples = per_class[label]
        if not samples["step"]:
            continue
        steps = len(samples["step"])
        shapes = batch_shape[label]
        mean_requests = statistics.fmean(shape[0] for shape in shapes)
        mean_tokens = statistics.fmean(shape[1] for shape in shapes)
        mean_step = statistics.fmean(samples["step"])
        print()
        print(f"=== {label}: {steps} steps ===")
        print(
            f"mean_requests_per_step: {mean_requests:.2f}, "
            f"mean_scheduled_tokens: {mean_tokens:.2f}"
        )
        print(f"{'phase':<18}{'mean_ms':>10}{'p50_ms':>10}{'p95_ms':>10}{'share':>9}")
        for name in ADDITIVE_PHASES:
            values = samples[name]
            mean_ms = statistics.fmean(values)
            share = 100.0 * mean_ms / mean_step if mean_step else float("nan")
            print(
                f"{name:<18}{mean_ms:>10.3f}{percentile(values, 0.50):>10.3f}"
                f"{percentile(values, 0.95):>10.3f}{share:>8.1f}%"
            )
        print(f"{'step (total)':<18}{mean_step:>10.3f}"
              f"{percentile(samples['step'], 0.50):>10.3f}"
              f"{percentile(samples['step'], 0.95):>10.3f}{100.0:>8.1f}%")
        print("  breakdown inside input_build (not additive):")
        for name in BREAKDOWN_PHASES:
            values = samples[name]
            print(
                f"  {name:<16}{statistics.fmean(values):>10.3f}"
                f"{percentile(values, 0.50):>10.3f}{percentile(values, 0.95):>10.3f}"
            )
        counts = per_class_counts[label]
        if counts:
            summary = ", ".join(
                f"{name}={statistics.fmean(values):.1f}"
                for name, values in sorted(counts.items())
            )
            print(f"  per-step call counts: {summary}")

    if args.profiler_steps > 0:
        print()
        print(f"=== torch.profiler over {args.profiler_steps} steps ===")
        try:
            from torch.profiler import ProfilerActivity, profile

            with torch.inference_mode():
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as prof:
                    for _ in range(args.profiler_steps):
                        run_step(collect=False)
                    torch.cuda.synchronize()

            def device_time(event) -> float:
                for attribute in ("self_device_time_total", "self_cuda_time_total"):
                    value = getattr(event, attribute, 0.0) or 0.0
                    if value:
                        return float(value)
                return 0.0

            rows = list(prof.key_averages())

            def is_device_row(row) -> bool:
                device_type = getattr(row, "device_type", None)
                if device_type is not None:
                    return str(device_type).upper().endswith("CUDA")
                return row.key.startswith(("void ", "Memcpy", "Memset"))

            kernel_rows = [row for row in rows if is_device_row(row)]
            op_rows = [row for row in rows if not is_device_row(row)]
            launches = sum(row.count for row in kernel_rows)
            device_us = sum(device_time(row) for row in kernel_rows)
            device_ms_per_step = device_us / 1000.0 / args.profiler_steps
            # aten::item and aten::_local_scalar_dense are two rows for the same
            # scalar read, so counting both would double the sync count.
            sync_ops = sum(
                row.count
                for row in op_rows
                if row.key == "aten::_local_scalar_dense"
            )
            dtoh_copies = sum(
                row.count for row in kernel_rows if row.key.startswith("Memcpy DtoH")
            )
            reference_step = None
            for label in ("decode_only", "with_prefill"):
                if per_class[label]["step"]:
                    reference_step = statistics.fmean(per_class[label]["step"])
                    break

            print(
                f"kernel/memcpy launches: {launches} total, "
                f"{launches / args.profiler_steps:.1f} per step"
            )
            print(
                f"device time (kernel rows only): {device_us / 1000.0:.2f} ms total, "
                f"{device_ms_per_step:.3f} ms per step"
            )
            if reference_step:
                print(
                    f"device busy share of the measured step: "
                    f"{100.0 * device_ms_per_step / reference_step:.1f}% "
                    f"(step {reference_step:.3f} ms)"
                )
            print(
                f"blocking device->host scalar reads per step "
                f"(aten::_local_scalar_dense): {sync_ops / args.profiler_steps:.1f}"
            )
            print(f"Memcpy DtoH per step: {dtoh_copies / args.profiler_steps:.1f}")
            print()
            print(f"{'kernel / memcpy':<44}{'count':>8}{'per_step':>10}{'dev_ms':>10}")
            for row in sorted(kernel_rows, key=lambda row: row.count, reverse=True)[:12]:
                print(
                    f"{row.key[:43]:<44}{row.count:>8}"
                    f"{row.count / args.profiler_steps:>10.1f}"
                    f"{device_time(row) / 1000.0:>10.3f}"
                )
            print()
            print(
                f"{'host-side op':<44}{'count':>8}{'per_step':>10}"
                f"{'cpu_ms/step':>13}"
            )

            def cpu_time(event) -> float:
                for attribute in ("self_cpu_time_total", "self_cpu_time"):
                    value = getattr(event, attribute, 0.0) or 0.0
                    if value:
                        return float(value)
                return 0.0

            for row in sorted(op_rows, key=cpu_time, reverse=True)[:14]:
                print(
                    f"{row.key[:43]:<44}{row.count:>8}"
                    f"{row.count / args.profiler_steps:>10.1f}"
                    f"{cpu_time(row) / 1000.0 / args.profiler_steps:>13.3f}"
                )
        except Exception as error:  # pragma: no cover - diagnostic path only
            print(f"profiler section failed: {type(error).__name__}: {error}")

    print()
    print(
        "wall mode reports CPU-side occupancy; GPU work surfaces in the phase that "
        "blocks on the device. Re-run with --timing sync to attribute device time."
    )


if __name__ == "__main__":
    main()
