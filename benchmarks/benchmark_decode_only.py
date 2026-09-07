"""Fixed-batch decode-only throughput for einf, with an optional vLLM A/B.

Prefill is drained first and never counted. The measured window keeps the same
``concurrency`` running requests alive, so every step is decode-only.

vLLM uses offline ``LLM.generate`` on the same prompt/batch/greedy settings.
Its decode window is ``first_token_ts → last_token_ts`` (TTFT/prefill excluded).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path(
    os.environ.get("EINF_MODEL_DIR", "~/models/Qwen2.5-0.5B")
).expanduser()


def parse_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
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
        description="Measure decode-only throughput at a fixed batch"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-lens", default="512")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
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
    parser.add_argument(
        "--engine",
        choices=("einf", "vllm", "both"),
        default="einf",
        help="einf, vLLM, or both in isolated subprocesses",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a JSON stats line after the human report",
    )
    return parser.parse_args()


def print_stats(stats: dict[str, Any]) -> None:
    print()
    print(f"{stats['engine']} decode-only results")
    for key in (
        "prefill_drain_steps",
        "measurement_s",
        "output_tokens",
        "output_tokens_per_second",
        "steps",
        "batch",
        "step_latency_ms_mean",
        "step_latency_ms_p50",
        "step_latency_ms_p95",
        "inter_token_ms_p50",
        "inter_token_ms_p95",
        "mixed_step_fraction",
        "cuda_peak_allocated_mib",
        "notes",
    ):
        if key not in stats:
            continue
        value = stats[key]
        if isinstance(value, float):
            if key.endswith("_mib") or key.endswith("_second") or key == "measurement_s":
                print(f"{key}: {value:.3f}" if key != "output_tokens_per_second" else f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    print()
    print("Prefill is excluded. Concurrency is held fixed; no request is replaced.")


def print_comparison(einf_stats: dict[str, Any], vllm_stats: dict[str, Any]) -> None:
    def ratio(left: float, right: float) -> str:
        if not right:
            return "n/a"
        return f"{left / right:.3f}x"

    print()
    print("decode-only comparison (same host, prefill excluded)")
    print(f"{'metric':<28}{'einf':>12}{'vLLM':>12}{'einf/vLLM':>12}")
    rows = (
        ("output_tokens_per_second", "tok/s"),
        ("step_latency_ms_mean", "step mean ms"),
        ("step_latency_ms_p50", "step p50 ms"),
        ("step_latency_ms_p95", "step p95 ms"),
        ("inter_token_ms_p50", "ITL p50 ms"),
        ("inter_token_ms_p95", "ITL p95 ms"),
    )
    for key, label in rows:
        left = float(einf_stats[key])
        right = float(vllm_stats[key])
        print(f"{label:<28}{left:>12.3f}{right:>12.3f}{ratio(left, right):>12}")


def run_einf(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
    from einf.executors.torch import QwenConfig, QwenModelRunner, TorchExecutor
    from einf.request import RequestSpec, RequestState
    from einf.scheduler import Scheduler, WorkType

    dtypes = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if args.max_batch_len < args.concurrency:
        raise SystemExit("--max-batch-len must be >= --concurrency so decode stays one batch")
    if args.max_batch_len < args.max_prefill_chunk_len:
        raise SystemExit("--max-batch-len must be >= --max-prefill-chunk-len")

    prompt_lens = parse_int_tuple(args.prompt_lens, name="--prompt-lens")
    model_dir = args.model_dir.expanduser().resolve()
    config = QwenConfig.from_json(model_dir / "config.json")
    decode_tokens_needed = args.warmup_steps + args.steps + max(prompt_lens)
    if max(prompt_lens) + decode_tokens_needed > config.max_position_embeddings:
        raise SystemExit(
            "prompt + warmup + measured decode exceeds the model context limit "
            f"of {config.max_position_embeddings}"
        )

    blocks_per_request = math.ceil(
        (max(prompt_lens) + decode_tokens_needed) / args.block_len
    )
    required_blocks = args.concurrency * blocks_per_request
    num_blocks = args.num_blocks or required_blocks + args.concurrency
    if num_blocks < required_blocks:
        raise SystemExit(
            f"--num-blocks must be at least {required_blocks} for this workload"
        )

    dtype = dtypes[args.dtype]
    device = torch.device("cuda")
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
    )
    runner = (
        QwenModelRunner(
            config,
            cache=cache,
            use_paged_decode_attention=args.decode_backend == "paged",
            use_flashinfer_attention=args.decode_backend == "flashinfer",
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
    prompt_len_by_id: dict[str, int] = {}

    def submit() -> None:
        request_id = f"request-{len(prompt_len_by_id)}"
        prompt_len = rng.choice(prompt_lens)
        scheduler.submit(
            RequestSpec(
                request_id=request_id,
                prompt_token_ids=tuple(
                    rng.randrange(3, config.vocab_size) for _ in range(prompt_len)
                ),
                max_new_len=decode_tokens_needed,
            )
        )
        prompt_len_by_id[request_id] = prompt_len

    def all_in_decode() -> bool:
        return all(
            scheduler.request(request_id).cached_len >= prompt_len
            for request_id, prompt_len in prompt_len_by_id.items()
        )

    def run_step() -> tuple[bool, int, float]:
        step_start = time.perf_counter()
        batch = scheduler.schedule()
        if batch is None:
            raise RuntimeError("scheduler became idle with active requests")
        result = executor.execute(batch)
        torch.cuda.synchronize()
        scheduler.apply_result(result)
        step_ms = (time.perf_counter() - step_start) * 1000.0

        prefill_requests = sum(
            scheduled.work_type is WorkType.PREFILL for scheduled in batch.requests
        )
        decode_tokens = sum(
            len(scheduled.input_token_ids)
            for scheduled in batch.requests
            if scheduled.work_type is WorkType.DECODE
        )
        for scheduled in batch.requests:
            request = scheduler.request(scheduled.request_id)
            if request.state is RequestState.FAILED:
                raise RuntimeError(request.error)
            if request.state in (
                RequestState.FINISHED,
                RequestState.CANCELLED,
            ):
                raise RuntimeError(
                    f"{scheduled.request_id} left decode while the benchmark was running"
                )
        return prefill_requests == 0, decode_tokens, step_ms

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(f"model: {model_dir}")
    print(f"engine: einf, decode_backend: {args.decode_backend}")
    print(
        f"concurrency={args.concurrency}, prompt_lens={prompt_lens}, "
        f"chunk={args.max_prefill_chunk_len}, max_batch_len={args.max_batch_len}, "
        f"num_blocks={num_blocks}, warmup_steps={args.warmup_steps}, "
        f"steps={args.steps}, seed={args.seed}"
    )

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.concurrency):
            submit()

        prefill_steps = 0
        while not all_in_decode():
            run_step()
            prefill_steps += 1

        for _ in range(args.warmup_steps):
            decode_only, decode_tokens, _ = run_step()
            if not decode_only or decode_tokens != args.concurrency:
                raise RuntimeError(
                    "warmup step was not a full-batch decode "
                    f"(decode_only={decode_only}, tokens={decode_tokens})"
                )

        torch.cuda.synchronize()
        measure_start = time.perf_counter()
        step_ms: list[float] = []
        output_tokens = 0
        for _ in range(args.steps):
            decode_only, decode_tokens, latency_ms = run_step()
            if not decode_only or decode_tokens != args.concurrency:
                raise RuntimeError(
                    "measured step was not a full-batch decode "
                    f"(decode_only={decode_only}, tokens={decode_tokens})"
                )
            step_ms.append(latency_ms)
            output_tokens += decode_tokens
        torch.cuda.synchronize()
        measure_end = time.perf_counter()

    elapsed = measure_end - measure_start
    return {
        "engine": "einf",
        "prefill_drain_steps": prefill_steps,
        "measurement_s": elapsed,
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed,
        "steps": args.steps,
        "batch": args.concurrency,
        "step_latency_ms_mean": sum(step_ms) / len(step_ms),
        "step_latency_ms_p50": percentile(step_ms, 0.50),
        "step_latency_ms_p95": percentile(step_ms, 0.95),
        "inter_token_ms_p50": percentile(step_ms, 0.50),
        "inter_token_ms_p95": percentile(step_ms, 0.95),
        "mixed_step_fraction": 0.0,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "notes": "wall step includes schedule/execute/apply_result; CUDA sync after execute",
    }


def run_vllm(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    prompt_lens = parse_int_tuple(args.prompt_lens, name="--prompt-lens")
    model_dir = args.model_dir.expanduser().resolve()
    hf_config = json.loads((model_dir / "config.json").read_text())
    vocab_size = int(hf_config["vocab_size"])
    max_model_len = min(
        int(hf_config["max_position_embeddings"]),
        max(prompt_lens) + args.warmup_steps + args.steps + 8,
    )
    rng = random.Random(args.seed)
    prompts = [
        TokensPrompt(
            prompt_token_ids=[
                rng.randrange(3, vocab_size) for _ in range(rng.choice(prompt_lens))
            ]
        )
        for _ in range(args.concurrency)
    ]

    print(f"model: {model_dir}")
    print(f"engine: vllm, dtype: {args.dtype}")
    print(
        f"concurrency={args.concurrency}, prompt_lens={prompt_lens}, "
        f"max_num_batched_tokens={args.max_batch_len}, "
        f"warmup_steps={args.warmup_steps}, steps={args.steps}, seed={args.seed}"
    )

    llm = LLM(
        model=str(model_dir),
        tokenizer=str(model_dir),
        dtype=args.dtype,
        max_model_len=max_model_len,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.max_batch_len,
        gpu_memory_utilization=0.8,
        enforce_eager=False,
        enable_prefix_caching=False,
        disable_log_stats=False,
    )
    warmup = SamplingParams(
        temperature=0.0,
        max_tokens=max(args.warmup_steps, 1),
        ignore_eos=True,
        seed=args.seed,
    )
    # +1 so first_token→last_token covers `steps` decode intervals after TTFT.
    measured = SamplingParams(
        temperature=0.0,
        max_tokens=args.steps + 1,
        ignore_eos=True,
        seed=args.seed,
    )
    print("vLLM warmup generate")
    llm.generate(prompts, warmup, use_tqdm=False)

    print("vLLM measured generate")
    outputs = llm.generate(prompts, measured, use_tqdm=False)
    if any(item.metrics is None for item in outputs):
        raise RuntimeError("vLLM RequestOutput.metrics is missing; cannot isolate decode")

    first_ts = [item.metrics.first_token_ts for item in outputs]
    last_ts = [item.metrics.last_token_ts for item in outputs]
    generated = [item.metrics.num_generation_tokens for item in outputs]
    if min(first_ts) <= 0.0 or min(last_ts) <= min(first_ts):
        raise RuntimeError("vLLM decode timestamps were not populated")
    if min(generated) < args.steps + 1:
        raise RuntimeError(
            f"vLLM generated {min(generated)} tokens, expected {args.steps + 1}"
        )

    # Same batch: decode window is the union of first→last across the 8 requests.
    decode_s = max(last_ts) - min(first_ts)
    decode_tokens = args.concurrency * args.steps
    itl_ms = [
        1000.0 * (last - first) / args.steps
        for first, last in zip(first_ts, last_ts)
    ]
    step_ms = 1000.0 * decode_s / args.steps
    return {
        "engine": "vllm",
        "prefill_drain_steps": None,
        "measurement_s": decode_s,
        "output_tokens": decode_tokens,
        "output_tokens_per_second": decode_tokens / decode_s,
        "steps": args.steps,
        "batch": args.concurrency,
        "step_latency_ms_mean": step_ms,
        "step_latency_ms_p50": step_ms,
        "step_latency_ms_p95": step_ms,
        "inter_token_ms_p50": percentile(itl_ms, 0.50),
        "inter_token_ms_p95": percentile(itl_ms, 0.95),
        "mixed_step_fraction": 0.0,
        "notes": (
            "decode window is first_token_ts→last_token_ts; "
            "first generated token (prefill) is excluded"
        ),
    }


def parse_json_line(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    raise RuntimeError("child benchmark did not print a JSON stats line")


def run_child(engine: str, args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--engine",
        engine,
        "--json",
        "--model-dir",
        str(args.model_dir.expanduser().resolve()),
        "--prompt-lens",
        args.prompt_lens,
        "--concurrency",
        str(args.concurrency),
        "--warmup-steps",
        str(args.warmup_steps),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--dtype",
        args.dtype,
        "--block-len",
        str(args.block_len),
        "--num-blocks",
        str(args.num_blocks),
        "--max-batch-len",
        str(args.max_batch_len),
        "--max-prefill-chunk-len",
        str(args.max_prefill_chunk_len),
        "--decode-backend",
        args.decode_backend,
        "--paged-decode-max-splits",
        str(args.paged_decode_max_splits),
    ]
    env = os.environ.copy()
    if engine == "vllm":
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    print(f"\n===== {engine} =====")
    completed = subprocess.run(
        command,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stdout.write(completed.stdout)
    sys.stdout.flush()
    if completed.returncode != 0:
        raise SystemExit(f"{engine} decode-only benchmark failed with {completed.returncode}")
    return parse_json_line(completed.stdout)


def main() -> None:
    args = parse_args()
    if args.concurrency <= 0 or args.steps <= 0 or args.warmup_steps < 0:
        raise SystemExit("concurrency/steps must be positive; warmup-steps >= 0")

    if args.engine == "both":
        einf_stats = run_child("einf", args)
        vllm_stats = run_child("vllm", args)
        print_comparison(einf_stats, vllm_stats)
        if args.json:
            print(
                "JSON:"
                + json.dumps({"einf": einf_stats, "vllm": vllm_stats}, sort_keys=True)
            )
        return

    stats = run_einf(args) if args.engine == "einf" else run_vllm(args)
    print_stats(stats)
    if args.json:
        print("JSON:" + json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
