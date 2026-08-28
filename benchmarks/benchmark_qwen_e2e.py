from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path

import torch
from transformers import AutoTokenizer

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch import QwenConfig, QwenModelRunner, TorchExecutor
from einf.lib import LLMServer
from einf.request import RequestSpec, RequestState
from einf.scheduler import Scheduler


DEFAULT_MODEL_DIR = Path(
    os.environ.get("EINF_MODEL_DIR", "~/models/Qwen2.5-0.5B")
).expanduser()
DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass(frozen=True, slots=True)
class RequestMetrics:
    prompt_len: int
    generated_len: int
    steps: int
    ttft_ms: float
    total_ms: float
    tpot_ms: float
    decode_tokens_per_second: float
    output_tokens_per_second: float
    effective_prefill_tokens_per_second: float


def parse_int_tuple(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-request Qwen2.5 inference through einf"
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-lens", default="16,128,512,2048")
    parser.add_argument("--max-new-len", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-repetitions", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--max-batch-len", type=int, default=0)
    parser.add_argument("--max-prefill-chunk-len", type=int, default=128)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--paged-decode-attention", action="store_true")
    parser.add_argument("--paged-decode-max-splits", type=int, default=64)
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help="stop at EOS instead of forcing exactly --max-new-len tokens",
    )
    return parser.parse_args()


def build_prompt(base_token_ids: tuple[int, ...], prompt_len: int) -> tuple[int, ...]:
    return tuple(islice(cycle(base_token_ids), prompt_len))


def run_request(
    server: LLMServer,
    scheduler: Scheduler,
    *,
    request_id: str,
    prompt_token_ids: tuple[int, ...],
    max_new_len: int,
) -> RequestMetrics:
    torch.cuda.synchronize()
    start = time.perf_counter()
    scheduler.submit(
        RequestSpec(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_len=max_new_len,
        )
    )

    first_token_time: float | None = None
    steps = 0
    while True:
        if not server.run_once():
            raise RuntimeError("server became idle before the request completed")
        torch.cuda.synchronize()
        steps += 1

        request = scheduler.request(request_id)
        if first_token_time is None and request.generated_token_ids:
            first_token_time = time.perf_counter()

        if request.state in (
            RequestState.FINISHED,
            RequestState.FAILED,
            RequestState.CANCELLED,
        ):
            break

    end = time.perf_counter()
    if request.state is RequestState.FAILED:
        raise RuntimeError(request.error)
    if request.state is not RequestState.FINISHED:
        raise RuntimeError(f"request ended in unexpected state {request.state}")
    if first_token_time is None:
        raise RuntimeError("request completed without producing a token")

    generated_len = len(request.generated_token_ids)
    ttft_seconds = first_token_time - start
    total_seconds = end - start
    decode_len = max(generated_len - 1, 0)
    decode_seconds = max(end - first_token_time, 0.0)
    tpot_seconds = decode_seconds / decode_len if decode_len else 0.0

    return RequestMetrics(
        prompt_len=len(prompt_token_ids),
        generated_len=generated_len,
        steps=steps,
        ttft_ms=ttft_seconds * 1000,
        total_ms=total_seconds * 1000,
        tpot_ms=tpot_seconds * 1000,
        decode_tokens_per_second=(
            decode_len / decode_seconds if decode_len and decode_seconds else math.inf
        ),
        output_tokens_per_second=generated_len / total_seconds,
        effective_prefill_tokens_per_second=len(prompt_token_ids) / ttft_seconds,
    )


def median(values: list[RequestMetrics], field: str) -> float:
    return statistics.median(getattr(value, field) for value in values)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.max_new_len <= 0:
        raise SystemExit("--max-new-len must be positive")
    if args.repetitions <= 0 or args.warmup_repetitions < 0:
        raise SystemExit("repetition counts must be non-negative and measured > 0")

    prompt_lens = parse_int_tuple(args.prompt_lens, name="--prompt-lens")
    model_dir = args.model_dir.expanduser().resolve()
    config = QwenConfig.from_json(model_dir / "config.json")
    dtype = DTYPES[args.dtype]
    device = torch.device("cuda")
    prefill_chunk_len = args.max_prefill_chunk_len or max(prompt_lens)
    max_batch_len = args.max_batch_len or prefill_chunk_len
    if max_batch_len < prefill_chunk_len:
        raise SystemExit("--max-batch-len must be >= --max-prefill-chunk-len")

    required_blocks = math.ceil(
        (max(prompt_lens) + args.max_new_len) / args.block_len
    )
    num_blocks = args.num_blocks or required_blocks + 8
    if num_blocks < required_blocks:
        raise SystemExit(
            f"--num-blocks must be at least {required_blocks} for this workload"
        )

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    seed_ids = tuple(
        tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            add_special_tokens=False,
        ).input_ids
    )
    if not seed_ids:
        raise RuntimeError("tokenizer produced an empty seed prompt")

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
    runner = QwenModelRunner(
        config,
        cache=cache,
        use_flash_attention=args.flash_attention,
        use_paged_decode_attention=args.paged_decode_attention,
        paged_decode_max_splits=args.paged_decode_max_splits,
    ).to(device=device, dtype=dtype).eval()
    runner.load_checkpoint(model_dir / "model.safetensors")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    scheduler = Scheduler(
        policy="fcfs",
        num_blocks=num_blocks,
        block_len=args.block_len,
        max_batch_len=max_batch_len,
        max_prefill_chunk_len=prefill_chunk_len,
    )
    executor = TorchExecutor(
        model_runner=runner,
        block_len=args.block_len,
        eos_token_id=config.eos_token_id if args.respect_eos else -1,
        device=device,
    )
    server = LLMServer(scheduler, executor)

    prefill_backend = "native-flash" if args.flash_attention else "eager"
    decode_backend = (
        "paged-split-kv"
        if args.paged_decode_attention
        else f"gather+{prefill_backend}"
    )
    backend = f"prefill={prefill_backend},decode={decode_backend}"
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}, cuda: {torch.version.cuda}, dtype: {dtype}")
    print(f"model: {model_dir}")
    print(f"backend: {backend}")
    print(
        f"prompt_lens={prompt_lens}, max_new_len={args.max_new_len}, "
        f"prefill_chunk_len={prefill_chunk_len}, num_blocks={num_blocks}, "
        f"warmup={args.warmup_repetitions}, repetitions={args.repetitions}"
    )
    print(f"model_startup_s: {load_seconds:.3f}")
    print(f"cuda_allocated_mib: {torch.cuda.memory_allocated() / 2**20:.1f}")

    rows: list[tuple[int, list[RequestMetrics]]] = []
    request_index = 0
    with torch.inference_mode():
        for prompt_len in prompt_lens:
            prompt_token_ids = build_prompt(seed_ids, prompt_len)
            for _ in range(args.warmup_repetitions):
                run_request(
                    server,
                    scheduler,
                    request_id=f"warmup-{request_index}",
                    prompt_token_ids=prompt_token_ids,
                    max_new_len=args.max_new_len,
                )
                request_index += 1

            measured = []
            for _ in range(args.repetitions):
                measured.append(
                    run_request(
                        server,
                        scheduler,
                        request_id=f"measured-{request_index}",
                        prompt_token_ids=prompt_token_ids,
                        max_new_len=args.max_new_len,
                    )
                )
                request_index += 1
            rows.append((prompt_len, measured))

    print()
    print(
        f"{'prompt':>8} {'output':>8} {'steps':>7} {'TTFT ms':>10} "
        f"{'TPOT ms':>10} {'decode tok/s':>13} {'E2E tok/s':>11} "
        f"{'prefill tok/s':>14} {'total ms':>10}"
    )
    for prompt_len, values in rows:
        print(
            f"{prompt_len:8d} "
            f"{int(median(values, 'generated_len')):8d} "
            f"{int(median(values, 'steps')):7d} "
            f"{median(values, 'ttft_ms'):10.3f} "
            f"{median(values, 'tpot_ms'):10.3f} "
            f"{median(values, 'decode_tokens_per_second'):13.2f} "
            f"{median(values, 'output_tokens_per_second'):11.2f} "
            f"{median(values, 'effective_prefill_tokens_per_second'):14.2f} "
            f"{median(values, 'total_ms'):10.3f}"
        )

    print()
    print("TTFT includes submit, scheduling, chunked Prefill, and first-token sampling.")
    print("TPOT/decode tok/s cover tokens after the first generated token.")
    print("Model loading and the per-shape warmup requests are excluded from timings.")
    if args.paged_decode_attention:
        print("Decode uses direct Paged Attention with adaptive Split-KV.")
    else:
        print("Decode still gathers paged KV before Attention.")


if __name__ == "__main__":
    main()
