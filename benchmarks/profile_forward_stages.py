"""Split decode forward into stages (eager record_function) and graph kernels."""

from __future__ import annotations

import argparse
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch import QwenConfig, QwenModelRunner, TorchExecutor
from einf.request import RequestSpec, RequestState
from einf.scheduler import Scheduler, WorkType


DEFAULT_MODEL_DIR = Path(
    os.environ.get("EINF_MODEL_DIR", "~/models/Qwen2.5-0.5B")
).expanduser()

KERNEL_GROUPS = (
    ("gemm", ("cutlass", "wmma", "gemm")),
    ("attention", ("batchprefill", "batchdecode", "splitkreduce", "persistentvariable")),
    ("rmsnorm", ("rmsnorm", "fusedaddrms")),
    ("rope", ("rotary", "rope")),
    ("write_slots", ("write_slots",)),
    ("activation", ("act_and_mul", "silu")),
    ("elementwise", ("elementwise",)),
    ("memcpy", ("memcpy", "memset")),
)

STAGE_KEYS = (
    "fwd.embed",
    "layer.input_norm",
    "attn.qkv",
    "attn.rope",
    "attn.write",
    "attn.flashinfer",
    "attn.o_proj",
    "layer.post_attn_norm",
    "layer.mlp",
    "fwd.final_norm",
    "fwd.lm_head",
)


def device_time_us(event) -> float:
    for attribute in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(event, attribute, 0.0) or 0.0
        if value:
            return float(value)
    return 0.0


def cpu_time_us(event) -> float:
    for attribute in ("self_cpu_time_total", "self_cpu_time"):
        value = getattr(event, attribute, 0.0) or 0.0
        if value:
            return float(value)
    return 0.0


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    for group, needles in KERNEL_GROUPS:
        if any(needle in lowered for needle in needles):
            return group
    return "other"


def print_table(title: str, rows: list[tuple[str, int, float]], steps: int) -> None:
    total = sum(ms for _, _, ms in rows)
    print(f"\n=== {title} ===")
    print(f"{'name':<28}{'count/step':>12}{'ms/step':>10}{'share':>8}")
    for name, count, ms in rows:
        share = 100.0 * ms / total if total else 0.0
        print(f"{name:<28}{count / steps:>12.1f}{ms / steps:>10.3f}{share:>7.1f}%")
    print(f"{'TOTAL':<28}{'':>12}{total / steps:>10.3f}{100.0:>7.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    config = QwenConfig.from_json(model_dir / "config.json")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    blocks_per_request = math.ceil((args.prompt_len + 64) / args.block_len)
    num_blocks = args.concurrency * blocks_per_request + args.concurrency
    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=num_blocks + 1,
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
            use_flashinfer_attention=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    runner.load_checkpoint(model_dir / "model.safetensors")
    scheduler = Scheduler(
        policy="fcfs",
        num_blocks=num_blocks,
        block_len=args.block_len,
        max_batch_len=512,
        max_prefill_chunk_len=128,
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
        scheduler.submit(
            RequestSpec(
                request_id=request_id,
                prompt_token_ids=tuple(
                    rng.randrange(3, config.vocab_size) for _ in range(args.prompt_len)
                ),
                max_new_len=64,
            )
        )
        active.add(request_id)

    def step() -> object:
        batch = scheduler.schedule()
        result = executor.execute(batch)
        scheduler.apply_result(result)
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
        return batch

    for _ in range(args.concurrency):
        submit()
    decode_batch = None
    with torch.inference_mode():
        for _ in range(200):
            batch = step()
            prefill = sum(
                scheduled.work_type is WorkType.PREFILL for scheduled in batch.requests
            )
            if prefill == 0 and len(batch.requests) == args.concurrency:
                decode_batch = batch
                break
        torch.cuda.synchronize()
        if decode_batch is None:
            raise SystemExit("never saw a decode-only batch")

        graph = runner.decode_graph
        packed = graph.pack_plan(decode_batch)
        if packed is None:
            raise SystemExit("pack_plan rejected decode batch")
        bucket, batch_size = packed
        graph._ensure_captured(bucket)
        torch.cuda.synchronize()

        print(f"device: {torch.cuda.get_device_name()}")
        print(f"decode batch={batch_size}, bucket={bucket}, context~{args.prompt_len}")
        print(f"repeats={args.repeats}")

        def replay() -> None:
            graph._replay_bucket(bucket, batch_size)

        def eager() -> None:
            runner.flashinfer.plan_decode_bucket(bucket, graph._inputs[bucket])
            runner.forward_compute(graph._inputs[bucket])

        for _ in range(5):
            replay()
            eager()
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as graph_prof:
            for _ in range(args.repeats):
                replay()
            torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as eager_prof:
            for _ in range(args.repeats):
                eager()
            torch.cuda.synchronize()

    kernel_ms: dict[str, float] = defaultdict(float)
    kernel_count: dict[str, int] = defaultdict(int)
    raw_kernels: list[tuple[str, int, float]] = []
    for row in graph_prof.key_averages():
        name = row.key
        if not (
            name.startswith(("void ", "Memcpy", "Memset"))
            or str(getattr(row, "device_type", "")).upper().endswith("CUDA")
        ):
            continue
        ms = device_time_us(row) / 1000.0
        if ms <= 0:
            continue
        kernel_ms[classify_kernel(name)] += ms
        kernel_count[classify_kernel(name)] += row.count
        raw_kernels.append((name, row.count, ms))

    print_table(
        "CUDA graph kernels grouped",
        sorted(
            ((name, kernel_count[name], ms) for name, ms in kernel_ms.items()),
            key=lambda item: item[2],
            reverse=True,
        ),
        args.repeats,
    )
    print("\n=== CUDA graph top kernels ===")
    print(f"{'kernel':<56}{'count/step':>12}{'ms/step':>10}")
    for name, count, ms in sorted(raw_kernels, key=lambda item: item[2], reverse=True)[:15]:
        print(f"{name[:55]:<56}{count / args.repeats:>12.1f}{ms / args.repeats:>10.3f}")

    stage_rows = []
    seen = set()
    for row in eager_prof.key_averages():
        if row.key not in STAGE_KEYS:
            continue
        seen.add(row.key)
        stage_rows.append((row.key, row.count, device_time_us(row) / 1000.0))
    stage_rows.sort(key=lambda item: STAGE_KEYS.index(item[0]))
    print_table(
        "eager forward record_function (device time, includes launch gaps)",
        stage_rows,
        args.repeats,
    )
    missing = [key for key in STAGE_KEYS if key not in seen]
    if missing:
        print("missing stages:", ", ".join(missing))


if __name__ == "__main__":
    main()
