"""Group vLLM worker CUDA kernels for decode.

vLLM V1 runs the model in a child process, so the parent torch.profiler is
empty. This uses the engine's TorchProfilerWrapper and dumps
``profiler_out_0.txt``. Prefill iterations are skipped with delay_iterations.
"""

from __future__ import annotations

import argparse
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


KERNEL_GROUPS = (
    ("gemm", ("cutlass", "wmma", "gemm", "gemv", "scaled_mm", "nvjet", "bmm")),
    (
        "attention",
        (
            "flash",
            "fmha",
            "batchprefill",
            "batchdecode",
            "splitkreduce",
            "persistentvariable",
            "paged",
        ),
    ),
    ("rmsnorm", ("rmsnorm", "fusedaddrms", "rms_norm")),
    ("rope", ("rotary", "rope")),
    ("kv_cache", ("reshape_and_cache", "append_paged", "write_slots", "cache_kernel")),
    ("activation", ("act_and_mul", "silu", "gelu")),
    ("elementwise", ("elementwise",)),
    ("sampler", ("softmax", "argmax", "multinomial", "top_k", "top_p")),
    ("memcpy", ("memcpy", "memset")),
)

ROW = re.compile(
    r"^(?P<name>.+?)\s+(?P<count>\d+)\.\d+\s+.*?(?P<cuda>[0-9.]+)(?P<unit>us|ms|s)\s*$"
)


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    for group, needles in KERNEL_GROUPS:
        if any(needle in lowered for needle in needles):
            return group
    return "other"


def parse_profiler_table(path: Path) -> list[tuple[str, int, float]]:
    """Parse key_averages().table(sort_by=self_cuda_time_total) dump."""
    rows: list[tuple[str, int, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        if "void " not in line and "Memcpy" not in line and "Memset" not in line:
            continue
        parts = line.rsplit()
        if len(parts) < 4:
            continue
        # table columns vary; CUDA time is typically the last numeric+unit field
        # Format is messy; fall back to scanning for a trailing duration.
        match = re.search(r"(\d+\.\d+)\s*(us|ms|s)\s*$", line)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2)
        ms = value / 1000.0 if unit == "us" else value * 1000.0 if unit == "s" else value
        count_match = re.search(r"\s(\d+)\s", line)
        count = int(count_match.group(1)) if count_match else 0
        name = line.strip()
        if "void " in name:
            name = name[name.index("void ") :]
        elif "Memcpy" in name:
            name = name[name.index("Memcpy") :]
        elif "Memset" in name:
            name = name[name.index("Memset") :]
        name = re.split(r"\s{2,}", name)[0]
        rows.append((name, count, ms))
    return rows


def print_grouped(title: str, rows: list[tuple[str, int, float]], steps: int) -> None:
    grouped: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for name, count, ms in rows:
        grouped[classify_kernel(name)][0] += count
        grouped[classify_kernel(name)][1] += ms
    total = sum(ms for _, ms in grouped.values())
    print(f"\n=== {title} ===")
    print(f"{'name':<28}{'count/step':>12}{'ms/step':>10}{'share':>8}")
    for name, (count, ms) in sorted(
        grouped.items(), key=lambda item: item[1][1], reverse=True
    ):
        share = 100.0 * ms / total if total else 0.0
        print(f"{name:<28}{count / steps:>12.1f}{ms / steps:>10.3f}{share:>7.1f}%")
    print(f"{'TOTAL':<28}{'':>12}{total / steps:>10.3f}{100.0:>7.1f}%")
    print("\n=== top kernels ===")
    print(f"{'kernel':<56}{'count/step':>12}{'ms/step':>10}")
    for name, count, ms in sorted(rows, key=lambda item: item[2], reverse=True)[:18]:
        print(f"{name[:55]:<56}{count / steps:>12.1f}{ms / steps:>10.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/media/zzx/新加卷2/models/Qwen2.5-0.5B"),
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--prefill-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("/tmp/vllm-forward-profile"),
    )
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    args = parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    prompts = [
        TokensPrompt(
            prompt_token_ids=[rng.randrange(3, 32000) for _ in range(args.prompt_len)]
        )
        for _ in range(args.concurrency)
    ]

    llm = LLM(
        model=str(args.model_dir),
        tokenizer=str(args.model_dir),
        dtype="bfloat16",
        max_model_len=32768,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.8,
        enforce_eager=False,
        enable_prefix_caching=False,
        disable_log_stats=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(args.trace_dir),
            "torch_profiler_with_stack": False,
            "torch_profiler_use_gzip": False,
            "delay_iterations": args.prefill_iters,
            "max_iterations": args.decode_tokens,
            "ignore_frontend": True,
        },
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        ignore_eos=True,
        seed=args.seed,
    )
    print("warmup generate")
    llm.generate(prompts, params, use_tqdm=False)

    print(
        f"profile decode: skip {args.prefill_iters} prefill iters, "
        f"capture {args.decode_tokens} decode iters"
    )
    llm.start_profile("vllm_decode")
    llm.generate(prompts, params, use_tqdm=False)
    llm.stop_profile()

    dump = args.trace_dir / "profiler_out_0.txt"
    if not dump.is_file():
        dumps = list(args.trace_dir.glob("profiler_out_*.txt"))
        if not dumps:
            raise SystemExit(f"no profiler_out_*.txt under {args.trace_dir}")
        dump = dumps[0]
    print(f"parsed {dump}")
    rows = parse_profiler_table(dump)
    if not rows:
        print(dump.read_text()[:4000])
        raise SystemExit("failed to parse profiler table")
    print_grouped(
        f"vLLM decode kernels / {args.decode_tokens} steps",
        rows,
        args.decode_tokens,
    )


if __name__ == "__main__":
    main()
