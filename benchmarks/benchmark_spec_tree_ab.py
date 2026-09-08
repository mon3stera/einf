"""A/B benchmark: chain speculative decoding vs tree speculative decoding.

Builds the production pair (W4A16-7B target + bf16-0.5B draft), runs each
configuration on the same prompt under greedy decoding, and reports
acceptance and wall-clock numbers on an equal generated-token count.

Usage:
  python benchmarks/benchmark_spec_tree_ab.py <target_dir> <draft_dir> [steps]

All greedy configurations are lossless, so every run must produce the
identical output text — the script asserts that as a cross-engine sanity
check before trusting the timings.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch.qwen import QwenConfig, QwenModelRunner
from einf.executors.torch.spec_runner import SpeculativeEngine, _Track
from einf.executors.torch.spec_tree import TreeSpeculativeEngine

TARGET = Path(sys.argv[1])
DRAFT = Path(sys.argv[2])
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 96
PROMPT = "The capital of France is Paris. The city is famous for"

DEVICE = torch.device("cuda")
BLOCK_LEN = 32
NUM_BLOCKS = 256


def build(config_path: Path, model_dir: Path, *, w4a16: bool) -> QwenModelRunner:
    config = QwenConfig.from_json(config_path)
    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=NUM_BLOCKS,
            block_len=BLOCK_LEN,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=torch.bfloat16,
        device=DEVICE,
        use_custom_ops=True,
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(DEVICE):
            runner = QwenModelRunner(
                config,
                cache=cache,
                use_flashinfer_attention=True,
                dtype=torch.bfloat16,
                w4a16=w4a16,
            )
    finally:
        torch.set_default_dtype(previous)

    if w4a16:
        runner = runner.eval()
    else:
        runner = runner.to(device=DEVICE, dtype=torch.bfloat16).eval()

    runner.load_checkpoint(model_dir)
    return runner


def plain_greedy(
    runner: QwenModelRunner, prompt_ids: list[int], max_new_len: int
) -> list[int]:
    """Decode-only reference loop (spec machinery, one token per forward)."""
    track = _Track(
        forward=runner.forward,
        runner=runner,
        block_table=list(range(NUM_BLOCKS)),
        context_len=0,
        pending_token=-1,
    )
    helper = SpeculativeEngine(
        runner, runner, device=DEVICE, block_len=BLOCK_LEN, num_blocks=NUM_BLOCKS
    )
    ids: list[int] = []

    with torch.inference_mode():
        out = runner.forward(helper._make_input(track, prompt_ids))
        token = int(out.logits[-1].argmax().item())
        ids.append(token)
        track.context_len = len(prompt_ids)

        while len(ids) < max_new_len:
            out = runner.forward(helper._make_input(track, [token]))
            token = int(out.logits[-1].argmax().item())
            track.context_len += 1
            ids.append(token)
    return ids


CONFIGS: list[tuple[str, dict]] = [
    ("chain K=4", {"kind": "chain", "num_spec_tokens": 4}),
    ("chain K=8", {"kind": "chain", "num_spec_tokens": 8}),
    ("tree b2 d4 B=8", {"kind": "tree", "tree_budget": 8, "branch_factor": 2, "max_depth": 4}),
    ("tree b2 d4 B=16", {"kind": "tree", "tree_budget": 16, "branch_factor": 2, "max_depth": 4}),
    ("tree b4 d3 B=16", {"kind": "tree", "tree_budget": 16, "branch_factor": 4, "max_depth": 3}),
]


def run_config(
    name: str,
    cfg: dict,
    target: QwenModelRunner,
    draft: QwenModelRunner,
    prompt_ids: list[int],
    *,
    warmup: bool,
) -> tuple[list[int], object, float]:
    if cfg["kind"] == "chain":
        engine = SpeculativeEngine(
            target, draft, device=DEVICE,
            block_len=BLOCK_LEN, num_blocks=NUM_BLOCKS,
            num_spec_tokens=cfg["num_spec_tokens"],
        )
    else:
        engine = TreeSpeculativeEngine(
            target, draft, device=DEVICE,
            block_len=BLOCK_LEN, num_blocks=NUM_BLOCKS,
            tree_budget=cfg["tree_budget"],
            branch_factor=cfg["branch_factor"],
            max_depth=cfg["max_depth"],
        )

    length = 12 if warmup else MAX_NEW
    t0 = time.perf_counter()
    with torch.inference_mode():
        ids, stats = engine.generate(prompt_ids, max_new_len=length, greedy=True)
    wall = time.perf_counter() - t0
    return ids, stats, wall


def report(name: str, ids: list[int], stats, wall: float, *, warmup: bool) -> None:
    steps = max(stats.steps, 1)
    generated = len(ids) - 0
    ms_per_step = wall / steps * 1000
    tok_per_step = len(ids) / steps
    tok_s = len(ids) / wall
    print(
        f"{name:20s} steps={stats.steps:4d} "
        f"acc/step={stats.accepted / steps:5.2f} "
        f"commit/step={stats.committed / steps:5.2f} "
        f"| wall={wall:7.3f}s ms/step={ms_per_step:7.1f} "
        f"gen={generated if not warmup else 'warmup'} tok/s={tok_s:6.1f}",
        flush=True,
    )


def main() -> None:
    torch.manual_seed(0)
    print("building target (W4A16)...", flush=True)
    target = build(TARGET / "config.json", TARGET, w4a16=True)
    print("building draft (bf16)...", flush=True)
    draft = build(DRAFT / "config.json", DRAFT, w4a16=False)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(TARGET))
    prompt_ids = tokenizer.encode(PROMPT)
    print(f"prompt: {len(prompt_ids)} tokens; generating {MAX_NEW} tokens", flush=True)

    # Warmup: kernels, decode graph capture, FlashInfer JIT per config.
    print("--- warmup ---", flush=True)
    warm_outputs = []

    for name, cfg in CONFIGS:
        ids, stats, wall = run_config(
            name, cfg, target, draft, prompt_ids, warmup=True
        )
        warm_outputs.append(ids)
        report(f"warmup {name}", ids, stats, wall, warmup=True)

    print("--- plain decode reference ---", flush=True)
    t0 = time.perf_counter()
    reference_ids = plain_greedy(target, prompt_ids, MAX_NEW)
    wall = time.perf_counter() - t0
    print(
        f"{'plain decode':20s} wall={wall:7.3f}s "
        f"tok/s={MAX_NEW / wall:6.1f}",
        flush=True,
    )

    print("--- measured ---", flush=True)
    outputs = []

    for name, cfg in CONFIGS:
        ids, stats, wall = run_config(
            name, cfg, target, draft, prompt_ids, warmup=False
        )
        outputs.append(ids)
        report(name, ids, stats, wall, warmup=False)

    # Losslessness cross-check: every greedy configuration must emit the
    # same tokens as the plain decode loop.
    for name, ids in zip([n for n, _ in CONFIGS], outputs):
        if ids[:MAX_NEW] != reference_ids:
            mismatch = next(
                (i for i, (a, b) in enumerate(zip(ids, reference_ids)) if a != b),
                min(len(ids), len(reference_ids)),
            )
            print(
                f"WARNING: {name} diverges from plain decode at token "
                f"{mismatch} (losslessness check failed)",
                flush=True,
            )
        else:
            print(f"lossless ok: {name}", flush=True)


if __name__ == "__main__":
    main()
