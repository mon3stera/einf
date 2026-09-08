"""Per-phase time decomposition for the speculative decoding step.

Owns the outer loop (no production edits) and wraps engine entry points
with phase timers. Tree phases:

    expand_q1     _draft_forward_q1       draft q=1 forward (graph-eligible)
    expand_level  level draft forwards    batched per-depth draft forwards
    expand_input  tree input builds       level mask pack + H2D inside expand
    verify_input  verify input build      tree mask pack + H2D for the verify
    verify_fwd    target.forward          masked [pending + T] verify forward
    walk          find_greedy_path        CPU walk over verify logits
    gather        gather_commit_kv x2     per-layer KV slot copies
    step_total    full step               residual = bookkeeping + Python

Chain phases: chain_draft (K+1 draft decode loop), verify_fwd, walk, step_total.

Modes
    wall  no added synchronisation; phase times are CPU-side occupancy and
          device cost surfaces in whichever phase blocks (normally verify_fwd).
    sync  torch.cuda.synchronize() after each phase; totals are inflated but
          phase costs are device-attributed. Use for optimization targets.

--profiler adds a torch.profiler kernel table over a few steps.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "benchmarks")

from repro_tree_divergence_7b import BLOCK_LEN, NUM_BLOCKS, build  # noqa: E402

import einf.executors.torch.spec_tree as spec_tree_mod  # noqa: E402
from einf.executors.torch.spec_runner import SpeculativeEngine  # noqa: E402
from einf.executors.torch.spec_tree import TreeSpeculativeEngine  # noqa: E402

DEVICE = torch.device("cuda")


def instrument(engine, state: dict) -> None:
    """Wrap engine entry points with phase timers feeding ``state['ms']``."""

    def timed(name: str, fn, sync: bool):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            try:
                return fn(*args, **kwargs)
            finally:
                if sync:
                    torch.cuda.synchronize()

                state["ms"][name] = (
                    state["ms"].get(name, 0.0) + (time.perf_counter() - start) * 1e3
                )

        return wrapper

    sync = state["sync"]
    draft_track = engine._draft
    target_track = engine._target

    if state["engine"] == "tree":
        orig_expand = engine._expand_draft

        def expand_wrapper(*args, **kwargs):
            state["in_expand"] = True

            try:
                return orig_expand(*args, **kwargs)
            finally:
                state["in_expand"] = False

        engine._expand_draft = expand_wrapper

        orig_q1 = engine._draft_forward_q1

        def q1_wrapper(*args, **kwargs):
            state["in_q1"] = True

            try:
                return orig_q1(*args, **kwargs)
            finally:
                state["in_q1"] = False

        engine._draft_forward_q1 = timed("expand_q1", q1_wrapper, sync)

        draft_track.forward = timed("draft_fwd", draft_track.forward, sync)
        engine._make_tree_input = timed("tree_input", engine._make_tree_input, sync)

        def draft_router(*args, **kwargs):
            if state.get("in_q1"):
                return draft_track.forward(*args, **kwargs)

            if state.get("in_expand"):
                return timed("expand_level", draft_track.forward, sync)(
                    *args, **kwargs
                )

            return draft_track.forward(*args, **kwargs)

        # replace the raw forward with the routing wrapper
        draft_track.forward = draft_router
        engine._draft.forward = draft_router
    else:
        engine._draft_decode_forward = timed("chain_draft", engine._draft_decode_forward, sync)

    target_track.forward = timed("verify_fwd", target_track.forward, sync)

    spec_tree_mod.gather_commit_kv = timed("gather", spec_tree_mod.gather_commit_kv, sync)
    spec_tree_mod.find_greedy_path = timed("walk", spec_tree_mod.find_greedy_path, sync)

    orig_step = engine.step

    def stepped(*args, **kwargs):
        start = time.perf_counter()

        try:
            return orig_step(*args, **kwargs)
        finally:
            if sync:
                torch.cuda.synchronize()

            state["ms"]["step_total"] = (
                state["ms"].get("step_total", 0.0) + (time.perf_counter() - start) * 1e3
            )

    engine.step = stepped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("draft_dir", type=Path)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--branch", type=int, default=2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--engine", choices=["tree", "chain"], default="tree")
    parser.add_argument("--mode", choices=["wall", "sync"], default="sync")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profiler", action="store_true")
    args = parser.parse_args()

    target = build(args.target_dir / "config.json", args.target_dir, w4a16=True, custom_ops=True)
    draft = build(args.draft_dir / "config.json", args.draft_dir, w4a16=False, custom_ops=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.target_dir))
    prompt_ids = tokenizer.encode(
        "The capital of France is Paris. The city is famous for"
    )

    if args.engine == "tree":
        engine = TreeSpeculativeEngine(
            target,
            draft,
            device=DEVICE,
            block_len=BLOCK_LEN,
            num_blocks=NUM_BLOCKS,
            tree_budget=args.budget,
            branch_factor=args.branch,
            max_depth=args.depth,
        )
    else:
        engine = SpeculativeEngine(
            target,
            draft,
            device=DEVICE,
            block_len=BLOCK_LEN,
            num_blocks=NUM_BLOCKS,
            num_spec_tokens=args.budget,
        )

    state = {
        "engine": args.engine,
        "sync": args.mode == "sync",
        "ms": {},
        "per_step": {},
    }

    if args.mode == "sync":
        torch.cuda.synchronize()

    with torch.inference_mode():
        engine.prefill(prompt_ids, greedy=True)
        instrument(engine, state)

        started = time.perf_counter()

        for _ in range(args.max_new):
            snapshot = dict(state["ms"])
            engine.step(greedy=True)
            torch.cuda.synchronize()
            now = state["ms"]

            for name, total in now.items():
                state["per_step"].setdefault(name, []).append(
                    total - snapshot.get(name, 0.0)
                )

        wall = time.perf_counter() - started

    generated = engine._stats.committed
    print(
        f"{args.engine} engine ({args.mode}): committed={generated} "
        f"steps={engine._stats.steps} "
        f"acc/step={engine._stats.accepted / max(engine._stats.steps, 1):.2f} "
        f"wall={wall:.2f}s tok/s={generated / wall:.1f}"
    )

    if args.profiler:
        from torch.profiler import ProfilerActivity, profile

        with torch.inference_mode():
            engine.prefill(prompt_ids, greedy=True)

            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                for _ in range(4):
                    engine.step(greedy=True)

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
        return

    print(f"\n=== phase ms/step ({args.mode}, first {args.warmup} steps discarded) ===")
    rows = []

    for name, samples in state["per_step"].items():
        per = samples[args.warmup:]

        if not per:
            continue

        p50 = statistics.median(per)
        p95 = sorted(per)[max(int(0.95 * len(per)) - 1, 0)]
        rows.append((name, statistics.fmean(per), p50, p95, len(per)))

    order = [
        "step_total",
        "expand_q1",
        "expand_level",
        "tree_input",
        "verify_fwd",
        "chain_draft",
        "walk",
        "gather",
    ]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)

    for name, mean, p50, p95, n in rows:
        print(f"  {name:12s} mean {mean:7.3f}  p50 {p50:7.3f}  p95 {p95:7.3f}  n={n}")

    total = next((r[1] for r in rows if r[0] == "step_total"), 0.0)
    covered = sum(r[1] for r in rows if r[0] != "step_total")

    print(f"  {'residual':12s} mean {total - covered:7.3f}  (bookkeeping/python)")


if __name__ == "__main__":
    main()
