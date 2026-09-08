"""Reproduce the tree-engine well-separated divergence on the tiny model.

Runs the tree engine against a plain greedy decode loop across shapes and
reports the reference top-2 gap at each first divergence, so a real
losslessness break can be separated from near-tie numerics flips.
"""

from __future__ import annotations

import sys

import torch

sys.path.insert(0, "tests")

from test_speculative_tree_gpu import (  # noqa: E402
    MAX_NEW,
    NUM_BLOCKS,
    BLOCK_LEN,
    _plain_greedy,
    _tiny_runner,
)

from einf.executors.torch.spec_tree import TreeSpeculativeEngine  # noqa: E402

SHAPES = [
    ("b2 d3 B=8", 8, 2, 3),
    ("b2 d4 B=16", 16, 2, 4),
    ("b4 d3 B=16", 16, 4, 3),
    ("b4 d4 B=32", 32, 4, 4),
]


def main() -> None:
    device = torch.device("cuda")
    target = _tiny_runner(device, seed=7)
    draft = _tiny_runner(device, seed=11)
    plain_ids, plain_gaps = _plain_greedy(target, device, [11, 5, 23])

    for name, budget, branch, depth in SHAPES:
        engine = TreeSpeculativeEngine(
            target,
            draft,
            device=device,
            block_len=BLOCK_LEN,
            num_blocks=NUM_BLOCKS,
            tree_budget=budget,
            branch_factor=branch,
            max_depth=depth,
        )
        with torch.inference_mode():
            ids, stats = engine.generate(
                [11, 5, 23], max_new_len=MAX_NEW, greedy=True
            )
        diverged = [
            i for i, (a, b) in enumerate(zip(ids, plain_ids)) if a != b
        ]

        if not diverged:
            print(f"{name:14s} lossless ok (steps={stats.steps})", flush=True)
            continue

        first = diverged[0]
        gap = plain_gaps[first] if first < len(plain_gaps) else -1.0
        verdict = "near-tie" if gap < 1e-2 else "WELL-SEPARATED"
        print(
            f"{name:14s} diverged at {diverged} first-gap={gap:.3e} "
            f"({verdict}) steps={stats.steps} "
            f"acc/step={stats.accepted / max(stats.steps, 1):.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
