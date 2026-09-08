"""Bisect the 7B tree-engine divergence: toggle W4A16 / custom ops / shape.

Usage:
  python benchmarks/repro_tree_divergence_7b.py <target> <draft> [--bf16] [--no-custom-ops] [budget branch depth]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, "tests")

from test_speculative_tree_gpu import _top2_gap  # noqa: E402

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage  # noqa: E402
from einf.executors.torch.qwen import QwenConfig, QwenModelRunner  # noqa: E402
from einf.executors.torch.spec_runner import SpeculativeEngine, _Track  # noqa: E402
from einf.executors.torch.spec_tree import TreeSpeculativeEngine  # noqa: E402

DEVICE = torch.device("cuda")
BLOCK_LEN = 32
NUM_BLOCKS = 256


def build(config_path: Path, model_dir: Path, *, w4a16: bool, custom_ops: bool) -> QwenModelRunner:
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
        use_custom_ops=custom_ops,
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


def plain_greedy(runner: QwenModelRunner, prompt_ids: list[int], max_new_len: int):
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
    gaps: list[float] = []

    with torch.inference_mode():
        out = runner.forward(helper._make_input(track, prompt_ids))
        token = int(out.logits[-1].argmax().item())
        ids.append(token)
        gaps.append(_top2_gap(out.logits[-1]))
        track.context_len = len(prompt_ids)

        while len(ids) < max_new_len:
            out = runner.forward(helper._make_input(track, [token]))
            token = int(out.logits[-1].argmax().item())
            gaps.append(_top2_gap(out.logits[-1]))
            track.context_len += 1
            ids.append(token)
    return ids, gaps


def main() -> None:
    args = [a for a in sys.argv[1:]]
    w4a16 = "--bf16" not in args
    custom_ops = "--no-custom-ops" not in args
    args = [a for a in args if not a.startswith("--")]
    target_dir, draft_dir = Path(args[0]), Path(args[1])
    budget = int(args[2]) if len(args) > 2 else 16
    branch = int(args[3]) if len(args) > 3 else 2
    depth = int(args[4]) if len(args) > 4 else 4
    max_new = int(args[5]) if len(args) > 5 else 96

    print(
        f"config: w4a16={w4a16} custom_ops={custom_ops} "
        f"tree=({budget},{branch},{depth}) max_new={max_new}",
        flush=True,
    )
    target = build(target_dir / "config.json", target_dir, w4a16=w4a16, custom_ops=custom_ops)
    draft = build(draft_dir / "config.json", draft_dir, w4a16=False, custom_ops=custom_ops)
    # independent reference runner: validates the engine's output as a valid
    # greedy trajectory, immune to tie-flip trajectory cascades
    reference = build(target_dir / "config.json", target_dir, w4a16=w4a16, custom_ops=custom_ops)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(target_dir))
    prompt_ids = tokenizer.encode(
        "The capital of France is Paris. The city is famous for"
    )

    engine = TreeSpeculativeEngine(
        target,
        draft,
        device=DEVICE,
        block_len=BLOCK_LEN,
        num_blocks=NUM_BLOCKS,
        tree_budget=budget,
        branch_factor=branch,
        max_depth=depth,
    )

    # reference decode state: validate each committed token as the argmax
    # (within tie tolerance) of the reference at the evolving context
    ref_track = _Track(
        forward=reference.forward,
        runner=reference,
        block_table=list(range(NUM_BLOCKS)),
        context_len=0,
        pending_token=-1,
    )
    ref_helper = SpeculativeEngine(
        reference, reference, device=DEVICE,
        block_len=BLOCK_LEN, num_blocks=NUM_BLOCKS,
    )
    TOL = 1e-2
    invalid: list[tuple[int, int, float]] = []
    generated: list[int] = []
    ref_state = {"logits": None}

    def validate(token: int, output_index: int) -> None:
        logits = ref_state["logits"]
        top = float(logits.max())

        if float(logits[token]) < top - TOL:
            invalid.append(
                (output_index, engine._stats.steps, top - float(logits[token]))
            )

        out = reference.forward(ref_helper._make_input(ref_track, [token]))
        ref_track.context_len += 1
        ref_state["logits"] = out.logits[-1]

    with torch.inference_mode():
        out = reference.forward(ref_helper._make_input(ref_track, prompt_ids))
        ref_track.context_len = len(prompt_ids)
        ref_state["logits"] = out.logits[-1]

        while len(generated) < max_new:
            print(
                f"step {engine._stats.steps}: pending(draft={engine._draft.pending_token}, "
                f"target={engine._target.pending_token}) "
                f"ctx(draft={engine._draft.context_len}, target={engine._target.context_len})",
                flush=True,
            )
            committed = engine.step(greedy=True)
            print(f"  committed {committed}", flush=True)

            for token in committed:
                validate(token, len(generated))
                generated.append(token)

    if not invalid:
        print(
            f"trajectory valid (steps={engine._stats.steps} "
            f"acc/step={engine._stats.accepted / max(engine._stats.steps, 1):.2f})",
            flush=True,
        )
        return

    for output_index, step, deficit in invalid[:8]:
        print(
            f"INVALID token at output {output_index} (step {step}): "
            f"logit deficit {deficit:.3e}",
            flush=True,
        )
    print(f"invalid tokens: {len(invalid)}", flush=True)


if __name__ == "__main__":
    main()
