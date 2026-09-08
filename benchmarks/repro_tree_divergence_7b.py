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


def diagnose_cache(
    snapshot: dict, engine_cache, ref_cache
) -> None:
    """Compare the two caches slot-by-slot over the committed region:
    both runners hold the same token ids at the same logical slots with the
    same rope positions, so K/V should agree to bf16 numerics."""
    spec = snapshot["spec"]
    live_now = spec.total_kv if snapshot.get("spec") is not None else 0
    # region committed BEFORE the failing step's verify wrote its scratch:
    # everything except this step's scratch [live, live+1+T)
    live = snapshot["live"]
    span = live
    for name, cache in (("engine", engine_cache), ("reference", ref_cache)):
        print(
            f"{name} cache K shape {tuple(cache.K.shape)} dtype {cache.K.dtype}",
            flush=True,
        )

    ek = engine_cache.K[0].view(-1, *engine_cache.K.shape[2:])
    rk = ref_cache.K[0].view(-1, *ref_cache.K.shape[2:])
    ev = engine_cache.V[0].view(-1, *engine_cache.V.shape[2:])
    rv = ref_cache.V[0].view(-1, *ref_cache.V.shape[2:])
    diff_k = (ek[:span].float() - rk[:span].float()).abs().amax(dim=-1)
    diff_v = (ev[:span].float() - rv[:span].float()).abs().amax(dim=-1)
    bad_k = (diff_k > 1e-2).nonzero().flatten().tolist()
    bad_v = (diff_v > 1e-2).nonzero().flatten().tolist()
    print(
        f"committed region [0, {span}): K bad slots {len(bad_k)} {bad_k[:12]} "
        f"max diff {float(diff_k.max()):.4f}; V bad slots {len(bad_v)} "
        f"{bad_v[:12]} max diff {float(diff_v.max()):.4f}",
        flush=True,
    )


def diagnose_row(
    snapshot: dict,
    reference: QwenModelRunner,
    ref_helper: SpeculativeEngine,
    ref_track: _Track,
    prompt_ids: list[int],
    generated: list[int],
) -> None:
    """Recompute every verify row of the failing step with an independent
    runner (causal forward over exactly the row's visible token set, RoPE
    by depth) and compare argmaxes against the engine's verify logits."""
    from einf.executors.torch.input import ModelInput

    spec = snapshot["spec"]
    logits = snapshot["logits"]
    live = snapshot["live"]
    pending = snapshot["pending"]
    history = prompt_ids + generated[: live - len(prompt_ids)]
    t = len(spec.tokens)
    mismatches = 0

    for r in range(spec.total_q):
        if r == 0:
            visible_nodes: list[int] = []
            row_pending = True
        else:
            j = r - 1
            word = spec.node_masks[j]
            visible_nodes = [k for k in range(t) if (word >> (k + 1)) & 1]
            row_pending = bool(word & 1)

        tokens = list(history)
        positions = list(range(live))

        if row_pending:
            tokens.append(pending)
            positions.append(live)

        for k in visible_nodes:
            tokens.append(spec.tokens[k])
            positions.append(spec.rope_pos(k))

        tokens.append(spec.tokens[r - 1] if r > 0 else pending)
        positions.append(spec.rope_pos(r - 1) if r > 0 else live)

        base = 2000
        model_input = ModelInput(
            input_token_ids=torch.tensor(tokens, dtype=torch.long, device=DEVICE),
            position=torch.tensor(positions, dtype=torch.long, device=DEVICE),
            slot_mapping=torch.tensor(
                [base + i for i in range(len(tokens))], dtype=torch.long, device=DEVICE
            ),
            query_start_loc=torch.tensor([0, len(tokens)], dtype=torch.long, device=DEVICE),
            block_tables=torch.arange(NUM_BLOCKS, dtype=torch.long, device=DEVICE).unsqueeze(0),
            context_lens=torch.tensor([len(tokens)], dtype=torch.long, device=DEVICE),
            query_start_loc_host=(0, len(tokens)),
            context_lens_host=(len(tokens),),
        )
        out = reference.forward(model_input)
        ref_argmax = int(out.logits[-1].argmax().item())
        engine_argmax = int(logits[r].argmax().item())
        top_ref = torch.topk(out.logits[-1].float(), 2)
        ok = engine_argmax == ref_argmax
        deficit = float(top_ref.values[0] - out.logits[-1].float()[engine_argmax])
        mark = "ok" if ok else f"MISMATCH deficit={deficit:.3f}"
        if not ok:
            mismatches += 1
        print(
            f"row {r:2d} ({'pending' if r == 0 else f'node {r - 1}'}): "
            f"visible={len(tokens)} engine_argmax={engine_argmax} "
            f"ref_argmax={ref_argmax} {mark}",
            flush=True,
        )

    print(f"rows mismatched: {mismatches}/{spec.total_q}", flush=True)


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
    # per-step snapshot for row-level diagnosis of the first invalid token
    snapshot: dict = {}

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

        first = engine.prefill(prompt_ids, greedy=True)
        validate(first, len(generated))
        generated.append(first)

        while len(generated) < max_new and not invalid:
            snapshot["pending"] = engine._target.pending_token
            snapshot["live"] = engine._target.context_len
            committed = engine.step(greedy=True)
            snapshot.update(spec=engine._last_tree[0], logits=engine._last_tree[2])

            for token in committed:
                validate(token, len(generated))
                generated.append(token)

    if invalid:
        diagnose_row(snapshot, reference, ref_helper, ref_track, prompt_ids, generated)
        diagnose_cache(snapshot, engine._target.runner.cache, reference.cache)

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
