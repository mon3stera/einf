"""Decompose the draft-loop host time empirically.

Wraps each stage of the engine's draft decode forward with perf_counter
timers (make_input / fill / plan / graph replay / clone) and reports the
per-call average over N spec steps. No torch.profiler — pure wall clock,
so the numbers are directly comparable to the A/B benchmarks.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_spec_step import K, DRAFT, TARGET, build  # noqa: E402

from einf.executors.torch.spec_runner import SpeculativeEngine  # noqa: E402

STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(0)
    target = build(TARGET / "config.json", TARGET, w4a16=True)
    draft = build(DRAFT / "config.json", DRAFT, w4a16=False)
    engine = SpeculativeEngine(
        target, draft, device=device, block_len=32, num_blocks=256, num_spec_tokens=K
    )

    t: dict[str, float] = defaultdict(float)
    c: dict[str, int] = defaultdict(int)

    def timed(name: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            t[name] += time.perf_counter() - t0
            c[name] += 1
            return out

        return wrapper

    engine._make_input = timed("make_input", engine._make_input)

    # FlashInferPagedAttention instances are frozen — patch at class level.
    from einf.executors.torch.decode_graph import DecodeCudaGraph

    DecodeCudaGraph._fill = timed("fill", DecodeCudaGraph._fill)
    DecodeCudaGraph._replay_bucket = timed("replay", DecodeCudaGraph._replay_bucket)

    orig_try = DecodeCudaGraph.try_replay

    def try_wrapped(self, model_input):
        out = orig_try(self, model_input)
        c["try_replay_hits" if out is not None else "try_replay_miss"] += 1
        return out

    DecodeCudaGraph.try_replay = try_wrapped
    print("draft.decode_graph:", draft.decode_graph is not None, flush=True)
    from einf.executors.torch.flashinfer_attn import FlashInferPagedAttention

    FlashInferPagedAttention.plan_decode_bucket = timed(
        "plan", FlashInferPagedAttention.plan_decode_bucket
    )

    orig_clone_zone = engine._draft_decode_forward

    def draft_wrapped(track, token):
        t0 = time.perf_counter()
        out = orig_clone_zone(track, token)
        t["draft_total"] += time.perf_counter() - t0
        c["draft_total"] += 1
        return out

    engine._draft_decode_forward = draft_wrapped

    orig_verify = engine._target.forward

    def verify_wrapped(mi):
        t0 = time.perf_counter()
        out = orig_verify(mi)
        t["verify"] += time.perf_counter() - t0
        c["verify"] += 1
        return out

    engine._target.forward = verify_wrapped

    try:
        from transformers import AutoTokenizer

        prompt_ids = AutoTokenizer.from_pretrained(str(TARGET))(
            "The capital of France is", return_tensors=None
        )["input_ids"]
    except Exception:
        prompt_ids = [785, 6722, 315, 9625, 374]

    with torch.inference_mode():
        engine.prefill(prompt_ids)
        for _ in range(3):
            engine.step(greedy=True)
        torch.cuda.synchronize()

        t.clear()
        c.clear()
        t0 = time.perf_counter()
        for _ in range(STEPS):
            engine.step(greedy=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0

    print(f"wall: {wall / STEPS * 1000:.2f} ms/step over {STEPS} steps")
    for name in sorted(t, key=lambda k: -t[k]):
        n = max(c[name], 1)
        print(f"  {name:14s} {t[name] / STEPS * 1000:8.3f} ms/step "
              f"({t[name] / n * 1e6:8.1f} us/call, {c[name]} calls)")


if __name__ == "__main__":
    main()
