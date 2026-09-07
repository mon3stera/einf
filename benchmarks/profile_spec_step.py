"""Profile the speculative step on the target hardware.

Builds the W4A16-7B target + bf16-0.5B draft pair, warms up the decode
graph, then profiles a handful of spec steps with record_function segment
markers so the profiler table attributes cost to:
  SPEC.step        one draft-then-verify cycle (total)
  SPEC.draft       the K+1 graph-replay decode loop
  SPEC.verify      the eager q=K+1 target forward
  SPEC.accept      accept/reject + bookkeeping
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch.flashinfer_attn import FlashInferPagedAttention  # noqa: F401
from einf.executors.torch.qwen import QwenConfig, QwenModelRunner
from einf.executors.torch.spec_runner import SpeculativeEngine

TARGET = Path(sys.argv[1])
DRAFT = Path(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
PROMPT = "The capital of France is"
K = 4

device = torch.device("cuda")


def build(config_path: Path, model_dir: Path, *, w4a16: bool) -> QwenModelRunner:
    config = QwenConfig.from_json(config_path)
    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=256,
            block_len=32,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
        use_custom_ops=True,
    )
    # Mirror run_qwen's construction: default dtype bf16 + device context so
    # nothing transiently materializes in fp32.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
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
        runner = runner.to(device=device, dtype=torch.bfloat16).eval()
    print("  weights on device", flush=True)
    runner.load_checkpoint(model_dir)
    print("  checkpoint loaded", flush=True)
    return runner


def main() -> None:
    torch.manual_seed(0)
    t0 = time.perf_counter()
    print("building target (W4A16)...", flush=True)
    target = build(TARGET / "config.json", TARGET, w4a16=True)
    print("building draft (bf16)...", flush=True)
    draft = build(DRAFT / "config.json", DRAFT, w4a16=False)
    print(f"load: {time.perf_counter() - t0:.1f}s", flush=True)

    engine = SpeculativeEngine(
        target, draft, device=device, block_len=32, num_blocks=256, num_spec_tokens=K
    )

    # Segment markers via instance-attribute wrapping.
    orig_draft = engine._draft_decode_forward

    def draft_wrapped(draft_track, token):
        with torch.profiler.record_function("SPEC.draft"):
            return orig_draft(draft_track, token)

    engine._draft_decode_forward = draft_wrapped

    orig_verify = engine._target.forward

    def verify_wrapped(model_input):
        with torch.profiler.record_function("SPEC.verify"):
            return orig_verify(model_input)

    engine._target.forward = verify_wrapped

    orig_accept = engine._accept_reject if hasattr(engine, "_accept_reject") else None
    if orig_accept is not None:
        def accept_wrapped(*args, **kwargs):
            with torch.profiler.record_function("SPEC.accept"):
                return orig_accept(*args, **kwargs)

        engine._accept_reject = accept_wrapped

    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(TARGET))
    except Exception:
        pass
    if tokenizer is not None:
        prompt_ids = tokenizer(PROMPT, return_tensors=None)["input_ids"]
    else:
        prompt_ids = [785, 6722, 315, 9625, 374]

    with torch.inference_mode():
        engine.prefill(prompt_ids)
        for _ in range(3):
            engine.step(greedy=True)  # warmup: capture graph, JIT kernels

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            for _ in range(STEPS):
                with torch.profiler.record_function("SPEC.step"):
                    engine.step(greedy=True)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t1

    print(f"profiled {STEPS} steps in {wall:.3f}s ({wall / STEPS * 1000:.1f} ms/step, "
          f"profiler overhead included)", flush=True)

    for sort_key in ("self_cuda_time_total", "self_cpu_time_total"):
        print(f"\n===== key_averages by {sort_key} =====", flush=True)
        table = prof.key_averages().table(
            sort_by=sort_key, row_limit=40, max_name_column_width=70
        )
        print(table, flush=True)

    print("done (trace export skipped)", flush=True)


if __name__ == "__main__":
    main()
