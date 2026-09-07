from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
KV_DTYPES = {
    **DTYPES,
    "fp8_e4m3": torch.float8_e4m3fn,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5 through einf")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-len", type=int, default=8)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--kv-cache-dtype",
        choices=KV_DTYPES,
        default="auto",
        help="KV cache storage dtype; auto follows --dtype. fp8_e4m3 halves "
        "KV memory and read bandwidth and requires the flashinfer backend",
    )
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--max-batch-len", type=int, default=128)
    parser.add_argument("--max-prefill-chunk-len", type=int, default=128)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--paged-decode-attention", action="store_true")
    parser.add_argument("--paged-decode-max-splits", type=int, default=64)
    parser.add_argument(
        "--attn-backend",
        choices=("einf", "flashinfer"),
        default="einf",
        help="einf keeps in-house kernels; flashinfer uses paged BatchPrefill",
    )
    parser.add_argument("--compare-hf", action="store_true")
    parser.add_argument("--w4a16", action="store_true")
    parser.add_argument(
        "--spec-tokens",
        type=int,
        default=0,
        help="chain speculative decoding depth K (0 = off); bypasses the "
        "scheduler and runs the eager spec loop instead",
    )
    parser.add_argument(
        "--draft-model-dir",
        type=Path,
        default=None,
        help="draft model directory; defaults to --model-dir (self-speculation)",
    )
    return parser.parse_args()


def build_runner(config, cache, *, args, dtype, device, model_dir):
    """Construct, load, and eval a Qwen runner (shared by target and draft)."""
    # Large models must be constructed directly in the working dtype on the
    # device: default fp32 CPU construction transiently needs ~4 bytes/param
    # and OOM-kills a 7B model on this box (no swap).
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            runner = QwenModelRunner(
                config,
                cache=cache,
                use_flash_attention=args.flash_attention,
                use_paged_decode_attention=args.paged_decode_attention,
                use_flashinfer_attention=args.attn_backend == "flashinfer",
                paged_decode_max_splits=args.paged_decode_max_splits,
                dtype=dtype,
                w4a16=args.w4a16,
            )
    finally:
        torch.set_default_dtype(previous_dtype)
    if args.w4a16:
        # W4A16 masters stay on CPU through load_checkpoint; the runner moves
        # only the quantized weights to the device after quantization.
        runner = runner.eval()
    else:
        runner = runner.to(
            device=device,
            dtype=dtype,
        ).eval()
    runner.load_checkpoint(model_dir)
    return runner


def _run_scheduler_path(config, runner, args, prompt_ids, *, device) -> tuple[int, ...]:
    """Plain (non-speculative) serving path through the Rust control plane."""
    scheduler = Scheduler(
        policy="fcfs",
        num_blocks=args.num_blocks,
        block_len=args.block_len,
        max_batch_len=args.max_batch_len,
        max_prefill_chunk_len=args.max_prefill_chunk_len,
    )
    executor = TorchExecutor(
        model_runner=runner,
        block_len=args.block_len,
        eos_token_id=config.eos_token_id,
        device=device,
    )
    server = LLMServer(scheduler, executor)
    request_id = scheduler.submit(
        RequestSpec(
            request_id="qwen",
            prompt_token_ids=prompt_ids,
            max_new_len=args.max_new_len,
        )
    )

    with torch.inference_mode():
        server.run_until_idle()

    request = scheduler.request(request_id)
    if request.state is RequestState.FAILED:
        raise RuntimeError(request.error)

    print("completion_reason:", request.completion_reason)
    return request.generated_token_ids


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    model_dir = args.model_dir.expanduser().resolve()
    config = QwenConfig.from_json(model_dir / "config.json")
    dtype = DTYPES[args.dtype]
    kv_dtype = dtype if args.kv_cache_dtype == "auto" else KV_DTYPES[args.kv_cache_dtype]
    if kv_dtype != dtype and args.attn_backend != "flashinfer":
        raise SystemExit("--kv-cache-dtype=fp8_e4m3 requires --attn-backend=flashinfer")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt_ids = tuple(
        tokenizer(args.prompt, return_tensors="pt").input_ids[0].tolist()
    )

    cache_blocks = args.num_blocks + int(args.attn_backend == "flashinfer")
    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=cache_blocks,
            block_len=args.block_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=dtype,
        device=device,
        use_custom_ops=True,
        kv_dtype=kv_dtype,
    )
    if args.attn_backend == "flashinfer" and (
        args.flash_attention or args.paged_decode_attention
    ):
        raise SystemExit(
            "--attn-backend=flashinfer cannot be combined with in-house attention flags"
        )
    runner = build_runner(config, cache, args=args, dtype=dtype, device=device, model_dir=model_dir)

    if args.spec_tokens > 0:
        from einf.executors.torch.spec_runner import SpeculativeEngine

        draft_dir = (
            args.draft_model_dir or args.model_dir
        ).expanduser().resolve()
        draft_config = QwenConfig.from_json(draft_dir / "config.json")
        draft_cache = TorchKVCacheStorage(
            KVCacheGeometry(
                num_layers=draft_config.num_hidden_layers,
                num_blocks=cache_blocks,
                block_len=args.block_len,
                num_kv_heads=draft_config.num_key_value_heads,
                head_dim=draft_config.head_dim,
            ),
            dtype=dtype,
            device=device,
            use_custom_ops=True,
            kv_dtype=kv_dtype,
        )
        draft_runner = build_runner(
            draft_config,
            draft_cache,
            args=args,
            dtype=dtype,
            device=device,
            model_dir=draft_dir,
        )
        engine = SpeculativeEngine(
            runner,
            draft_runner,
            device=device,
            block_len=args.block_len,
            num_blocks=cache_blocks,
            num_spec_tokens=args.spec_tokens,
        )
        with torch.inference_mode():
            generated_ids, stats = engine.generate(
                list(prompt_ids),
                max_new_len=args.max_new_len,
                eos_token_id=config.eos_token_id,
                greedy=True,
            )
        print(
            "spec_stats:",
            f"steps={stats.steps} proposed={stats.proposed} "
            f"accepted={stats.accepted} rate={stats.accept_rate:.3f}",
        )
    else:
        generated_ids = _run_scheduler_path(
            config, runner, args, prompt_ids, device=device
        )

    print("prompt_ids:", list(prompt_ids))
    print("generated_ids:", generated_ids)
    print("generated_text:", repr(tokenizer.decode(generated_ids)))

    if args.compare_hf:
        index_path = model_dir / "model.safetensors.index.json"
        total_size = 0
        if index_path.is_file():
            total_size = json.loads(index_path.read_text())["metadata"]["total_size"]
        elif (model_dir / "model.safetensors").is_file():
            total_size = (model_dir / "model.safetensors").stat().st_size

        # The reference must coexist with the einf runner; when the weights
        # cannot fit twice in VRAM, compare against a CPU reference.
        vram_bytes = torch.cuda.get_device_properties(device).total_memory
        ref_device = device if total_size * 2 < vram_bytes else torch.device("cpu")

        reference = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
        ).to(ref_device).eval()
        print(f"hf_reference_device: {ref_device}")
        inputs = tokenizer(args.prompt, return_tensors="pt").to(ref_device)
        with torch.inference_mode():
            output_ids = reference.generate(
                **inputs,
                max_new_tokens=args.max_new_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        reference_ids = output_ids[0, inputs.input_ids.size(1) :].tolist()
        print("hf_generated_ids:", reference_ids)
        print("tokens_equal:", generated_ids == reference_ids)


if __name__ == "__main__":
    main()
