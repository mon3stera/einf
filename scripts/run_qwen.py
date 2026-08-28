from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen2.5 through einf")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-len", type=int, default=8)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--block-len", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--max-batch-len", type=int, default=128)
    parser.add_argument("--max-prefill-chunk-len", type=int, default=128)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--paged-decode-attention", action="store_true")
    parser.add_argument("--paged-decode-max-splits", type=int, default=64)
    parser.add_argument("--compare-hf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    model_dir = args.model_dir.expanduser().resolve()
    config = QwenConfig.from_json(model_dir / "config.json")
    dtype = DTYPES[args.dtype]
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompt_ids = tuple(
        tokenizer(args.prompt, return_tensors="pt").input_ids[0].tolist()
    )

    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=args.num_blocks,
            block_len=args.block_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=dtype,
        device=device,
        use_custom_ops=True,
    )
    runner = QwenModelRunner(
        config,
        cache=cache,
        use_flash_attention=args.flash_attention,
        use_paged_decode_attention=args.paged_decode_attention,
        paged_decode_max_splits=args.paged_decode_max_splits,
    ).to(
        device=device,
        dtype=dtype,
    ).eval()
    runner.load_checkpoint(model_dir / "model.safetensors")

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

    generated_ids = request.generated_token_ids
    print("prompt_ids:", list(prompt_ids))
    print("generated_ids:", generated_ids)
    print("generated_text:", repr(tokenizer.decode(generated_ids)))
    print("completion_reason:", request.completion_reason)

    if args.compare_hf:
        reference = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
        ).to(device).eval()
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
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
