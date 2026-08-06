from __future__ import annotations

from pathlib import Path
from threading import Lock

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

_EXTENSION_NAME = "einf_torch_ops"
_LOAD_LOCK = Lock()
_LOADED = False
_REQUIRED_OPS = (
    "write_slots_",
    "gather_context",
    "contiguous_attention",
    "flash_attention",
    "paged_decode_attention",
    "paged_decode_attention_split_kv",
)


def custom_ops_available() -> bool:
    """Return whether all required einf operator schemas are registered."""
    return all(hasattr(torch.ops.einf, name) for name in _REQUIRED_OPS)


def load_custom_ops(*, verbose: bool = False) -> None:
    """JIT-build and load the einf Torch C++/CUDA extension once per process."""
    global _LOADED

    if _LOADED or custom_ops_available():
        _LOADED = True
        return

    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit is required to build einf custom ops")

    with _LOAD_LOCK:
        if _LOADED or custom_ops_available():
            _LOADED = True
            return

        csrc = Path(__file__).resolve().parent / "csrc"
        sources = [
            str(csrc / "kv_cache.cpp"),
            str(csrc / "kv_cache_cuda.cu"),
            str(csrc / "gather_context.cpp"),
            str(csrc / "gather_context_cuda.cu"),
            str(csrc / "contiguous_attention.cpp"),
            str(csrc / "contiguous_attention_cuda.cu"),
            str(csrc / "flash_attention.cpp"),
            str(csrc / "flash_attention_cuda.cu"),
            str(csrc / "paged_attention.cpp"),
            str(csrc / "paged_attention_cuda.cu"),
            str(csrc / "paged_attention_split_kv.cpp"),
            str(csrc / "paged_attention_split_kv_cuda.cu"),
        ]

        load(
            name=_EXTENSION_NAME,
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )

        if not custom_ops_available():
            raise RuntimeError("einf custom ops loaded without registering all schemas")

        _LOADED = True
