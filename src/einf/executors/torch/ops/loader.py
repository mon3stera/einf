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
    "cute_copy",
    "cute_elementwise_add",
    "cute_gemm",
    "cute_mma_qk",
    "cute_reduce_sum",
    "cute_shared_copy",
    "cute_transpose",
    "flash_attention",
    "tensor_core_qk",
    "paged_decode_attention",
    "paged_decode_attention_split_kv",
    "paged_decode_attention_batched",
    "marlin_gemm",
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

        ops_dir = Path(__file__).resolve().parent
        csrc = ops_dir / "csrc"
        project_root = next(
            (
                parent
                for parent in ops_dir.parents
                if (parent / "pyproject.toml").is_file()
            ),
            None,
        )
        if project_root is None:
            raise RuntimeError("could not locate the einf project root for CUTLASS")

        cutlass_include = project_root / "third_party" / "cutlass" / "include"
        if not (cutlass_include / "cute" / "tensor.hpp").is_file():
            raise RuntimeError(
                "CUTLASS/CuTe headers are missing; run "
                "`git submodule update --init third_party/cutlass`"
            )

        sources = [
            str(csrc / "kv_cache.cpp"),
            str(csrc / "kv_cache_cuda.cu"),
            str(csrc / "gather_context.cpp"),
            str(csrc / "gather_context_cuda.cu"),
            str(csrc / "contiguous_attention.cpp"),
            str(csrc / "contiguous_attention_cuda.cu"),
            str(csrc / "marlin" / "marlin_cuda_kernel.cu"),
            str(csrc / "marlin_gemm.cpp"),
            str(csrc / "marlin_gemm_cuda.cu"),
            str(csrc / "cute_copy.cpp"),
            str(csrc / "cute_copy_cuda.cu"),
            str(csrc / "cute_elementwise_add.cpp"),
            str(csrc / "cute_elementwise_add_cuda.cu"),
            str(csrc / "cute_gemm.cpp"),
            str(csrc / "cute_gemm_cuda.cu"),
            str(csrc / "cute_mma_qk.cpp"),
            str(csrc / "cute_mma_qk_cuda.cu"),
            str(csrc / "cute_reduce_sum.cpp"),
            str(csrc / "cute_reduce_sum_cuda.cu"),
            str(csrc / "cute_shared_copy.cpp"),
            str(csrc / "cute_shared_copy_cuda.cu"),
            str(csrc / "cute_transpose.cpp"),
            str(csrc / "cute_transpose_cuda.cu"),
            str(csrc / "flash_attention.cpp"),
            str(csrc / "flash_attention_cuda.cu"),
            str(csrc / "tensor_core_qk.cpp"),
            str(csrc / "tensor_core_qk_cuda.cu"),
            str(csrc / "paged_attention.cpp"),
            str(csrc / "paged_attention_cuda.cu"),
            str(csrc / "paged_attention_split_kv.cpp"),
            str(csrc / "paged_attention_split_kv_cuda.cu"),
            str(csrc / "paged_attention_batched.cpp"),
            str(csrc / "paged_attention_batched_cuda.cu"),
        ]

        load(
            name=_EXTENSION_NAME,
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            extra_include_paths=[str(cutlass_include)],
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )

        if not custom_ops_available():
            raise RuntimeError("einf custom ops loaded without registering all schemas")

        _LOADED = True
