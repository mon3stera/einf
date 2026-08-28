"""CuTe DSL (Python) 内核。

与 ``einf.executors.torch.ops`` 的区别
------------------------------------
``ops`` 是 C++/CUDA torch extension，经 ``TORCH_LIBRARY`` 注册为 ``torch.ops.einf.*``，
由 nvcc 在首次调用时 JIT 构建。

本包是 CuTe DSL（``nvidia-cutlass-dsl``）写的内核，纯 Python，经 MLIR JIT，
用 ``from_dlpack`` 直接吃 torch 张量。两者互不依赖，可以并存。

DSL 不是所有环境都装了（本地开发机通常没有），所以这里用惰性导入：
``import einf`` 不会因为缺少 DSL 而失败，只有真正取用内核时才导入。
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "dsl_available",
    "cute_online_softmax",
    "cute_online_softmax_layout",
    "cute_flash_attention_single_tile",
    "cute_flash_attention",
]


def dsl_available() -> bool:
    """Return whether the CuTe DSL (nvidia-cutlass-dsl) can be imported."""
    try:
        importlib.import_module("cutlass.cute")
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def __getattr__(name: str) -> Any:
    """Import DSL kernels lazily so a missing DSL never breaks ``import einf``."""
    if name == "cute_online_softmax":
        from einf.executors.torch.dsl.online_softmax import cute_online_softmax

        return cute_online_softmax
    if name == "cute_online_softmax_layout":
        from einf.executors.torch.dsl.online_softmax_layout import (
            cute_online_softmax_layout,
        )

        return cute_online_softmax_layout
    if name == "cute_flash_attention_single_tile":
        from einf.executors.torch.dsl.flash_attention_single_tile import (
            cute_flash_attention_single_tile,
        )

        return cute_flash_attention_single_tile
    if name == "cute_flash_attention":
        from einf.executors.torch.dsl.flash_attention import cute_flash_attention

        return cute_flash_attention
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
