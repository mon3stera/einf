from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from einf.executors.torch.flashinfer_attn import flashinfer_available


def use_flashinfer_fused(tensor: Tensor) -> bool:
    return (
        tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16)
        and flashinfer_available()
    )


def _flashinfer_version_tuple() -> tuple[int, ...]:
    import flashinfer

    parts: list[int] = []

    for part in flashinfer.__version__.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())

        if digits:
            parts.append(int(digits))

    return tuple(parts)


def use_flashinfer_silu(tensor: Tensor) -> bool:
    """Delegate silu_and_mul only on flashinfer versions verified on this
    stack. FlashInfer 0.6.18's tvm_ffi silu_and_mul segfaults inside
    cudaLaunchKernelExC at every shape (4090, JIT via nvcc 12.9), while
    0.6.4 is validated in production; fall back to the Triton kernel or the
    reference implementation elsewhere."""
    return use_flashinfer_fused(tensor) and _flashinfer_version_tuple() < (0, 6, 5)


def rmsnorm(input: Tensor, weight: Tensor, eps: float) -> Tensor:
    if use_flashinfer_fused(input):
        import flashinfer

        return flashinfer.rmsnorm(input.contiguous(), weight, eps)
    variance = input.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = input * torch.rsqrt(variance + eps)
    return (weight * normalized).to(dtype=input.dtype)


def fused_add_rmsnorm(
    input: Tensor,
    residual: Tensor,
    weight: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """``residual += input`` then ``input = rmsnorm(residual) * weight``.

    FlashInfer writes both tensors in place. The PyTorch fallback returns new
    tensors with the same math so CPU tests stay on the original path.
    """
    if use_flashinfer_fused(input):
        import flashinfer

        flashinfer.fused_add_rmsnorm(input, residual, weight, eps)
        return input, residual
    residual = residual + input
    return rmsnorm(residual, weight, eps), residual


def silu_and_mul(gate_up: Tensor) -> Tensor:
    if use_flashinfer_silu(gate_up):
        import flashinfer

        return flashinfer.silu_and_mul(gate_up)
    hidden = gate_up.shape[-1] // 2
    return F.silu(gate_up[..., :hidden]) * gate_up[..., hidden:]


def apply_rope_inplace(
    query: Tensor,
    key: Tensor,
    position: Tensor,
    *,
    rope_theta: float,
) -> None:
    import flashinfer

    flashinfer.apply_rope_pos_ids_inplace(
        query,
        key,
        position.to(dtype=torch.int32),
        interleave=False,
        rope_scale=1.0,
        rope_theta=rope_theta,
    )
