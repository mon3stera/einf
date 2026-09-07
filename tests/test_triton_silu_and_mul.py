"""Numerical tests for the hand-written Triton ``silu_and_mul`` kernel.

CPU cases always run and pin the reference math. CUDA cases run the kernel
itself against the reference; they are the acceptance test for the
``YOUR CODE HERE`` body in ``triton_ops.silu_and_mul_kernel``.
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.triton_ops import (
    silu_and_mul_reference,
    silu_and_mul_triton,
    triton_silu_and_mul_enabled,
)

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# hidden values cover: tiny, non-power-of-two (mask tail!), Qwen2.5-7B
# intermediate (18944), DeepSeek-V2-Lite intermediate (10944).
_HIDDENS = [1, 7, 100, 10944, 18944]


def _cases():
    for hidden in _HIDDENS:
        for dtype in _DTYPES:
            yield hidden, dtype


def test_reference_matches_torch_composition() -> None:
    gate_up = torch.randn(8, 2 * 64)
    expected = torch.nn.functional.silu(gate_up[..., :64]) * gate_up[..., 64:]
    actual = silu_and_mul_reference(gate_up)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("hidden,dtype", list(_cases()))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_kernel_matches_reference(hidden: int, dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(hidden)
    gate_up = torch.randn(
        (17, 2 * hidden),
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    ).to(dtype)

    actual = silu_and_mul_triton(gate_up)
    expected = silu_and_mul_reference(gate_up)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_env_gate_dispatch() -> None:
    gate_up = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

    import os

    previous = os.environ.get("EINF_TRITON_SILU")
    try:
        os.environ["EINF_TRITON_SILU"] = "0"
        assert not triton_silu_and_mul_enabled(gate_up)
        os.environ["EINF_TRITON_SILU"] = "1"
        assert triton_silu_and_mul_enabled(gate_up)
    finally:
        if previous is None:
            os.environ.pop("EINF_TRITON_SILU", None)
        else:
            os.environ["EINF_TRITON_SILU"] = previous
