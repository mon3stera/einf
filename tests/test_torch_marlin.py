from __future__ import annotations

import pytest
import torch

from einf.executors.torch.dsl.w4a16_pack import (
    fakequant_w4a16,
    pack_w4a16_marlin,
    permute_scales_marlin,
    unpack_w4a16_marlin,
)
from einf.executors.torch.ops import load_custom_ops
from einf.executors.torch.w4a16 import MarlinW4A16Linear


def _ops_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        load_custom_ops()
    except Exception:  # pragma: no cover - depends on the build environment
        return False
    return True


requires_marlin = pytest.mark.skipif(
    not _ops_available(),
    reason="requires CUDA and the einf torch ops extension",
)


@requires_marlin
@pytest.mark.parametrize(("k", "n"), [(896, 1152), (4864, 896)])
@pytest.mark.parametrize("m", [1, 16, 17, 64, 100])
def test_marlin_gemm_matches_dequant_reference(k: int, n: int, m: int) -> None:
    torch.manual_seed(k * 1000 + n * 10 + m)
    device = torch.device("cuda")

    weight_kn = torch.randn(k, n, device=device) * 0.02
    qweight, scales = pack_w4a16_marlin(weight_kn, group_size=128)
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)

    out = torch.empty(m, n, device=device, dtype=torch.float16)
    workspace = torch.zeros(n // 128 * 8, dtype=torch.int32, device=device)
    torch.ops.einf.marlin_gemm(
        a.to(torch.float16),
        qweight,
        permute_scales_marlin(scales).to(torch.float16),
        out,
        workspace,
        group_size=128,
    )

    reference_weight = unpack_w4a16_marlin(
        qweight,
        scales.to(torch.float32),
        group_size=128,
    )
    reference = a.to(torch.float32) @ reference_weight
    torch.testing.assert_close(
        out.to(torch.float32),
        reference,
        atol=0.05,
        rtol=0.02,
    )


@requires_marlin
def test_marlin_gemm_rejects_undersized_workspace() -> None:
    device = torch.device("cuda")
    qweight, scales = pack_w4a16_marlin(
        torch.randn(128, 128, device=device) * 0.02,
        group_size=128,
    )
    out = torch.empty(1, 128, device=device, dtype=torch.float16)
    workspace = torch.zeros(1, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError, match="workspace"):
        torch.ops.einf.marlin_gemm(
            torch.randn(1, 128, device=device, dtype=torch.float16),
            qweight,
            permute_scales_marlin(scales).to(torch.float16),
            out,
            workspace,
            group_size=128,
        )


@requires_marlin
def test_marlin_linear_module_matches_fakequant() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")

    linear = torch.nn.Linear(128, 256, bias=True).to(device=device, dtype=torch.bfloat16)
    quantized = MarlinW4A16Linear.from_linear(linear)
    x = torch.randn(5, 128, device=device, dtype=torch.bfloat16)

    out = quantized(x)
    reference = torch.nn.functional.linear(
        x,
        fakequant_w4a16(linear.weight.detach().t(), group_size=128).T,
        linear.bias,
    )
    torch.testing.assert_close(out, reference, atol=0.05, rtol=0.02)


@requires_marlin
def test_marlin_linear_forward_rejects_cpu() -> None:
    linear = torch.nn.Linear(128, 256, bias=True)
    quantized = MarlinW4A16Linear.from_linear(linear)
    x = torch.randn(3, 128, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="CUDA"):
        quantized(x)
