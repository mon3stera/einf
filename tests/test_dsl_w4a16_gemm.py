"""第 4 级 W4A16 GEMM 脚手架测试。

正确性测试使用 ``xfail(raises=NotImplementedError, strict=False)``：

    未实现      -> XFAIL
    实现且正确  -> XPASS
    实现但算错  -> FAIL（AssertionError 不匹配 NotImplementedError）

实现通过后必须摘除 xfail 标记。
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.dsl import dsl_available
from einf.executors.torch.dsl.w4a16_pack import fakequant_w4a16, pack_w4a16_marlin

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not dsl_available(),
    reason="CuTe DSL (nvidia-cutlass-dsl) and a CUDA device are required",
)

_gemm_scaffold = pytest.mark.xfail(
    reason="第 4 级练习：W4A16 GEMM kernel 尚未实现（实现正确后自动 XPASS）",
    raises=NotImplementedError,
    strict=False,
)


def _reference(A: torch.Tensor, weight_kn: torch.Tensor, group_size: int) -> torch.Tensor:
    dequant = fakequant_w4a16(weight_kn, group_size=group_size)
    return (A.float() @ dequant.float()).to(torch.bfloat16)


@_gemm_scaffold
@pytest.mark.parametrize(
    ("m", "k", "n", "group_size"),
    [
        (16, 128, 64, 128),
        (32, 128, 128, 128),
        (16, 256, 256, 128),
        (64, 128, 64, 64),
    ],
)
def test_cute_w4a16_gemm_matches_fakequant(
    m: int, k: int, n: int, group_size: int
) -> None:
    from einf.executors.torch.dsl import cute_w4a16_gemm

    torch.manual_seed(0)
    A = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
    qweight, scales = pack_w4a16_marlin(weight, group_size=group_size)
    actual = cute_w4a16_gemm(A, qweight, scales, group_size=group_size)
    expected = _reference(A, weight, group_size)

    assert actual.dtype is torch.bfloat16
    assert tuple(actual.shape) == (m, n)
    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1.6e-2)


@_gemm_scaffold
def test_cute_w4a16_gemm_writes_into_out() -> None:
    from einf.executors.torch.dsl import cute_w4a16_gemm

    torch.manual_seed(2)
    A = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    qweight, scales = pack_w4a16_marlin(weight, group_size=128)
    out = torch.empty((16, 64), device="cuda", dtype=torch.bfloat16)
    result = cute_w4a16_gemm(A, qweight, scales, group_size=128, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, _reference(A, weight, 128), rtol=1.6e-2, atol=1.6e-2)


def test_cute_w4a16_gemm_rejects_zeros_and_bad_shapes() -> None:
    from einf.executors.torch.dsl import cute_w4a16_gemm

    A = torch.randn((16, 128), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((128, 64), device="cuda", dtype=torch.bfloat16)
    qweight, scales = pack_w4a16_marlin(weight, group_size=128)
    zeros = torch.full_like(scales, 8)
    with pytest.raises(ValueError, match="zeros"):
        cute_w4a16_gemm(A, qweight, scales, zeros, group_size=128)
    with pytest.raises(ValueError, match="multiple of 16"):
        cute_w4a16_gemm(A[:15], qweight, scales, group_size=128)
