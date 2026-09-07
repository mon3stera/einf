"""W4A16 Marlin pack/unpack tests. These do not need CUDA or CuTe DSL."""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.dsl.w4a16_pack import (
    MARLIN_WEIGHT_INV_PERM,
    MARLIN_WEIGHT_PERM,
    PERM_ELEMS,
    compute_w4a16_scales,
    fakequant_w4a16,
    pack_w4a16_marlin,
    unpack_w4a16_marlin,
)


def test_marlin_perm_is_a_permutation() -> None:
    assert MARLIN_WEIGHT_PERM.shape == (PERM_ELEMS,)
    assert torch.equal(
        MARLIN_WEIGHT_PERM.sort().values,
        torch.arange(PERM_ELEMS, dtype=torch.int64),
    )
    assert torch.equal(
        MARLIN_WEIGHT_PERM[MARLIN_WEIGHT_INV_PERM],
        torch.arange(PERM_ELEMS, dtype=torch.int64),
    )


@pytest.mark.parametrize(
    ("k", "n", "group_size"),
    [
        (16, 64, 16),
        (128, 64, 128),
        (128, 128, 128),
        (256, 256, 128),
        (128, 256, 64),
    ],
)
def test_pack_unpack_matches_fakequant(k: int, n: int, group_size: int) -> None:
    torch.manual_seed(0)
    weight = torch.randn((k, n), dtype=torch.bfloat16)
    qweight, scales = pack_w4a16_marlin(weight, group_size=group_size)
    assert qweight.dtype == torch.int32
    assert tuple(qweight.shape) == (k // 16, n * 2)
    assert tuple(scales.shape) == (k // group_size, n)

    restored = unpack_w4a16_marlin(qweight, scales, group_size=group_size)
    expected = fakequant_w4a16(weight, group_size=group_size)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)

    scale_rows = scales.float().repeat_interleave(group_size, dim=0)
    quantized = torch.round(weight.float() / scale_rows).clamp(-8, 7)
    reference = (quantized * scale_rows).to(torch.bfloat16)
    torch.testing.assert_close(restored, reference, rtol=1e-3, atol=1e-3)


def test_pack_uses_supplied_scales() -> None:
    torch.manual_seed(1)
    weight = torch.randn((128, 64), dtype=torch.bfloat16)
    scales = compute_w4a16_scales(weight, group_size=128)
    qweight, packed_scales = pack_w4a16_marlin(
        weight, scales, group_size=128
    )
    assert torch.equal(packed_scales, scales)
    restored = unpack_w4a16_marlin(qweight, packed_scales, group_size=128)
    torch.testing.assert_close(
        restored,
        fakequant_w4a16(weight, group_size=128, scales=scales),
        rtol=0,
        atol=0,
    )


def test_pack_rejects_unaligned_shapes() -> None:
    weight = torch.randn((15, 64), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 16"):
        pack_w4a16_marlin(weight, group_size=16)
    weight = torch.randn((16, 32), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 64"):
        pack_w4a16_marlin(weight, group_size=16)
    weight = torch.randn((128, 64), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="divisible"):
        pack_w4a16_marlin(weight, group_size=96)
