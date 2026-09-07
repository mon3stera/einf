"""W4A16 Marlin-style host pack / unpack.

This is the **logical → physical** conversion. The GEMM kernel never sees
AWQ vs GPTQ; it only consumes ``qweight`` in this layout and logical
``scales[K/group, N]``.

Physical ``qweight``
-------------------
    int32[K/16, N*2]

    1. Quantize ``weight_kn[K, N]`` group-wise: ``q = clamp(round(w/s) + 8, 0, 15)``.
    2. Tile 16×16, concatenate along N as ``(n_tile, k_inner, n_inner)``.
    3. Every 1024 values (one ``k_tile`` × 64 N = four 16×16 tiles) apply
       ``MARLIN_WEIGHT_PERM`` (MMA B ownership + nibble interleave).
    4. Pack 8 consecutive INT4 into one int32, low nibble first.

``scales`` stay logical ``[K/group_size, N]``. The Marlin CUDA kernel
additionally expects an N-side 64-column interleave of the scales (its B
fragment threads read 8 scale columns in an interleaved order); use
``permute_scales_marlin`` for that kernel, while the DSL GEMM keeps the
logical layout.

``zeros`` are implicit 8 (symmetric unsigned INT4). The GEMM formula is
``w = scale * (q - 8)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

TILE_K = 16
TILE_N = 16
PERM_N = 64
PACK_NIBBLES = 8
PERM_ELEMS = TILE_K * PERM_N  # 1024
MAX_Q = 15
ZERO_POINT = 8


def _marlin_weight_perm() -> Tensor:
    """1024-entry gather index: tile-major 16×64 → MMA-B / dequant nibble order."""
    perm: list[int] = []
    for thread in range(32):
        col = thread // 4
        perm1: list[int] = []
        for block in (0, 1):
            tid = thread % 4
            for row in (
                2 * tid,
                2 * tid + 1,
                2 * tid + 8,
                2 * tid + 9,
            ):
                perm1.append(16 * row + col + 8 * block)
        for tile in range(4):
            perm.extend(index + 256 * tile for index in perm1)
    interleaved = torch.tensor(perm, dtype=torch.int64).reshape(-1, PACK_NIBBLES)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], dtype=torch.int64)
    return interleaved[:, order].reshape(-1)


MARLIN_WEIGHT_PERM = _marlin_weight_perm()
MARLIN_WEIGHT_INV_PERM = torch.empty_like(MARLIN_WEIGHT_PERM)
MARLIN_WEIGHT_INV_PERM[MARLIN_WEIGHT_PERM] = torch.arange(
    PERM_ELEMS, dtype=torch.int64
)


def _marlin_scale_perm() -> Tensor:
    """64-entry N interleave: output column c = 8*ci + cj reads input 8*cj + ci."""
    return torch.tensor(
        [i + 8 * j for i in range(8) for j in range(8)],
        dtype=torch.int64,
    )


MARLIN_SCALE_PERM = _marlin_scale_perm()


def permute_scales_marlin(scales: Tensor) -> Tensor:
    """Interleave ``[K/group_size, N]`` scales for the Marlin CUDA kernel.

    The kernel's B fragments pair each thread with 8 scale columns in an
    interleaved order, so the stored scale row must be permuted in blocks of
    64 logical N columns. The DSL GEMM keeps the logical layout instead.
    """
    if scales.dim() != 2:
        raise ValueError(f"scales must be 2-D [K/group, N], got {tuple(scales.shape)}")
    rows, n = int(scales.shape[0]), int(scales.shape[1])
    if n % PERM_N != 0:
        raise ValueError(f"N must be a multiple of {PERM_N} for scale permutation, got {n}")
    perm = MARLIN_SCALE_PERM.to(device=scales.device)
    return (
        scales.reshape(rows, n // PERM_N, PERM_N)[:, :, perm]
        .reshape(rows, n)
        .contiguous()
    )


def _require_pack_shapes(weight_kn: Tensor, group_size: int) -> tuple[int, int]:
    if weight_kn.dim() != 2:
        raise ValueError(f"weight_kn must be [K, N], got {tuple(weight_kn.shape)}")
    k, n = int(weight_kn.shape[0]), int(weight_kn.shape[1])
    if k == 0 or n == 0:
        raise ValueError("weight_kn must have positive K and N")
    if k % TILE_K != 0:
        raise ValueError(f"K must be a multiple of {TILE_K}, got {k}")
    if n % PERM_N != 0:
        raise ValueError(f"N must be a multiple of {PERM_N}, got {n}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if group_size % TILE_K != 0:
        raise ValueError(f"group_size must be a multiple of {TILE_K}, got {group_size}")
    if k % group_size != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    return k, n


def compute_w4a16_scales(weight_kn: Tensor, *, group_size: int = 128) -> Tensor:
    """Per-group absmax scales so ``round(w/s)`` fits signed INT4 ``[-8, 7]``."""
    k, n = _require_pack_shapes(weight_kn, group_size)
    grouped = weight_kn.detach().to(torch.float32).reshape(k // group_size, group_size, n)
    amax = grouped.abs().amax(dim=1).clamp_min(1e-8)
    return (amax / float(ZERO_POINT)).to(dtype=weight_kn.dtype)


def _broadcast_scales(scales: Tensor, k: int, n: int, group_size: int) -> Tensor:
    if scales.dim() == 1:
        scales = scales.view(1, -1)
    if scales.dim() != 2 or int(scales.shape[1]) != n:
        raise ValueError(
            f"scales must be [K/group_size, N] or [N], got {tuple(scales.shape)}"
        )
    groups = k // group_size
    if int(scales.shape[0]) == 1 and groups != 1:
        scales = scales.expand(groups, n)
    if int(scales.shape[0]) != groups:
        raise ValueError(
            f"scales.shape[0] must be 1 or K/group_size={groups}, got {int(scales.shape[0])}"
        )
    return scales


def pack_w4a16_marlin(
    weight_kn: Tensor,
    scales: Tensor | None = None,
    *,
    group_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Pack ``weight_kn[K, N]`` into Marlin ``qweight[K/16, N*2]``.

    ``weight_kn`` is ``nn.Linear.weight.T`` (the GEMM B matrix). Returns
    ``(qweight, scales)`` with logical scales ``[K/group_size, N]``.
    """
    k, n = _require_pack_shapes(weight_kn, group_size)
    if scales is None:
        scales = compute_w4a16_scales(weight_kn, group_size=group_size)
    scales = _broadcast_scales(scales, k, n, group_size).contiguous()
    if scales.device != weight_kn.device:
        raise ValueError("weight_kn and scales must be on the same device")

    scale_rows = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    quantized = torch.round(weight_kn.to(torch.float32) / scale_rows).to(torch.int32)
    quantized = (quantized + ZERO_POINT).clamp(0, MAX_Q)

    tiled = (
        quantized.reshape(k // TILE_K, TILE_K, n // TILE_N, TILE_N)
        .permute(0, 2, 1, 3)
        .reshape(k // TILE_K, n * TILE_K)
        .contiguous()
    )
    perm = MARLIN_WEIGHT_PERM.to(device=tiled.device)
    tiled = tiled.reshape(-1, PERM_ELEMS)[:, perm].reshape(k // TILE_K, n * TILE_K)

    qweight = torch.zeros(
        (k // TILE_K, n * TILE_K // PACK_NIBBLES),
        dtype=torch.int32,
        device=weight_kn.device,
    )
    for nibble in range(PACK_NIBBLES):
        qweight.bitwise_or_(tiled[:, nibble::PACK_NIBBLES].bitwise_and(MAX_Q) << (4 * nibble))
    return qweight, scales.detach().clone()


def unpack_w4a16_marlin(
    qweight: Tensor,
    scales: Tensor,
    *,
    group_size: int = 128,
) -> Tensor:
    """Invert ``pack_w4a16_marlin`` to a dequantized ``[K, N]`` tensor."""
    if qweight.dim() != 2 or qweight.dtype != torch.int32:
        raise ValueError("qweight must be an int32 [K/16, N*2] tensor")
    k = int(qweight.shape[0]) * TILE_K
    n = int(qweight.shape[1]) * PACK_NIBBLES // TILE_K
    _require_pack_shapes(torch.empty((k, n), device=qweight.device), group_size)
    scales = _broadcast_scales(scales, k, n, group_size)

    unpacked = torch.zeros(
        (int(qweight.shape[0]), n * TILE_K),
        dtype=torch.int32,
        device=qweight.device,
    )
    for nibble in range(PACK_NIBBLES):
        unpacked[:, nibble::PACK_NIBBLES] = (qweight >> (4 * nibble)) & MAX_Q

    inv = MARLIN_WEIGHT_INV_PERM.to(device=qweight.device)
    logical = unpacked.reshape(-1, PERM_ELEMS)[:, inv].reshape(qweight.shape[0], n * TILE_K)
    quantized = (
        logical.reshape(qweight.shape[0], n // TILE_N, TILE_K, TILE_N)
        .permute(0, 2, 1, 3)
        .reshape(k, n)
    )
    scale_rows = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    return ((quantized.to(torch.float32) - float(ZERO_POINT)) * scale_rows).to(scales.dtype)


def fakequant_w4a16(
    weight_kn: Tensor,
    *,
    group_size: int = 128,
    scales: Tensor | None = None,
) -> Tensor:
    """Pack then unpack; the W4A16 GEMM reference should match ``A @ this``."""
    qweight, packed_scales = pack_w4a16_marlin(
        weight_kn, scales, group_size=group_size
    )
    return unpack_w4a16_marlin(qweight, packed_scales, group_size=group_size)
