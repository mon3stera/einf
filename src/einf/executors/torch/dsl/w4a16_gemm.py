"""第 4 级：W4A16 Marlin-layout GEMM（CuTe DSL 学习脚手架）。

阶梯位置
--------
    pack        w4a16_pack.py     逻辑 [K,N] → 物理 int32[K/16, N*2]     [已完成]
    本文件      w4a16_gemm.py     完整 (M,N,K,group) 的 decode GEMM       [练习]

契约
----
    A        : (M, K)     BF16 CUDA     激活，行主序
    qweight  : (K/16, N*2) int32 CUDA   ``pack_w4a16_marlin`` 的物理 B
    scales   : (K/group_size, N) BF16   逻辑 N，未做 Marlin scale permute
    zeros    : 本级不用；反量化固定 ``w = scale * (q - 8)``
    group_size : 128，且 ``K % group_size == 0``
    ->
    C        : (M, N)     BF16          ``C = A @ dequant(B)``，与 ``F.linear`` 的
                                        ``x @ weight.T`` 一致（weight 是 [N,K]）

对齐（本级不做 predication）
----------------------------
    M % 16 == 0
    K % 16 == 0
    N % 64 == 0
    K % group_size == 0
    group_size % 16 == 0

CTA / 数据划分
--------------
    1 warp = 32 threads
    1 CTA  = 1 warp
    一块输出 tile = 16×64   （M_TILE × PERM_N，正好一块 1024 INT4）
    grid = (N/64, M/16)

    每个 K tile（16）上：
        A  : gA 的 (16, 16) 逻辑块，``partition_A`` + ``autovec_copy``
        B  : ``qweight[k16, n_tile*128 : n_tile*128+128]`` 共 128 个 int32
             thread i 读 4 个连续 int32（16B）= 自己的 32 个 INT4
        反量化后的 B 逻辑块是 MMA 的 (N, K) = (64, 16)

    K 方向在 kernel 内循环 ``num_k_tiles = K/16``。

需要实现的关键变换
------------------
1. 从 4 个 int32 解 32 个 nibble（低位先），下标
   ``packed = 32 * lane + 8 * j + nib``。
2. ``gInv[packed]`` 映回 16×64 tile-major 线性下标，再拆
   ``(k_inner, n_local)``，scatter 进 smem ``sB[n_local, k_inner]``。
3. ``partition_B(sB)`` 之后就是普通 SM80 ``m16n8k16``；
   ``make_fragment_C((16, 64))`` 会在 N 上重复 8 个 atom。
4. 用 ``partition_C(make_identity_tensor((16, 64)))`` 核对写回 (m, n)，
   不要手写 lane 公式。
5. scale 用逻辑下标 ``gS[k16 * 16 / group_size, n_tile * 64 + n_local]``。

本级故意把 B 反量化进 smem 再 MMA，先把 layout 做对。寄存器 fused dequant
（Marlin 的 lop3 → FragB）是下一小级，不要现在写进这条路径。

注意：本 DSL 文件不能加入 ``from __future__ import annotations``。
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp as warp
import torch
from cutlass.cute.runtime import from_dlpack

from einf.executors.torch.dsl.w4a16_pack import (
    MARLIN_WEIGHT_INV_PERM,
    PACK_NIBBLES,
    PERM_N,
    TILE_K,
)

__all__ = ["cute_w4a16_gemm"]

M_TILE = 16
N_TILE = PERM_N
K_TILE = TILE_K
SM80_MMA_M = 16
SM80_MMA_N = 8
SM80_MMA_K = 16
INT32_PER_THREAD = 4
QWEIGHT_COLS_PER_N_TILE = N_TILE * TILE_K // PACK_NIBBLES  # 128


@cute.kernel
def _w4a16_gemm_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gS: cute.Tensor,
    gC: cute.Tensor,
    gInv: cute.Tensor,
    tiled_mma: cute.TiledMma,
    group_size: cutlass.Constexpr,
    num_k_tiles: cutlass.Constexpr,
) -> None:
    """One warp computes a 16×64 output tile, looping over K/16 packed tiles.

    TODO(第 4 级练习)：按下面七步实现。``print(x.layout)`` 在 @cute.kernel
    里是编译期打印；用 identity tensor 核对 B/C 归属，不要手写 lane 公式。
    """
    lane, _, _ = cute.arch.thread_idx()
    n_tile, m_tile, _ = cute.arch.block_idx()
    thr_mma = tiled_mma.get_slice(lane)

    # ---- 1. Output accumulator over the 16×64 tile ----
    # tCgC = thr_mma.partition_C(local_tile(gC, (M_TILE, N_TILE), (m_tile, n_tile)))
    # tCrC = thr_mma.make_fragment_C(thr_mma.partition_shape_C((M_TILE, N_TILE)))
    # tCrC.fill(0.0)
    # tCcC = thr_mma.partition_C(cute.make_identity_tensor((M_TILE, N_TILE)))
    # print(tCrC.layout) / print(tCcC)  核对这些 lane 的 (m, n)

    # ---- 2. Shared B tile after dequant: MMA B is (N, K) = (64, 16) ----
    # smem = cutlass.utils.SmemAllocator()
    # sB = smem.allocate_tensor(
    #     cutlass.BFloat16,
    #     cute.make_layout((N_TILE, K_TILE)),
    #     byte_alignment=16,
    # )
    # tBsB = thr_mma.partition_B(sB)
    # tCrB = thr_mma.make_fragment_B(tBsB)

    # ---- 3. K loop: num_k_tiles is constexpr ----
    # for k16 in cutlass.range_constexpr(num_k_tiles):
    #     gA_tile = cute.local_tile(gA, (M_TILE, K_TILE), (m_tile, k16))
    #     tAgA = thr_mma.partition_A(gA_tile)
    #     tCrA = thr_mma.make_fragment_A(tAgA)
    #     cute.autovec_copy(tAgA, tCrA)

    # ---- 4. Load this thread's 4 packed int32 (16B) ----
    # gB_tile = cute.local_tile(gB, (1, QWEIGHT_COLS_PER_N_TILE), (k16, n_tile))
    # col0 = INT32_PER_THREAD * lane
    # for j in cutlass.range_constexpr(INT32_PER_THREAD):
    #     word = gB_tile[0, col0 + j]

    # ---- 5. Unpack nibbles, inverse-perm scatter into sB ----
    # for nib in cutlass.range_constexpr(PACK_NIBBLES):
    #     q = (word >> (4 * nib)) & 15
    #     packed = 32 * lane + PACK_NIBBLES * j + nib
    #     logical = gInv[packed]            # 0..1023 in one 16×64 group
    #     tile_j = logical // (TILE_K * TILE_N)
    #     inner = logical % (TILE_K * TILE_N)
    #     k_inner = inner // TILE_N
    #     n_inner = inner % TILE_N
    #     n_local = tile_j * TILE_N + n_inner
    #     group = (k16 * K_TILE) // group_size
    #     scale = gS[group, n_tile * N_TILE + n_local]
    #     sB[n_local, k_inner] = (Float32(q) - 8) * scale   # then to BF16
    # cute.arch.sync_threads()

    # ---- 6. MMA: tCrC += tCrA @ tCrB ----
    # cute.autovec_copy(tBsB, tCrB)
    # cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # ---- 7. Convert FP32 acc to BF16 and store the C tile ----
    # rC = cute.make_fragment_like(tCrC, cutlass.BFloat16)
    # rC.store(tCrC.load().to(cutlass.BFloat16))
    # cute.autovec_copy(rC, tCgC)

    _ = (gA, gB, gS, gC, gInv, tiled_mma, group_size, num_k_tiles, thr_mma, n_tile, m_tile)


@cute.jit
def _w4a16_gemm_launch(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mS: cute.Tensor,
    mC: cute.Tensor,
    mInv: cute.Tensor,
    group_size: cutlass.Constexpr,
    num_k_tiles: cutlass.Constexpr,
) -> None:
    """One-warp 16×64 tiles; atom stays m16n8k16, N=64 is fragment repeat."""
    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(
            warp.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (SM80_MMA_M, SM80_MMA_N, SM80_MMA_K),
            )
        )
    )
    _w4a16_gemm_kernel(
        mA,
        mB,
        mS,
        mC,
        mInv,
        tiled_mma,
        group_size,
        num_k_tiles,
    ).launch(
        grid=(
            cute.ceil_div(cute.size(mC.shape[1]), N_TILE),
            cute.ceil_div(cute.size(mC.shape[0]), M_TILE),
            1,
        ),
        block=(cute.arch.WARP_SIZE, 1, 1),
    )


def cute_w4a16_gemm(
    A: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    *,
    group_size: int = 128,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``C = A @ dequant(qweight)`` for Marlin-packed W4A16 weights.

    Args:
        A: ``[M, K]`` BF16 CUDA activations.
        qweight: ``[K/16, N*2]`` int32 packed B from ``pack_w4a16_marlin``.
        scales: ``[K/group_size, N]`` BF16 logical scales.
        zeros: reserved; this rung uses a fixed zero-point of 8.
        group_size: quantization group along K; must divide K and be a
            multiple of 16. Default 128.
        out: optional preallocated ``[M, N]`` BF16 output (CUDA-graph friendly).
    """
    if zeros is not None:
        raise ValueError("cute_w4a16_gemm v1 ignores zeros; pass zeros=None (zp=8)")
    if not A.is_cuda or not qweight.is_cuda or not scales.is_cuda:
        raise ValueError("A, qweight, and scales must be CUDA tensors")
    if A.device != qweight.device or A.device != scales.device:
        raise ValueError("A, qweight, and scales must be on the same CUDA device")
    if A.dtype is not torch.bfloat16:
        raise ValueError("cute_w4a16_gemm requires BF16 A")
    if qweight.dtype != torch.int32:
        raise ValueError("qweight must be int32")
    if scales.dtype is not torch.bfloat16:
        raise ValueError("cute_w4a16_gemm requires BF16 scales")
    if A.dim() != 2 or qweight.dim() != 2 or scales.dim() != 2:
        raise ValueError("A, qweight, and scales must be rank-2")

    m, k = int(A.shape[0]), int(A.shape[1])
    if k % TILE_K != 0 or int(qweight.shape[0]) != k // TILE_K:
        raise ValueError(
            f"qweight.shape[0] must be K/{TILE_K}={k // TILE_K}, got {int(qweight.shape[0])}"
        )
    n = int(qweight.shape[1]) * PACK_NIBBLES // TILE_K
    if n % N_TILE != 0:
        raise ValueError(f"N must be a multiple of {N_TILE}, got {n}")
    if m % M_TILE != 0:
        raise ValueError(f"M must be a multiple of {M_TILE}, got {m}")
    if k % group_size != 0 or group_size <= 0 or group_size % TILE_K != 0:
        raise ValueError(
            f"group_size must be a positive multiple of {TILE_K} that divides K={k}"
        )
    if tuple(scales.shape) != (k // group_size, n):
        raise ValueError(
            f"scales must have shape ({k // group_size}, {n}), got {tuple(scales.shape)}"
        )
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    elif (
        not out.is_cuda
        or out.device != A.device
        or out.dtype is not torch.bfloat16
        or tuple(out.shape) != (m, n)
    ):
        raise ValueError(f"out must be a CUDA BF16 tensor of shape {(m, n)}")

    raise NotImplementedError("第 4 级练习：实现 _w4a16_gemm_kernel 后删除这行")

    inv = MARLIN_WEIGHT_INV_PERM.to(device=A.device, dtype=torch.int32)

    _w4a16_gemm_launch(
        from_dlpack(A.contiguous()),
        from_dlpack(qweight.contiguous()),
        from_dlpack(scales.contiguous()),
        from_dlpack(out),
        from_dlpack(inv.contiguous()),
        group_size,
        k // K_TILE,
    )
    return out
