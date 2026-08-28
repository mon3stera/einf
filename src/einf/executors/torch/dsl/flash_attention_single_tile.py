"""第 3.0 级：单个 K/V tile 的最小 P @ V（CuTe DSL 学习脚手架）。

阶梯位置
--------
    第 2 级    online_softmax.py                 手算 SM80 online softmax       [已完成]
    第 2.5 级  online_softmax_layout.py          layout/policy + 通用 softmax  [已完成]
    第 3.0 级  flash_attention_single_tile.py    S -> P -> P @ V               [本文件]
    第 3.1 级  flash attention KV loop           O 的 online rescale            [待做]

本级只增加一个难点：把 QK 的 FP32 C accumulator 原地变成 P，再把它转换为第二个
Tensor Core GEMM 接受的 BF16 A fragment，立即执行 P @ V。刻意只有一个 16-token
K/V tile，因此没有 KV 循环，也没有 ``O *= exp(m_old - m_new)``；那是下一小级。

契约
----
    Q : (32, 16)  BF16
    K : (16, 16)  BF16
    V : (16, 16)  BF16
    ->
    O : (32, 16)  FP32

数学目标是 ``softmax(Q.float() @ K.float().T, dim=-1) @ V.float()``。不过真实 Tensor
Core 路径在 P @ V 前必须将未归一化的 ``exp(S-m)`` 转成 BF16，因此测试同时固定：

    E     = exp(S - row_max)
    P_bf16 = E.to(bfloat16)
    O     = (P_bf16.float() @ V.float()) / E.sum(dim=-1)

这使容差能够收紧并明确区分算法错误与设计中刻意保留的 BF16 P 量化。

需要实现的关键变换
------------------
QK 的 C fragment 是 ``((V0,V1), MMA_M, MMA_N)``，当前 32x16 tile 为
``((2,2),2,2)``。P @ V 的 A fragment 需要 reduction-K=16，故 SM80 的 back-to-back
GEMM view 要做：

    (4, MMA_M, MMA_N)
      -> logical_divide mode MMA_N by 2
      -> ((4, 2), MMA_M, MMA_N / 2)

这里只重解释同一份 P 寄存器，不做寄存器间搬运。V 的 host tensor 是 row-major
``(reduction_N, output_D)``；MMA B 的逻辑 tensor 是 ``(output_D, reduction_N)``，所以
要创建 stride ``(1, HEAD_DIM)`` 的零拷贝 transpose view，而不能真的转置 V。

注意：本 DSL 文件不能加入 ``from __future__ import annotations``，否则
``cutlass.Constexpr`` 注解会失去 DSL 编译期语义。
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp as warp
import torch
from cutlass.cute.runtime import from_dlpack

from einf.executors.torch.dsl.online_softmax_layout import (
    _online_softmax_update,
    _sm80_m16n8_acc_rowcol_policy,
)

__all__ = ["cute_flash_attention_single_tile"]

M_QUERIES = 32
N_TILE = 16
HEAD_DIM = 16
NEG_INF = -1.0e30


@cute.jit
def _sm80_accumulator_to_fragment_a(acc):
    """TODO: Re-view an SM80 QK accumulator as the A fragment for P @ V.

    Checkpoint for this exercise:

    ``acc.layout``
        ``((2,2),2,2):((1,2),4,8)``

    returned layout
        ``(((2,2),2),2,1):(((1,2),8),4,0)``

    Use ``cute.logical_divide(acc.layout, (None, None, 2))`` and regroup the
    resulting shape/stride. Return ``cute.make_tensor(acc.iterator, new_layout)``.
    This is a view, not a copy.
    """
    divided = cute.logical_divide(acc.layout, (None, None, 2))
    v_shape = cute.flatten_to_tuple(divided.shape[0])
    v_stride = cute.flatten_to_tuple(divided.stride[0])
    fragment_a_layout = cute.make_layout(
        (
            (v_shape[0], v_shape[1], divided.shape[2][0]),
            divided.shape[1],
            divided.shape[2][1],
        ),
        stride = (
            (v_stride[0], v_stride[1], divided.stride[2][0]),
            divided.stride[1],
            divided.stride[2][1],
        )
    )
    return cute.make_tensor(acc.iterator, fragment_a_layout)

@cute.kernel
def _flash_attention_single_tile_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gO: cute.Tensor,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
) -> None:
    # gQ -> (32, 16)
    # gK -> (16, 16)
    # gV -> (16, 16)
    """Fill the seven steps below after implementing the layout helper."""
    # ---- 1. Q @ K.T -> FP32 score accumulator ----
    # Get this lane's QK ThrMma slice; partition/load Q and K; allocate a
    # (M_QUERIES, N_TILE) C fragment; clear it; run cute.gemm.

    tidx, _, _ = cute.arch.thread_idx()

    thr_mma_qk = tiled_mma_qk.get_slice(tidx)

    tQgQ = thr_mma_qk.partition_A(gQ)
    tCrQ = thr_mma_qk.make_fragment_A(tQgQ)
    cute.autovec_copy(tQgQ, tCrQ)

    tKgK = thr_mma_qk.partition_B(gK)
    tCrK = thr_mma_qk.make_fragment_B(tKgK)
    cute.autovec_copy(tKgK, tCrK)
    
    tCrS = thr_mma_qk.make_fragment_C(thr_mma_qk.partition_shape_C((M_QUERIES, N_TILE)))
    tCrS.fill(0.0)
    cute.gemm(tiled_mma_qk, tCrS, tCrQ, tCrK, tCrS)
    rc_layout, threads_per_row = _sm80_m16n8_acc_rowcol_policy(tCrS.layout)

    tCrS_rc = cute.make_tensor(tCrS.iterator, rc_layout)

    num_rows, num_cols = cute.size(tCrS_rc, mode=[0]), cute.size(tCrS_rc, mode=[1])

    ms = cute.make_rmem_tensor((num_rows, ), cutlass.Float32)
    ls = cute.make_rmem_tensor((num_rows, ), cutlass.Float32)
    alphas = cute.make_rmem_tensor((num_rows, ), cutlass.Float32)
    ms.fill(NEG_INF)
    ls.fill(0.0)

    _online_softmax_update(tCrS_rc, ms, ls, alphas, threads_per_row)

    for r in cutlass.range_constexpr(num_rows):
        for c in cutlass.range_constexpr(num_cols):
            tCrS_rc[r, c] = cute.math.exp(tCrS_rc[r, c] - ms[r])

    # rP.layout -> ((V0, V1), MMA_M, QK_MMA_N)
    rP = cute.make_fragment_like(tCrS, cutlass.BFloat16)
    rP.store(tCrS.load().to(cutlass.BFloat16))

    tOrP = _sm80_accumulator_to_fragment_a(rP)

    thr_mma_pv = tiled_mma_pv.get_slice(tidx)
    gV_t = cute.make_tensor(gV.iterator, cute.make_layout((HEAD_DIM, N_TILE), stride=(1, HEAD_DIM)))
    tVgV = thr_mma_pv.partition_B(gV_t)
    tCrV = thr_mma_pv.make_fragment_B(tVgV)
    cute.autovec_copy(tVgV, tCrV)

    tCrO = thr_mma_pv.make_fragment_C(thr_mma_pv.partition_shape_C((M_QUERIES, HEAD_DIM)))
    tOgO = thr_mma_pv.partition_C(gO)
    tCrO.fill(0.0)
    cute.gemm(tiled_mma_pv, tCrO, tOrP, tCrV, tCrO)

    tCrO_rc = cute.make_tensor(tCrO.iterator, rc_layout)

    for r in cutlass.range_constexpr(num_rows):
        for c in cutlass.range_constexpr(num_cols):
            tCrO_rc[r, c] = tCrO_rc[r, c] / ls[r]

    cute.autovec_copy(tCrO, tOgO)

    # ---- 2. Re-view scores as (local_row, local_column) ----
    # Use _sm80_m16n8_acc_rowcol_policy(tCrS.layout), then make a zero-copy
    # tensor view. Allocate layout-shaped ms/ls and initialize to NEG_INF/0.

    # ---- 3. One-tile softmax state update ----
    # Call _online_softmax_update(scores_rc, ms, ls, threads_per_row). With one
    # tile this computes each row's max and E.sum, where E = exp(S-row_max).

    # ---- 4. Overwrite scores with unnormalized E and convert FP32 -> BF16 ----
    # For each local row/column write exp(score-ms[row]) into scores_rc.
    # Allocate rP = cute.make_fragment_like(tCrS, cutlass.BFloat16), then use
    # rP.store(tCrS.load().to(cutlass.BFloat16)). Do not divide by l yet:
    # normalization is linear and can happen once on O after P @ V.

    # ---- 5. Convert the QK C fragment layout into the P@V A fragment layout ----
    # tOrP = _sm80_accumulator_to_fragment_a(rP). This is the central exercise:
    # same BF16 P registers, new indexing accepted by the second MMA.

    # ---- 6. Load V as P@V operand B and run the second MMA ----
    # Host V is (N_TILE, HEAD_DIM) row-major. Build a zero-copy logical tensor
    # (HEAD_DIM, N_TILE) with stride (1, HEAD_DIM), partition_B with the P@V
    # ThrMma, copy into its B fragment, allocate/clear O's C fragment, and GEMM.

    # ---- 7. Normalize and write O ----
    # Re-view O with _sm80_m16n8_acc_rowcol_policy. For each local row multiply
    # all local output columns by 1/ls[row], then copy the C fragment to gO.
    # The output fragment shares the same logical M ownership as scores, so the
    # local row index aligns with ms/ls; no global-row lookup is needed here.

@cute.jit
def _flash_attention_single_tile_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
) -> None:
    """Construct independent QK and P@V m16n8k16 MMA objects and launch one warp."""
    tiled_mma_qk = cute.make_tiled_mma(
        cute.make_mma_atom(
            warp.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (16, 8, 16),
            )
        )
    )
    tiled_mma_pv = cute.make_tiled_mma(
        cute.make_mma_atom(
            warp.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (16, 8, 16),
            )
        )
    )
    _flash_attention_single_tile_kernel(
        mQ,
        mK,
        mV,
        mO,
        tiled_mma_qk,
        tiled_mma_pv,
    ).launch(
        grid=(1, 1, 1),
        block=(cute.arch.WARP_SIZE, 1, 1),
    )


def cute_flash_attention_single_tile(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Compute one unmasked 16-token FlashAttention tile.

    This public wrapper deliberately rejects every tail/loop case. Complete the
    kernel and remove the gate below; the correctness tests will change from
    XFAIL to XPASS, while a numerically wrong implementation remains a failure.
    """
    if not Q.is_cuda or not K.is_cuda or not V.is_cuda:
        raise ValueError("Q, K, and V must be CUDA tensors")
    if Q.device != K.device or Q.device != V.device:
        raise ValueError("Q, K, and V must be on the same CUDA device")
    if (
        Q.dtype is not torch.bfloat16
        or K.dtype is not torch.bfloat16
        or V.dtype is not torch.bfloat16
    ):
        raise ValueError("cute_flash_attention_single_tile requires BF16 Q, K, and V")
    if Q.shape != (M_QUERIES, HEAD_DIM):
        raise ValueError(
            f"Q must have shape ({M_QUERIES}, {HEAD_DIM}), got {tuple(Q.shape)}"
        )
    if K.shape != (N_TILE, HEAD_DIM):
        raise ValueError(
            f"K must have shape ({N_TILE}, {HEAD_DIM}), got {tuple(K.shape)}"
        )
    if V.shape != (N_TILE, HEAD_DIM):
        raise ValueError(
            f"V must have shape ({N_TILE}, {HEAD_DIM}), got {tuple(V.shape)}"
        )

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    O = torch.empty(
        (M_QUERIES, HEAD_DIM),
        dtype=torch.float32,
        device=Q.device,
    )
    _flash_attention_single_tile_launch(
        from_dlpack(Q),
        from_dlpack(K),
        from_dlpack(V),
        from_dlpack(O),
    )
    return O
