"""第 3.2 级：four-warp Split-Q CTA 的连续 CuTe DSL FlashAttention。

阶梯位置
--------
    第 2 级    online_softmax.py                 手算 SM80 online softmax       [已完成]
    第 2.5 级  online_softmax_layout.py          layout/policy + 通用 softmax  [已完成]
    第 3.0 级  flash_attention_single_tile.py    单 tile S -> P -> P @ V       [已完成]
    第 3.1 级  flash_attention.py                one-warp causal KV loop       [已完成]
    第 3.2 级  flash_attention.py                four-warp Split-Q             [本文件]

本级保持一个 CTA 独占一个 ``(q_tile, q_head)``，但把 CTA 的 Q rows 均匀分给
``num_warps``：

    1 CTA = 4 warps = 1 complete 64-row Q tile = one (q_tile, q_head)
    1 warp = 16 consecutive Q rows

``blockIdx.x`` 枚举 Q tile，``blockIdx.y`` 枚举 query head。每个 warp 独占自己行的
``(m, l, O)`` online state，CTA 内所有 warp 协作加载 Q/K/V，并串行扫描相同的可见
K/V tiles。Q rows 不重叠，因此无需跨 warp softmax merge 或原子操作。

契约
----
    Q       : (q_len,  Hq,  D) BF16
    K, V    : (kv_len, Hkv, D) BF16
    D >= 64，且满足 SM80 MMA 和 128-bit copy 的整除约束
    start_pos + q_len == kv_len
    Hq % Hkv == 0
    scale > 0 且 finite
    -> O    : (q_len,  Hq,  D) BF16

使用 lower-right causal mask：query ``q`` 的绝对位置是 ``start_pos + q``，只能读取
``key <= start_pos + q``。GQA 映射为：

    group_size = Hq // Hkv
    kv_head = q_head // group_size

第一版对齐约束
--------------
为了先隔离 CTA/grid、GQA、causal 和 online-rescale，本级暂时要求：

    q_len  % block_m == 0
    kv_len % block_n == 0
    head_dim >= 64
    head_dim % 8 == 0
    (block_m * head_dim) % (32 * num_warps * 8) == 0
    (block_n * head_dim) % (32 * num_warps * 8) == 0

最后三条是 128-bit BF16 global-to-shared copy policy 的覆盖约束：每条 copy 搬运
8 个 BF16，并且当前 specialization 让 CTA 全部线程均匀参与每个 Q、K 和 V tile。
它们属于当前 copy policy，而不是 FlashAttention 数学本身的限制。

``block_m``、``block_n``、``head_dim`` 和 ``num_warps`` 作为
``cutlass.Constexpr`` 从 launch 传入 kernel，因此每组配置生成独立 specialization。
Public wrapper 的默认配置为 ``(64, 16, 64, 4)``。

所以 Q load 和 O store 没有尾 tile，K/V global load 也没有越界。本级仍需对最后一个
可见 K/V tile 逐 score 做 causal mask；alignment 并不会消除 causal 边界。下一小级再
用 identity-coordinate predicates 支持任意 q_len/kv_len。

低精度 online recurrence
------------------------
进入当前 K/V tile 前：

    l_old = sum(exp(score_old - m_old))
    O_old = sum(BF16(exp(score_old - m_old)) @ V_old)

当前 tile：

    S_tile = (Q @ K_tile.T) * scale
    m_new  = max(m_old, rowmax(mask_causal(S_tile)))
    alpha  = exp(m_old - m_new)
    P_tile = exp(mask_causal(S_tile) - m_new)

更新：

    l_new = l_old * alpha + rowsum(P_tile)
    O_new = O_old * alpha + BF16(P_tile) @ V_tile

所有可见 K/V tiles 完成后才执行 ``O /= l``，再把 FP32 accumulator 转成 BF16 写回。
不要在 P 转 BF16 前除以 l，否则会改变低精度契约，也破坏 numerator 的线性更新形式。

注意：本 DSL 文件不能加入 ``from __future__ import annotations``；否则
``cutlass.Constexpr``/runtime scalar 注解会失去 DSL 编译语义。
"""

import math

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp as warp
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import CopyUniversalOp, cpasync

from einf.executors.torch.dsl.flash_attention_single_tile import (
    _sm80_accumulator_to_fragment_a,
)
from einf.executors.torch.dsl.online_softmax_layout import (
    _online_softmax_update,
    _sm80_m16n8_acc_rowcol_policy,
)

__all__ = ["cute_flash_attention"]

DEFAULT_BLOCK_M = 64
DEFAULT_BLOCK_N = 16
DEFAULT_HEAD_DIM = 64
DEFAULT_NUM_WARPS = 4
DEFAULT_STAGES = 2
MIN_VECTORIZED_HEAD_DIM = 64
GMEM_COPY_BITS = 128
GMEM_COPY_BYTES = GMEM_COPY_BITS // 8
BF16_BITS = 16
BF16_ELEMENTS_PER_GMEM_COPY = GMEM_COPY_BITS // BF16_BITS
SM80_MMA_M = 16
SM80_MMA_N = 8
SM80_MMA_K = 16
NEG_INF = -1.0e30


@cute.jit
def cal_num_visible_tiles(
    q_tile,
    kv_len,
    start_pos,
    block_m: cutlass.Constexpr[int],
    block_n: cutlass.Constexpr[int],
):
    q_first = q_tile * block_m
    q_last = q_first + block_m - 1
    last_abs_q = start_pos + q_last
    kv_end = cutlass.min(kv_len, last_abs_q + 1)
    return cute.ceil_div(kv_end, block_n)

@cute.jit
def copy_thr_layout(num_warps: cutlass.Constexpr[int]):
    if cutlass.const_expr(num_warps == 1):
        return cute.make_layout((4, 8), stride=(8, 1))
    elif cutlass.const_expr(num_warps == 2):
        return cute.make_layout((8, 8), stride=(8, 1))
    elif cutlass.const_expr(num_warps == 4):
        return cute.make_layout((16, 8), stride=(8, 1))
    else:
        return cute.make_layout((16, 16), stride=(16, 1))

@cute.jit
def tiled_copy(num_warps: cutlass.Constexpr[int]):
    thr_layout = copy_thr_layout(num_warps)
    value_layout = cute.make_layout((1, 8), stride=(8, 1))
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(
            cache_mode=cpasync.LoadCacheMode.GLOBAL,
        ),
        cutlass.BFloat16,
        num_bits_per_copy=GMEM_COPY_BITS,
    )
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, value_layout)

@cute.jit
def tiled_copy_s2g(num_warps: cutlass.Constexpr[int]):
    thr_layout = copy_thr_layout(num_warps)
    value_layout = cute.make_layout((1, 8), stride=(8, 1))
    copy_atom = cute.make_copy_atom(
        CopyUniversalOp(),
        cutlass.BFloat16,
        num_bits_per_copy=GMEM_COPY_BITS,
    )
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, value_layout)

@cute.jit
def smem_layouts(
    block_m: cutlass.Constexpr[int], 
    block_n: cutlass.Constexpr[int], 
    head_dim: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
):
    swizzle_op = cute.make_swizzle(3, 3, 3)
    
    atom_layout = cute.make_layout((8, 64), stride=(64, 1))
    atom_layout_T = cute.make_layout((64, 8), stride=(1, 64))

    smem_atom = cute.make_composed_layout(swizzle_op, 0, atom_layout)
    smem_atom_T = cute.make_composed_layout(swizzle_op, 0, atom_layout_T)
    sQ_layout = cute.tile_to_shape(smem_atom, (block_m, head_dim), (1, 0))
    sK_layout = cute.tile_to_shape(smem_atom, (block_n, head_dim, stages), (1, 0, 2))
    sV_layout = cute.tile_to_shape(smem_atom, (block_n, head_dim, stages), (1, 0, 2))
    sV_layout_T = cute.tile_to_shape(smem_atom_T, (head_dim, block_n, stages), (0, 1, 2))
    return sQ_layout, sK_layout, sV_layout, sV_layout_T

@cute.jit
def make_s2r_tiled_copy_A(tiled_mma, num_matrices: cutlass.Constexpr[int] = 4, transpose: cutlass.Constexpr[bool] = False):
    ldmatrix = cute.nvgpu.warp.LdMatrix8x8x16bOp(
        transpose=transpose,
        num_matrices=num_matrices
    )
    copy_atom = cute.make_copy_atom(ldmatrix, cutlass.BFloat16)
    return cute.make_tiled_copy_A(copy_atom, tiled_mma)

@cute.jit
def make_s2r_tiled_copy_B(tiled_mma, num_matrices: cutlass.Constexpr[int] = 4, transpose: cutlass.Constexpr[bool] = False):
    ldmatrix = cute.nvgpu.warp.LdMatrix8x8x16bOp(
        transpose=transpose,
        num_matrices=num_matrices
    )
    copy_atom = cute.make_copy_atom(ldmatrix, cutlass.BFloat16)
    return cute.make_tiled_copy_B(copy_atom, tiled_mma)


@cute.kernel
def _flash_attention_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gO: cute.Tensor,
    tiled_mma_qk: cute.TiledMma,
    tiled_mma_pv: cute.TiledMma,
    start_pos: cutlass.Int32,
    scale: cutlass.Float32,
    block_m: cutlass.Constexpr[int],
    block_n: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    num_warps: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
) -> None:
    """TODO: implement the ten checkpoints below, one mechanism at a time."""
    tidx, _,  _ = cute.arch.thread_idx()

    warp_id = tidx // 32
    m_per_warp = block_m // num_warps
    lane = tidx % 32

    num_q_heads = gQ.shape[1]
    kv_len, num_kv_heads = gK.shape[0], gK.shape[1]

    q_tile, q_head, _ = cute.arch.block_idx()

    kv_head = q_head // (num_q_heads // num_kv_heads)

    gQ_head = gQ[None, q_head, None]
    gK_head = gK[None, kv_head, None]
    gV_head = gV[None, kv_head, None]
    gO_head = gO[None, q_head, None]

    tiled_copy_qkv = tiled_copy(num_warps)
    thr_copy_qkv = tiled_copy_qkv.get_slice(tidx)

    s2r_tiled_copy_Q = make_s2r_tiled_copy_A(tiled_mma_qk)
    s2r_tiled_copy_K = make_s2r_tiled_copy_B(tiled_mma_qk)
    s2r_tiled_copy_V = make_s2r_tiled_copy_B(tiled_mma_pv, transpose=True)

    s2r_thr_copy_Q = s2r_tiled_copy_Q.get_slice(lane)
    s2r_thr_copy_K = s2r_tiled_copy_K.get_slice(lane)
    s2r_thr_copy_V = s2r_tiled_copy_V.get_slice(lane)

    thr_mma_qk = tiled_mma_qk.get_slice(lane)
    thr_mma_pv = tiled_mma_pv.get_slice(lane)

    smem = cutlass.utils.SmemAllocator()

    sQ_layout, sK_layout, sV_layout, sV_layout_T = smem_layouts(block_m, block_n, head_dim, stages)
 
    sQ_tensor = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=sQ_layout,
        byte_alignment=16,
    )
    sK_tensor = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=sK_layout,
        byte_alignment=16,
    )
    sV_tensor = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=sV_layout,
        byte_alignment=16,
    )

    sQ_tile = cute.local_tile(sQ_tensor, (m_per_warp, head_dim), (warp_id, 0))
    tQsQ = thr_copy_qkv.partition_D(sQ_tensor)
    tCsQ = thr_mma_qk.partition_A(sQ_tile)
    tCsQ_copy = s2r_thr_copy_Q.partition_S(sQ_tile)
    tCrQ = thr_mma_qk.make_fragment_A(tCsQ)
    tCrQ_copy = s2r_thr_copy_Q.retile(tCrQ)

    tCsK = thr_mma_qk.partition_B(sK_tensor)
    tCsK_copy = s2r_thr_copy_K.partition_S(sK_tensor)
    tCrK = thr_mma_qk.make_fragment_B(tCsK[None, None, None, 0])
    tCrK_copy = s2r_thr_copy_K.retile(tCrK)

    tCrP = thr_mma_qk.make_fragment_C(
        thr_mma_qk.partition_shape_C((m_per_warp, block_n))
    )
    tCcP = thr_mma_qk.partition_C(cute.make_identity_tensor((m_per_warp, block_n)))
    tCrO = thr_mma_pv.make_fragment_C(
        thr_mma_pv.partition_shape_C((m_per_warp, head_dim))
    )
    tCrO.fill(0.0)

    sV_tensor_T = cute.make_tensor(
        sV_tensor.iterator,
        sV_layout_T,
    )
    tCsV = thr_mma_pv.partition_B(sV_tensor_T)
    tCsV_copy = s2r_thr_copy_V.partition_S(sV_tensor_T)
    tCrV = thr_mma_pv.make_fragment_B(tCsV[None, None, None, 0])
    tCrV_copy = s2r_thr_copy_V.retile(tCrV)

    gQ_tile = cute.local_tile(gQ_head, (block_m, head_dim), (q_tile, 0))
    tQgQ = thr_copy_qkv.partition_S(gQ_tile)

    cute.copy(tiled_copy_qkv, tQgQ, tQsQ)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    cute.copy(s2r_tiled_copy_Q, tCsQ_copy, tCrQ_copy)

    tCrP_rs_layout, threads_per_row = _sm80_m16n8_acc_rowcol_policy(tCrP.layout)
    tCrO_rs_layout, _ = _sm80_m16n8_acc_rowcol_policy(tCrO.layout)
    tCrP_rs = cute.make_tensor(tCrP.iterator, tCrP_rs_layout)
    tCrO_rs = cute.make_tensor(tCrO.iterator, tCrO_rs_layout)

    num_rows_tCrP = cute.size(tCrP_rs_layout.shape[0])
    num_cols_tCrP = cute.size(tCrP_rs_layout.shape[1])
    num_rows_tCrO = cute.size(tCrO_rs_layout.shape[0])
    num_cols_tCrO = cute.size(tCrO_rs_layout.shape[1])

    ms = cute.make_rmem_tensor((num_rows_tCrP, ), cutlass.Float32)
    ls = cute.make_rmem_tensor((num_rows_tCrP, ), cutlass.Float32)
    alphas = cute.make_rmem_tensor((num_rows_tCrP, ), cutlass.Float32)
    ms.fill(NEG_INF)
    ls.fill(0.0)

    num_visible_tiles = cal_num_visible_tiles(
        q_tile,
        kv_len,
        start_pos,
        block_m,
        block_n,
    )

    q_start_row = q_tile * block_m + m_per_warp * warp_id

    write_stage, read_stage, kv_next = 0, 0, 0

    if cutlass.const_expr(stages == 1):
        prefetch_slots = 1
        steady_wait_groups = 0
    else:
        prefetch_slots = stages - 1
        steady_wait_groups = stages - 2

    for kv in range(min(num_visible_tiles, prefetch_slots)):
        gK_tile = cute.local_tile(
            gK_head,
            (block_n, head_dim),
            (kv, 0),
        )
        gV_tile = cute.local_tile(
            gV_head,
            (block_n, head_dim),
            (kv, 0),
        )

        tKgK = thr_copy_qkv.partition_S(gK_tile)
        tVgV = thr_copy_qkv.partition_S(gV_tile)

        tKsK = thr_copy_qkv.partition_D(sK_tensor[None, None, kv])
        tVsV = thr_copy_qkv.partition_D(sV_tensor[None, None, kv])
        cute.copy(tiled_copy_qkv, tKgK, tKsK)
        cute.copy(tiled_copy_qkv, tVgV, tVsV)
        cute.arch.cp_async_commit_group()

        kv_next += 1
        write_stage = (write_stage + 1) % stages
        

    for kv in range(num_visible_tiles):
        k_start_row = kv * block_n

        has_next_to_issue = kv_next < num_visible_tiles

        if has_next_to_issue:
            cute.arch.cp_async_wait_group(steady_wait_groups)
        else:
            cute.arch.cp_async_wait_group(0)

        cute.arch.sync_threads()
        cute.copy(s2r_tiled_copy_K, tCsK_copy[None, None, None, read_stage], tCrK_copy)
        cute.copy(s2r_tiled_copy_V, tCsV_copy[None, None, None, read_stage], tCrV_copy)
        cute.arch.sync_threads()

        if has_next_to_issue: 
            gK_tile = cute.local_tile(
                gK_head,
                (block_n, head_dim),
                (kv_next, 0),
            )
            gV_tile = cute.local_tile(
                gV_head,
                (block_n, head_dim),
                (kv_next, 0),
            )

            tKgK = thr_copy_qkv.partition_S(gK_tile)
            tVgV = thr_copy_qkv.partition_S(gV_tile)

            tKsK = thr_copy_qkv.partition_D(sK_tensor[None, None, write_stage])
            tVsV = thr_copy_qkv.partition_D(sV_tensor[None, None, write_stage])
            kv_next += 1
            write_stage = (write_stage + 1) % stages

            cute.copy(tiled_copy_qkv, tKgK, tKsK)
            cute.copy(tiled_copy_qkv, tVgV, tVsV)
            cute.arch.cp_async_commit_group()

        tCrP.fill(0.0)

        cute.gemm(tiled_mma_qk, tCrP, tCrQ, tCrK, tCrP)

        for r in cutlass.range_constexpr(num_rows_tCrP):
            for c in cutlass.range_constexpr(num_cols_tCrP):
                frag_idx = tCrP_rs_layout((r, c))
                q_offset, k_offset = tCcP[frag_idx]
                q_global = q_start_row + q_offset
                k_global = k_start_row + k_offset

                if k_global > start_pos + q_global:
                    tCrP_rs[r, c] = NEG_INF
                else:
                    tCrP_rs[r, c] *= scale

        _online_softmax_update(tCrP_rs, ms, ls, alphas, threads_per_row)

        for r in cutlass.range_constexpr(num_rows_tCrP):
            for c in cutlass.range_constexpr(num_cols_tCrP):
                tCrP_rs[r, c] = cute.math.exp(tCrP_rs[r, c] - ms[r])
        for r in cutlass.range_constexpr(num_rows_tCrO):
            for c in cutlass.range_constexpr(num_cols_tCrO):
                tCrO_rs[r, c] *= alphas[r]

        # recast aliases tCrP; store() does the FP32->BF16 cvt (not a bitcast).
        rP = cute.make_tensor(
            cute.recast_ptr(tCrP.iterator, dtype=cutlass.BFloat16),
            cute.make_layout(tCrP.shape, stride=tCrP.stride),
        )
        rP.store(tCrP.load().to(cutlass.BFloat16))
        rP = _sm80_accumulator_to_fragment_a(rP)

        cute.gemm(tiled_mma_pv, tCrO, rP, tCrV, tCrO)
        
        read_stage = (read_stage + 1) % stages

    for r in cutlass.range_constexpr(num_rows_tCrO):
        for c in cutlass.range_constexpr(num_cols_tCrO):
            tCrO_rs[r, c] /= ls[r]

    tCrO_BF16 = cute.make_fragment_like(tCrO, cutlass.BFloat16)
    tCrO_BF16.store(tCrO.load().to(cutlass.BFloat16))
    # Reuse sQ as O scratch: write MMA-C ownership, then reload with the
    # same 128-bit TV mapping used for Q/K/V so gmem stores are coalesced.
    tCsO = thr_mma_pv.partition_C(sQ_tile)
    cute.autovec_copy(tCrO_BF16, tCsO)
    cute.arch.sync_threads()

    tiled_copy_o = tiled_copy_s2g(num_warps)
    thr_copy_o = tiled_copy_o.get_slice(tidx)
    gO_tile = cute.local_tile(gO_head, (block_m, head_dim), (q_tile, 0))
    tOsO = thr_copy_o.partition_S(sQ_tensor)
    tOgO = thr_copy_o.partition_D(gO_tile)
    cute.copy(tiled_copy_o, tOsO, tOgO)

    
 
    # ---- 1. Decode CTA ownership and select this query/GQA head ----
    # Unpack lane from tidx. Read (q_tile_idx, q_head_idx, _) from
    # cute.arch.block_idx(). Derive:
    #
    #   group_size = Hq // Hkv
    #   kv_head_idx = q_head_idx // group_size
    #
    # Slice global tensors without copying:
    #
    #   gQ_head = gQ[None, q_head_idx, None]       # (q_len,  D)
    #   gK_head = gK[None, kv_head_idx, None]      # (kv_len, D)
    #   gV_head = gV[None, kv_head_idx, None]
    #   gO_head = gO[None, q_head_idx, None]
    #
    # Then local_tile Q/O at (q_tile_idx, 0) with shape (BLOCK_M, HEAD_DIM).
    # The wrapper guarantees there is no Q tail in this rung.

    # ---- 2. Partition and load this CTA's Q tile once ----
    # Get the lane's QK ThrMma slice, partition_A(gQ_tile), allocate its A
    # fragment, and copy Q into registers. Q remains live for the complete KV loop.
    # HEAD_DIM=64 means four m16n8k16 MMA-K atoms; the MMA atom itself remains K=16.

    # ---- 3. Allocate persistent score/output fragments and row/column views ----
    # Per-warp score C fragment:  (M_PER_WARP, BLOCK_N) = (16,16), FP32.
    # Per-warp output C fragment: (M_PER_WARP, HEAD_DIM) = (16,64), FP32.
    #
    # Derive the two row/column layouts independently with
    # _sm80_m16n8_acc_rowcol_policy. Their row ownership must agree, but their local
    # column counts differ: score has 4 local columns/lane; output has 16.
    # Partition gO_tile with the P@V ThrMma. Clear tCrO OUTSIDE the KV loop because
    # it is the persistent unnormalized numerator owned only by this CTA.

    # ---- 4. Allocate persistent online state and reusable P view ----
    # Allocate layout-sized FP32 ms, ls, alphas: one value per local score row.
    # Initialize ms=NEG_INF, ls=0, alphas=0, tCrO=0. Allocate a reusable BF16 P
    # fragment from tCrS, then create its zero-copy P@V fragment-A view with
    # _sm80_accumulator_to_fragment_a. Each warp's 16x16 score tile contains two
    # QK N=8 repeats, which pair into the P@V K=16 atom expected by the adapter.

    # ---- 5. Compute this CTA's visible KV-loop bound ----
    # The last query in this aligned Q tile has exclusive visible-key bound:
    #
    #   kv_end = min(kv_len, start_pos + (q_tile_idx + 1) * BLOCK_M)
    #   num_visible_tiles = ceil_div(kv_end, BLOCK_N)
    #
    # This is runtime CTA-dependent. Use a normal DSL ``range`` loop, not
    # range_constexpr; its index may be used by cute.local_tile because it is not a
    # Python-list/fragment static index. A CTA never scans keys wholly to its future.

    # ---- 6. Load K_tile, compute scaled scores, and apply lower-right causal mask ----
    # local_tile gK_head/gV_head at (kv_tile_idx, 0). Partition/load K with the QK
    # ThrMma, clear tCrS, and run Q @ K_tile.T.
    #
    # Build logical score coordinates from an identity tensor over (q_len, kv_len),
    # local_tile it at (q_tile_idx, kv_tile_idx), and partition_C with the same QK
    # ThrMma. Re-view those coordinates with the score row/column layout. For every
    # owned score (q_global, k_global):
    #
    #   score *= scale
    #   if k_global > start_pos + q_global: score = NEG_INF
    #
    # Alignment makes every global load valid; it does NOT remove this causal mask.

    # ---- 7. Save m_old, then update m/l and derive alpha ----
    # Save ms[r] to alphas[r], call _online_softmax_update on the masked/scaled score
    # view, then set alphas[r] = exp(saved_m_old - ms[r]). The helper updates:
    #
    #   ms = m_new
    #   ls = l_old * alpha + rowsum(exp(masked_score - m_new))

    # ---- 8. Form P_tile and rescale the old output numerator ----
    # Overwrite each valid/masked score with exp(score-ms[r]); NEG_INF causal entries
    # underflow to zero. Multiply every local output element by alpha before the
    # current P@V. Use the output view's own local-column count, not score's count.

    # ---- 9. Quantize P, load V_tile, and accumulate P @ V ----
    # Convert tCrS FP32 -> reusable BF16 P. V_tile host view is (BLOCK_N, HEAD_DIM),
    # while MMA B is logically (output_D, reduction_N). Build a zero-copy view:
    #
    #   cute.make_layout((HEAD_DIM, BLOCK_N), stride=(1, HEAD_DIM))
    #
    # over V_tile.iterator, partition/load it with the P@V ThrMma, and call cute.gemm
    # with tCrO as both D and C:
    #
    #   tCrO = tCrO + BF16(P_tile) @ BF16(V_tile)

    # ---- 10. Normalize, convert to BF16, and write this CTA's O tile ----
    # After the KV loop, divide every local tCrO row by ls[r]. Convert the normalized
    # FP32 C fragment into a BF16 fragment before copying to the BF16 gO partition.
    # No output predicate is needed in this aligned-only rung, and no other CTA owns
    # these output rows/head.


@cute.jit
def _flash_attention_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    start_pos: cutlass.Int32,
    scale: cutlass.Float32,
    block_m: cutlass.Constexpr[int],
    block_n: cutlass.Constexpr[int],
    head_dim: cutlass.Constexpr[int],
    num_warps: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
) -> None:
    """Specialize a CTA tile split evenly across ``num_warps`` Q-row warps."""
    assert block_m > 0 and block_m % SM80_MMA_M == 0, (
        "block_m must be a positive multiple of 16"
    )
    assert num_warps in (1, 2, 4, 8), (
        "num_warps must be one of 1, 2, 4, or 8"
    )
    assert block_m % num_warps == 0 and (
        block_m // num_warps
    ) % SM80_MMA_M == 0, (
        "block_m / num_warps must be a multiple of 16"
    )
    assert block_n > 0 and block_n % SM80_MMA_K == 0, (
        "block_n must be a positive multiple of 16"
    )
    assert head_dim >= MIN_VECTORIZED_HEAD_DIM, (
        "head_dim must be at least 64 for the vectorized FlashAttention kernel"
    )
    assert head_dim % SM80_MMA_K == 0, (
        "head_dim must be a multiple of 16 for SM80 m16n8k16 MMA"
    )
    assert head_dim % BF16_ELEMENTS_PER_GMEM_COPY == 0, (
        "head_dim must be divisible by 8 for 128-bit BF16 copies"
    )
    assert (
        block_m * head_dim
    ) % (cute.arch.WARP_SIZE * num_warps * BF16_ELEMENTS_PER_GMEM_COPY) == 0, (
        "Q tile must divide evenly across CTA threads in 128-bit BF16 copies"
    )
    assert (
        block_n * head_dim
    ) % (cute.arch.WARP_SIZE * num_warps * BF16_ELEMENTS_PER_GMEM_COPY) == 0, (
        "K/V tile must divide evenly across CTA threads in 128-bit BF16 copies"
    )
    assert block_n % SM80_MMA_N == 0
    assert (block_n // SM80_MMA_N) % 2 == 0, (
        "QK MMA_N repeats must pair into P@V K=16 atoms"
    )
    assert stages >= 1, ("Stages must be greater than 0")

    # Both GEMMs use the hardware m16n8k16 atom. Logical QK K=64 and P@V output
    # N=64 are expressed by the partitioned fragment's repeat modes, not by changing
    # the atom's shape_mnk.
    tiled_mma_qk = cute.make_tiled_mma(
        cute.make_mma_atom(
            warp.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (SM80_MMA_M, SM80_MMA_N, SM80_MMA_K),
            )
        )
    )
    tiled_mma_pv = cute.make_tiled_mma(
        cute.make_mma_atom(
            warp.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (SM80_MMA_M, SM80_MMA_N, SM80_MMA_K),
            )
        )
    )
    _flash_attention_kernel(
        mQ,
        mK,
        mV,
        mO,
        tiled_mma_qk,
        tiled_mma_pv,
        start_pos,
        scale,
        block_m,
        block_n,
        head_dim,
        num_warps,
        stages,
    ).launch(
        grid=(
            cute.ceil_div(cute.size(mQ.shape[0]), block_m),
            cute.size(mQ.shape[1]),
            1,
        ),
        block=(cute.arch.WARP_SIZE * num_warps, 1, 1),
    )


def cute_flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    start_pos: int,
    scale: float,
) -> torch.Tensor:
    """Run aligned BF16 lower-right-causal attention with four-way Split-Q."""
    if not Q.is_cuda or not K.is_cuda or not V.is_cuda:
        raise ValueError("Q, K, and V must be CUDA tensors")
    if Q.device != K.device or Q.device != V.device:
        raise ValueError("Q, K, and V must be on the same CUDA device")
    if (
        Q.dtype is not torch.bfloat16
        or K.dtype is not torch.bfloat16
        or V.dtype is not torch.bfloat16
    ):
        raise ValueError("cute_flash_attention requires BF16 Q, K, and V")
    if Q.dim() != 3:
        raise ValueError(
            "Q must have shape (q_len, num_attention_heads, head_dim), "
            f"got {tuple(Q.shape)}"
        )
    if K.dim() != 3:
        raise ValueError(
            "K must have shape (kv_len, num_kv_heads, head_dim), "
            f"got {tuple(K.shape)}"
        )
    if V.shape != K.shape:
        raise ValueError(
            f"V must match K shape, got K={tuple(K.shape)} and V={tuple(V.shape)}"
        )

    q_len, num_attention_heads, q_head_dim = Q.shape
    kv_len, num_kv_heads, kv_head_dim = K.shape
    if q_len <= 0:
        raise ValueError("q_len must be positive")
    if kv_len <= 0:
        raise ValueError("kv_len must be positive")
    if num_attention_heads <= 0:
        raise ValueError("num_attention_heads must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if q_head_dim != kv_head_dim:
        raise ValueError(
            "Q and K/V head_dim must match, got "
            f"{q_head_dim} and {kv_head_dim}"
        )
    if q_head_dim < MIN_VECTORIZED_HEAD_DIM:
        raise ValueError(
            "Q/K/V head_dim must be at least "
            f"{MIN_VECTORIZED_HEAD_DIM}, got {q_head_dim}"
        )
    if q_head_dim % SM80_MMA_K != 0:
        raise ValueError(
            "Q/K/V head_dim must be a multiple of 16 for SM80 m16n8k16 MMA"
        )
    if q_head_dim % BF16_ELEMENTS_PER_GMEM_COPY != 0:
        raise ValueError(
            "Q/K/V head_dim must be divisible by 8 for 128-bit BF16 copies"
        )
    copy_elements_per_cta = (
        cute.arch.WARP_SIZE
        * DEFAULT_NUM_WARPS
        * BF16_ELEMENTS_PER_GMEM_COPY
    )
    if (DEFAULT_BLOCK_M * q_head_dim) % copy_elements_per_cta != 0:
        raise ValueError(
            "Q tile must divide evenly across CTA threads in 128-bit BF16 copies"
        )
    if (DEFAULT_BLOCK_N * q_head_dim) % copy_elements_per_cta != 0:
        raise ValueError(
            "K/V tile must divide evenly across CTA threads in 128-bit BF16 copies"
        )
    if num_attention_heads % num_kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_kv_heads")
    if q_len % DEFAULT_BLOCK_M != 0:
        raise ValueError(
            "first version requires q_len to be a multiple of "
            f"{DEFAULT_BLOCK_M}, got {q_len}"
        )
    if kv_len % DEFAULT_BLOCK_N != 0:
        raise ValueError(
            "first version requires kv_len to be a multiple of "
            f"{DEFAULT_BLOCK_N}, got {kv_len}"
        )
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if start_pos + q_len != kv_len:
        raise ValueError(
            "start_pos + q_len must equal kv_len, got "
            f"{start_pos} + {q_len} != {kv_len}"
        )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if not Q.is_contiguous() or not K.is_contiguous() or not V.is_contiguous():
        raise ValueError("Q, K, and V must be contiguous")
    for name, tensor in (("Q", Q), ("K", K), ("V", V)):
        if tensor.data_ptr() % GMEM_COPY_BYTES != 0:
            raise ValueError(
                f"{name} data pointer must be {GMEM_COPY_BYTES}-byte aligned "
                f"for {GMEM_COPY_BITS}-bit copies"
            )

    O = torch.empty_like(Q)
    _flash_attention_launch(
        from_dlpack(Q, assumed_align=GMEM_COPY_BYTES),
        from_dlpack(K, assumed_align=GMEM_COPY_BYTES),
        from_dlpack(V, assumed_align=GMEM_COPY_BYTES),
        from_dlpack(O, assumed_align=GMEM_COPY_BYTES),
        start_pos,
        scale,
        DEFAULT_BLOCK_M,
        DEFAULT_BLOCK_N,
        q_head_dim,
        DEFAULT_NUM_WARPS,
        DEFAULT_STAGES,
    )
    return O
