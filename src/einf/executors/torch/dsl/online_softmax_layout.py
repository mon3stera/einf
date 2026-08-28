"""第 2.5 级：layout-derived Online Softmax（CuTe DSL 学习脚手架）。

阶梯位置
--------
    第 2 级    online_softmax.py         手算 SM80 m16n8k16 accumulator ownership  [已完成]
    第 2.5 级  online_softmax_layout.py  从 CuTe Layout 推导相同算法              [本文件]
    第 3 级    cute_flash_attention      加入 P @ V 与 O 的在线 rescale           [待做]

这一版刻意保留相同的 online-softmax 代数，只改变布局表达方式。不要把第 2 级中的
``tCrS[0..3]``、两个标量状态、固定 xor 1/2 或 ``tidx // 4`` 搬过来。

契约
----
    Q : (32, 16)      bf16
    K : (kv_len, 16)  bf16，kv_len 是 16 的正整数倍
    ->
    m : (32,)         f32，S = Q @ K.T 的逐行最大值
    l : (32,)         f32，exp(S - m[:, None]) 的逐行和

MMA atom 仍是 SM80 m16n8k16，但逻辑 score tile 放大为 32x16。因此 MMA_M=MMA_N=2，
每线程 score fragment 有 16 个值。这个规模会让手算版的四处布局假设立即失效。

学习目标
--------
1. 从 ``tiled_mma.tv_layout_C`` 的 stride 在逻辑 (M, N) 域中的位移，识别每个
   thread/value leaf 是沿行还是沿列。
2. 从 ``tCrS.layout`` 构造 ``(thread_rows, thread_cols)`` view；只换索引，不搬寄存器。
3. 用 row shape 创建 m/l fragment，让状态数量由 layout 决定。
4. 从只沿 N 移动的 thread modes 推导 ``threads_per_row``，并验证其适合连续
   power-of-two butterfly reduction。
5. 用 ``partition_C(make_identity_tensor(...))`` 取得真实写回行号。

真正的泛化边界
--------------
本练习只支持可分离的 TV layout：每个 leaf 必须纯沿 M 或纯沿 N；共享一行的
thread modes 必须是 thread domain 的连续低位前缀；``threads_per_row`` 必须是
不超过 warp size 的 2 次幂。不满足时应在编译期明确失败，不能静默套用本算法。

注意：本 DSL 文件不能加入 ``from __future__ import annotations``；否则
``cutlass.Constexpr`` 会变成字符串注解，JIT 特化失效。
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp as warp
import torch
from cutlass.cute.runtime import from_dlpack

M_QUERIES = 32
N_TILE = 16
HEAD_DIM = 16
NEG_INF = -1.0e30

@cute.jit
def _derive_acc_rowcol_policy(tiled_mma, acc_layout):
    # (16, 8, 16)
    atom_m, atom_n, atom_k = tiled_mma.shape_mnk

    # ((4, 8), (2, 2)):((32, 1), (16, 8))
    # T shape = (4, 8), T stride = (32, 1)
    # V shape = (2, 2), V stride = (16, 8)
    tv = tiled_mma.tv_layout_C

    tv_t_shape = cute.flatten_to_tuple(tv.shape[0])
    tv_t_stride = cute.flatten_to_tuple(tv.stride[0])
    tv_v_shape = cute.flatten_to_tuple(tv.shape[1])
    tv_v_stride = cute.flatten_to_tuple(tv.stride[1])

    # ((0, 2), (1, 0))
    thread_delta = tuple(cute.idx2crd(s, (atom_m, atom_n)) for s in tv_t_stride)

    # ((0, 1), (8, 0))
    value_delta = tuple(cute.idx2crd(s, (atom_m, atom_n)) for s in tv_v_stride)

    all_delta = thread_delta + value_delta
    assert all((dm == 0) != (dn == 0) for dm, dn in all_delta)

    row_value_modes = tuple(i for i, (_, dn) in enumerate(value_delta) if dn == 0)
    col_value_modes = tuple(i for i, (dm, _) in enumerate(value_delta) if dm == 0)
    col_thread_modes = tuple(i for i, (dm, _) in enumerate(thread_delta) if dm == 0)

    acc_v_shape = cute.flatten_to_tuple(tCrS.layout.shape[0])
    acc_v_stride = cute.flatten_to_tuple(tCrS.layout.stride[0])

    row_shape = tuple(acc_v_shape[i] for i in row_value_modes) + (tCrS.layout.shape[1], )
    row_stride = tuple(acc_v_stride[i] for i in row_value_modes) + (tCrS.layout.stride[1], )
    col_shape = tuple(acc_v_shape[i] for i in col_value_modes) + (tCrS.layout.shape[2], )
    col_stride = tuple(acc_v_stride[i] for i in col_value_modes) + (tCrS.layout.stride[2], )

    rowcol_layout = cute.make_layout((row_shape, col_shape), stride=(row_stride, col_stride))
    threads_per_row = 1
    for i in col_thread_modes:
        threads_per_row *= tv_t_shape[i]
    return rowcol_layout, threads_per_row

@cute.jit
def _sm80_m16n8_acc_rowcol_policy(acc_layout):
    # acc_layout -> ((V0, V1), MMA_M, MMA_N)

    flat = cute.flatten(acc_layout)
    l0 = cute.select(flat, mode=[1, 2])
    l1 = cute.select(flat, mode=[0, 3])
    new_shape = (l0.shape, l1.shape)
    new_stride = (l0.stride, l1.stride)
    return cute.make_layout(new_shape, stride=new_stride), 4

@cute.jit
def _online_softmax_update(scores_rc, ms, ls, alphas, threads_per_row: cutlass.Constexpr):
    num_rows, num_cols = cute.size(scores_rc, mode=[0]), cute.size(scores_rc, mode=[1])

    for r in cutlass.range_constexpr(num_rows):
        mt, lt = NEG_INF, 0.0

        for c in cutlass.range_constexpr(num_cols):
            mt = cutlass.max(mt, scores_rc[r, c])
        
        mt = cute.arch.warp_reduction_max(mt, threads_in_group=threads_per_row)
        m_new = cutlass.max(mt, ms[r])
        
        for c in cutlass.range_constexpr(num_cols):
            lt += cute.math.exp(scores_rc[r, c] - m_new)

        lt = cute.arch.warp_reduction_sum(lt, threads_in_group=threads_per_row)
        alpha = cute.math.exp(ms[r] - m_new)
        ls[r] = ls[r] * alpha + lt
        ms[r] = m_new
        alphas[r] = alpha
        

@cute.kernel
def _online_softmax_layout_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gM: cute.Tensor,
    gL: cute.Tensor,
    tiled_mma: cute.TiledMma,
    num_tiles: cutlass.Constexpr,
) -> None:
    """Fill the seven steps below; the public wrapper gates this scaffold for now."""
    
    tidx, _, _ = cute.arch.thread_idx()

    thr_mma = tiled_mma.get_slice(tidx)
    
    tQgQ = thr_mma.partition_A(gQ)
    tCrQ = thr_mma.make_fragment_A(tQgQ)
    cute.autovec_copy(tQgQ, tCrQ)

    # layout -> ((2, 2), 2, 2) : ((1, 2), 4, 8)
    tCrS = thr_mma.make_fragment_C(thr_mma.partition_shape_C((M_QUERIES, N_TILE)))

    cS = cute.make_identity_tensor((M_QUERIES, N_TILE))
    tCcS = thr_mma.partition_C(cS)
    rowcol_layout, threads_per_row = _sm80_m16n8_acc_rowcol_policy(tCrS.layout)
    tCrS_rc = cute.make_tensor(tCrS.iterator, rowcol_layout)

    num_rows, num_cols = cute.size(rowcol_layout, mode=[0]), cute.size(rowcol_layout, mode=[1])

    ms = cute.make_rmem_tensor((num_rows, ), cutlass.Float32)
    ls = cute.make_rmem_tensor((num_rows, ), cutlass.Float32)
    alphas = cute.make_rmem_tensor((num_rows, ), cutlass.Float32)
    ms.fill(NEG_INF)
    ls.fill(0.0)

    for t in cutlass.range_constexpr(num_tiles):
        gK_tile = cute.local_tile(gK, (N_TILE, HEAD_DIM), (t, 0))

        tKgK = thr_mma.partition_B(gK_tile)
        tCrK = thr_mma.make_fragment_B(tKgK)
        cute.autovec_copy(tKgK, tCrK)

        tCrS.fill(0.0)
        cute.gemm(tiled_mma, tCrS, tCrQ, tCrK, tCrS)
        _online_softmax_update(tCrS_rc, ms, ls, alphas, threads_per_row)

    if tidx % threads_per_row == 0:
        for r in cutlass.range_constexpr(num_rows):
            frag_idx = rowcol_layout((r, 0))
            gr , _ = tCcS[frag_idx]
            gM[gr] = ms[r]
            gL[gr] = ls[r]

    # ---- 1. Partition and load Q once ----
    # Unpack cute.arch.thread_idx(), get this lane's ThrMma slice, partition_A(gQ),
    # create the A fragment, and copy Q into registers.

    # ---- 2. Allocate one 32x16 FP32 score fragment ----
    # Use partition_shape_C((M_QUERIES, N_TILE)) plus make_fragment_C().
    # Checkpoint: tCrS.layout should be ((2,2),2,2):((1,2),4,8).

    # ---- 3. Derive a per-thread (row, column) view ----
    # Query tiled_mma.tv_layout_C and tiled_mma.shape_mnk. Flatten the thread/value
    # shapes and strides. For every TV stride s, cute.idx2crd(s, (atom_m, atom_n))
    # gives (delta_m, delta_n):
    #   delta_n == 0 -> row mode
    #   delta_m == 0 -> column mode
    # Reject zero/mixed movements. Match value leaves to tCrS.layout mode 0, append
    # MMA_M to the row axis and MMA_N to the column axis, then build rowcol_layout.
    # Re-view the same registers with cute.make_tensor(tCrS.iterator, rowcol_layout).
    # Checkpoint: ((2,2),(2,2)):((2,4),(1,8)), hence 4 rows x 4 columns per lane.

    # ---- 4. Derive and validate the cross-lane reduction group ----
    # The thread leaves with delta_m == 0 share a logical row. Their product is
    # threads_per_row. Require those leaves to be a low-order contiguous prefix and
    # require threads_per_row to be a power of two no larger than WARP_SIZE.
    # Checkpoint for this atom: threads_per_row == 4. Do not write a literal 4 into
    # the algorithm below.

    # ---- 5. Create layout-shaped online state ----
    # Derive num_rows/num_cols with cute.size(rowcol_layout, mode=[...]). Create
    # compact FP32 rmem tensors m/l from row_shape and initialize them to NEG_INF/0.
    # This replaces m_lower/m_upper/l_lower/l_upper.

    # ---- 6. Run QK and update online softmax for every K tile ----
    # Use cutlass.range_constexpr: fragment/list indices must remain Python ints.
    # For each tile: load K, clear scores, cute.gemm, then for every local row:
    #   tile_m = local max over num_cols
    #   tile_m = warp_reduction_max(..., threads_in_group=threads_per_row)
    #   m_new  = max(m_old, tile_m)
    #   alpha  = exp(m_old - m_new)
    #   tile_l = local sum(exp(score - m_new))
    #   tile_l = warp_reduction_sum(..., threads_in_group=threads_per_row)
    #   l      = l * alpha + tile_l
    #   m      = m_new

    # ---- 7. Derive output coordinates and write one copy per row ----
    # Partition cute.make_identity_tensor((M_QUERIES, N_TILE)) with the same ThrMma.
    # rowcol_layout((r, 0)) identifies a representative fragment slot; indexing the
    # coordinate fragment at that slot gives the actual global row. Only the leader
    # of each derived reduction group writes m/l.
    pass


@cute.jit
def _online_softmax_layout_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mM: cute.Tensor,
    mL: cute.Tensor,
    num_tiles: cutlass.Constexpr,
) -> None:
    """Construct one m16n8k16 MMA atom and launch one warp."""
    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(
            warp.MmaF16BF16Op(
                cutlass.BFloat16,
                cutlass.Float32,
                (16, 8, HEAD_DIM),
            )
        )
    )
    _online_softmax_layout_kernel(
        mQ,
        mK,
        mM,
        mL,
        tiled_mma,
        num_tiles,
    ).launch(
        grid=(1, 1, 1),
        block=(cute.arch.WARP_SIZE, 1, 1),
    )


def cute_online_softmax_layout(
    Q: torch.Tensor,
    K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute layout-derived online-softmax statistics for ``S = Q @ K.T``."""
    if not Q.is_cuda or not K.is_cuda:
        raise ValueError("Q and K must be CUDA tensors")
    if Q.device != K.device:
        raise ValueError("Q and K must be on the same CUDA device")
    if Q.dtype is not torch.bfloat16 or K.dtype is not torch.bfloat16:
        raise ValueError("cute_online_softmax_layout requires BF16 Q and K")
    if Q.shape != (M_QUERIES, HEAD_DIM):
        raise ValueError(
            f"Q must have shape ({M_QUERIES}, {HEAD_DIM}), got {tuple(Q.shape)}"
        )
    if K.dim() != 2 or K.shape[1] != HEAD_DIM:
        raise ValueError(f"K must have shape (kv_len, {HEAD_DIM}), got {tuple(K.shape)}")
    kv_len = K.shape[0]
    if kv_len == 0 or kv_len % N_TILE != 0:
        raise ValueError(f"kv_len must be a positive multiple of {N_TILE}, got {kv_len}")

    # Remove this gate only after completing the kernel. The correctness tests use
    # xfail(raises=NotImplementedError), so a wrong implementation remains a failure.

    Q = Q.contiguous()
    K = K.contiguous()
    m = torch.empty(M_QUERIES, dtype=torch.float32, device=Q.device)
    l = torch.empty(M_QUERIES, dtype=torch.float32, device=Q.device)

    _online_softmax_layout_launch(
        from_dlpack(Q),
        from_dlpack(K),
        from_dlpack(m),
        from_dlpack(l),
        kv_len // N_TILE,
    )
    return m, l
