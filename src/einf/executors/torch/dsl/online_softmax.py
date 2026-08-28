"""第 2 级：Online Softmax（CuTe DSL 实现）

阶梯位置
--------
    第 1 级  cute_mma_qk           一个 m16n8k16 BF16 atom 算 Q @ K^T      [已完成，C++]
    第 2 级  cute_online_softmax   在累加器布局上做在线 row softmax        [本文件]
    第 3 级  cute_flash_attention  加上 P @ V 与 O 的在线 rescale           [待做]

契约
----
    Q : (16, 16)      bf16    16 个 query，head_dim = 16（正好是一个 atom 的 K，无 K 循环）
    K : (kv_len, 16)  bf16    kv_len 必须是 8 的倍数（8 = atom 的 N，即一个 KV tile）
    →
    m : (16,)  f32           每行的最终 max，即 S.max(dim=-1)，S = Q @ K^T
    l : (16,)  f32           每行的最终 sum，即 exp(S - m).sum(dim=-1)

不做 1/sqrt(d) 缩放（与第 1 级保持一致）。softmax 本身可由 exp(S - m) / l 还原。

为什么输出是 (m, l) 而不是 P
---------------------------
真正的 FA **从不落地 P**：P 一算出来就立刻消耗进 acc = acc * alpha + P @ V。
所以第 2 级的产物就该是「留在寄存器里、跨 tile 携带的行统计量」，而不是一个 (16, kv_len) 矩阵。
第 3 级只需加两件事：V 的 MMA，以及把这里作用在标量 l 上的 alpha 改为同时作用在 acc 上。

累加器归属（m16n8k16，已实测；探针见 note/.../cute-probes/acc_rowcol.cu）
------------------------------------------------------------------------
    partition_C 给出 ((2, 2), 1, 1)，那个 (2, 2) 是：
        mode 0 (v0) = 相邻两列        col = 2 * (lane % 4) + v0
        mode 1 (v1) = 相隔 8 的两行   row = lane // 4 + 8 * v1

    扁平下标 i 与 (v0, v1) 的关系是 colex：v0 = i % 2，v1 = i // 2
        → 第 v1 行的两个元素是 i = 2 * v1 和 i = 2 * v1 + 1

    关键推论：**一行的 8 列恰好住在 4 个连续 lane 里**
        row r 与 row r+8 都在 lane 4r .. 4r+3
        所以 row 归约 = 寄存器内 2 个（列）+ quad 内 2 步 butterfly shuffle。
        且 quad 是 4 对齐的，xor 1 / xor 2 天然不出界，不需要算 mask。

    每个 lane 持有 2 行 → 每个 lane 只需 2 组 (m, l)，共 4 个寄存器。

三个坑（前两个是算法的，第三个是 DSL 的）
----------------------------------------
1. **不要用 cute.arch.warp_reduction_max / warp_reduction_sum。**
   它们跨全部 32 个 lane 归约，会把 8 个不同的行混成一个值。
   这里要的是 quad 内（4 lane）归约，正确工具是
   ``cute.arch.shuffle_sync_bfly(value, offset)``，offset 取 1 和 2。

2. **m 的初值用有限的 NEG_INF，不要用 float("-inf")。**
   第一个 tile 上 alpha = exp(m_old - m_new)，若两者都是 -inf 就得到 -inf - -inf = NaN。
   这与 paged_attention_split_kv 里那个 ``l_n == 0`` 跳过守卫是同一类 NaN 防护。

3. **本文件不能加 ``from __future__ import annotations``。**
   那行会把所有注解变成惰性字符串，于是 ``num_tiles: cutlass.Constexpr`` 变成字符串
   ``"cutlass.Constexpr"``，DSL 认不出它是编译期常量，把 num_tiles 当成运行期值，
   随后 ``cutlass.range_constexpr(num_tiles)`` 报
   ``DSLRuntimeError: range_constexpr requires constexpr ...``。
   报错信息完全没提注解，很难反推 —— 所以这里刻意不写那行 import。

4. **``range`` 与 ``range_constexpr`` 语义完全不同，而且和写在哪有关。**（本轮实测）

   ::

       for i in cutlass.range_constexpr(n):   # 编译期展开，i 是 Python int   <- 你要这个
       for i in range(n):                     # 被改写成【运行期】循环，i 是 ArithValue
       [f(i) for i in range(n)]               # 列表推导不被改写，i 仍是 Python int

   用 i 去索引 fragment（比如 ``tCrS[2 * v1]``）必须是 Python int，所以 tile 循环和
   行循环都要用 ``range_constexpr``。误用普通 ``range`` 的报错是
   ``'ArithValue' object cannot be interpreted as an integer``，
   而它的 Suggestions 会让你去加 ``dynamic_expr`` —— **别听**，那是反方向。

   反过来，``range_constexpr`` 只能出现在 ``for`` **语句**里，不能写进列表推导
   （预处理器不改写推导式），推导式里用普通 ``range`` 正好。

一处刻意的取舍
--------------
本文件是**手算架构布局**的版本：``tCrS[2*v1]``、``bfly(x,1)/(x,2)``、``tidx % 4``、
两个标量 m/l —— 四处都写死了 ``SM80_16x8x16 + AtomLayoutMNK=(1,1,1)``。
这是刻意的：先用最接近原味 CUDA 的方式跑通，建立对 fragment 的硬件直觉。

之后会有一个 layout-derived 的重构版（行列数从 ``cute.size`` 取、状态用 fragment、
归约宽度从 ``tv_layout_C`` 里"行方向 stride 为 0 的线程 mode"推出、坐标用
``partition_C(make_identity_tensor(...))`` 而非公式），并把 tile 放大到 32x16
让 ``MMA_M = MMA_N = 2``，那时泛化才不是仪式。两版对照才看得出 CuTe 换来了什么。
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp as warp
import torch
from cutlass.cute.runtime import from_dlpack

__all__ = ["cute_online_softmax"]

M_QUERIES = 16  # atom 的 M
N_TILE = 8  # atom 的 N，一个 KV tile 的列数
HEAD_DIM = 16  # atom 的 K
NEG_INF = -1.0e30  # 见上文「三个坑」第 2 条


@cute.kernel
def _online_softmax_kernel(
    gQ: cute.Tensor,  # (16, 16)     bf16
    gK: cute.Tensor,  # (kv_len, 16) bf16
    gM: cute.Tensor,  # (16,)        f32  输出：行 max
    gL: cute.Tensor,  # (16,)        f32  输出：行 sum
    tiled_mma: cute.TiledMma,
    num_tiles: cutlass.Constexpr,
) -> None:
    """一个 warp（32 线程）在线归约 16 x kv_len 的 score 矩阵。

    TODO(第 2 级练习)：实现下面六步。每一步的机制都已在上面的 docstring 里给出。

    小贴士：``print(x.layout)`` 写在 @cute.kernel 里是**编译期**执行的，
    可以直接把任何中间张量的布局打出来核对；``cute.printf`` 才是运行期打印。
    """
    tidx, _, _ = cute.arch.thread_idx()
    
    thr_mma = tiled_mma.get_slice(tidx)

    tQgQ = thr_mma.partition_A(gQ)
    tCrQ = thr_mma.make_fragment_A(tQgQ)
    cute.autovec_copy(tQgQ, tCrQ)

    m_lower = cutlass.Float32(NEG_INF)
    m_upper = cutlass.Float32(NEG_INF)
    l_lower = 0.0
    l_upper = 0.0
     
    for t in cutlass.range_constexpr(num_tiles):
        gK_tile = cute.local_tile(gK, (N_TILE, HEAD_DIM), (t, 0))
        
        tKgK = thr_mma.partition_B(gK_tile)
        tCrK = thr_mma.make_fragment_B(tKgK)
        cute.autovec_copy(tKgK, tCrK)
        
        tCrS = thr_mma.make_fragment_C(thr_mma.partition_shape_C((M_QUERIES, N_TILE)))
        tCrS.fill(0.0)
        
        cute.gemm(tiled_mma, tCrS, tCrQ, tCrK, tCrS)

        tile_m_lower = cutlass.max(tCrS[0], tCrS[1])
        tile_m_upper = cutlass.max(tCrS[2], tCrS[3])

        for offset in cutlass.range_constexpr(1, 3):
            tile_m_lower = cutlass.max(tile_m_lower, cute.arch.shuffle_sync_bfly(tile_m_lower, offset))
            tile_m_upper = cutlass.max(tile_m_upper, cute.arch.shuffle_sync_bfly(tile_m_upper, offset))

        m_lower_new, m_upper_new = cutlass.max(m_lower, tile_m_lower), cutlass.max(m_upper, tile_m_upper)
        
        alpha_lower, alpha_upper = cute.math.exp(m_lower - m_lower_new), cute.math.exp(m_upper - m_upper_new)

        tile_l_lower = cute.math.exp(tCrS[0] - m_lower_new) + cute.math.exp(tCrS[1] - m_lower_new)
        tile_l_upper = cute.math.exp(tCrS[2] - m_upper_new) + cute.math.exp(tCrS[3] - m_upper_new)

        for offset in cutlass.range_constexpr(1, 3):
            tile_l_lower += cute.arch.shuffle_sync_bfly(tile_l_lower, offset)
            tile_l_upper += cute.arch.shuffle_sync_bfly(tile_l_upper, offset)

        l_lower = l_lower * alpha_lower + tile_l_lower
        l_upper = l_upper * alpha_upper + tile_l_upper

        m_lower, m_upper = m_lower_new, m_upper_new

    gid, t_in_g = tidx // 4, tidx % 4

    if t_in_g == 0:
        gM[gid], gL[gid] = m_lower, l_lower
        gM[gid + 8], gL[gid + 8] = m_upper, l_upper
        

    # ---- 步骤 1：取本线程的 MMA 切片 ----
    #   cute.arch.thread_idx() -> (tidx, tidy, tidz)
    #   tiled_mma.get_slice(tidx)

    # ---- 步骤 2：Q 只划分并加载一次（所有 KV tile 共用同一份 Q fragment）----
    #   partition_A -> make_fragment_A -> cute.autovec_copy
    #   注意 DSL 的 make_fragment_A 接收【已划分】的张量（C++ 的 partition_fragment_A 接收源张量）

    # ---- 步骤 3：初始化每行的在线状态 ----
    #   每个 lane 持有 2 行，所以是 2 组：m 初值 cutlass.Float32(NEG_INF)，l 初值 0.0

    # ---- 步骤 4：遍历 KV tile ----
    #   for t in cutlass.range_constexpr(num_tiles):
    #       cute.local_tile(gK, (N_TILE, HEAD_DIM), (t, 0))   取第 t 块 K
    #       partition_B -> make_fragment_B -> autovec_copy
    #       累加器：thr_mma.make_fragment_C(thr_mma.partition_shape_C((M_QUERIES, N_TILE)))
    #               .fill(0.0)，然后 cute.gemm(tiled_mma, tCrS, tCrQ, tCrK, tCrS)

    # ---- 步骤 5：对这个 tile 做在线更新（每行一次，本 lane 共 2 行）----
    #       row max ：先取本行 2 个元素（i = 2*v1, 2*v1+1）的 max，
    #                 再 quad 内 2 步 butterfly：shuffle_sync_bfly(x, 1) 与 (x, 2)
    #       m_new = max(m_old, m_tile)
    #       alpha = cute.math.exp(m_old - m_new)
    #       row sum：exp(s - m_new) 两个相加，再同样 2 步 butterfly
    #       l = l * alpha + row_sum      <- 这就是 FA 的 acc *= alpha，此处作用在标量上
    #       m = m_new                    <- 必须最后更新：alpha 用的是【旧】m

    # ---- 步骤 6：写回 ----
    #       2 步 shuffle 之后，quad 内 4 个 lane 的 m/l 已完全相同，
    #       所以只让其中一个 lane 写，避免 4 倍重复写。
    #       本 lane 的两行是 tidx // 4 和 tidx // 4 + 8。


@cute.jit
def _online_softmax_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mM: cute.Tensor,
    mL: cute.Tensor,
    num_tiles: cutlass.Constexpr,
) -> None:
    """构造 TiledMMA 并启动一个 warp。"""
    tiled_mma = cute.make_tiled_mma(
        cute.make_mma_atom(
            warp.MmaF16BF16Op(
                cutlass.BFloat16,  # A/B 类型
                cutlass.Float32,  # 累加器类型
                (M_QUERIES, N_TILE, HEAD_DIM),
            )
        )
    )
    _online_softmax_kernel(mQ, mK, mM, mL, tiled_mma, num_tiles).launch(
        grid=(1, 1, 1),
        block=(cute.arch.WARP_SIZE, 1, 1),
    )


def cute_online_softmax(Q: torch.Tensor, K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """在线计算 ``S = Q @ K.T`` 的逐行 softmax 统计量。

    Args:
        Q: ``(16, 16)`` BF16 CUDA 张量。
        K: ``(kv_len, 16)`` BF16 CUDA 张量，``kv_len`` 为 8 的正整数倍。

    Returns:
        ``(m, l)``，均为 ``(16,)`` FP32 CUDA 张量：``m`` 是逐行最大值，
        ``l`` 是 ``exp(S - m)`` 的逐行和。故 ``softmax(S, dim=-1) == exp(S - m) / l``。
    """
    if not Q.is_cuda or not K.is_cuda:
        raise ValueError("Q and K must be CUDA tensors")
    if Q.device != K.device:
        raise ValueError("Q and K must be on the same CUDA device")
    if Q.dtype is not torch.bfloat16 or K.dtype is not torch.bfloat16:
        raise ValueError("cute_online_softmax requires BF16 Q and K")
    if Q.shape != (M_QUERIES, HEAD_DIM):
        raise ValueError(f"Q must have shape ({M_QUERIES}, {HEAD_DIM}), got {tuple(Q.shape)}")
    if K.dim() != 2 or K.shape[1] != HEAD_DIM:
        raise ValueError(f"K must have shape (kv_len, {HEAD_DIM}), got {tuple(K.shape)}")
    kv_len = K.shape[0]
    if kv_len == 0 or kv_len % N_TILE != 0:
        raise ValueError(f"kv_len must be a positive multiple of {N_TILE}, got {kv_len}")

    Q = Q.contiguous()
    K = K.contiguous()
    m = torch.empty(M_QUERIES, dtype=torch.float32, device=Q.device)
    l = torch.empty(M_QUERIES, dtype=torch.float32, device=Q.device)

    _online_softmax_launch(
        from_dlpack(Q),
        from_dlpack(K),
        from_dlpack(m),
        from_dlpack(l),
        kv_len // N_TILE,
    )
    return m, l
