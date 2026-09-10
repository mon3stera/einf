"""Stage 3: MoE expert MLP — grouped GEMM over routed (token, expert) pairs.

Pipeline (mirrors vLLM's ``fused_moe`` structure)::

    hidden [T, H]
      │  moe_align_block_size      (torch): sort (token, slot) pairs by
      │                                    expert, pad each expert's run to
      │                                    BLOCK_M, emit per-block expert ids
      ▼
    sorted_ids [P]  (pair id or -1 sentinel), expert_ids [P/BLOCK_M],
    num_tokens_post_padded [1]
      │
      │  moe_grouped_gemm_kernel  #1  (GATHER_A=True,  APPLY_WEIGHT=False)
      │    c1[row, 2I] = A_row @ w13[e].T        rows follow sorted order
      ▼
    silu_and_mul_triton (stage 1 kernel, reused)  →  act [P, I]
      │
      │  moe_grouped_gemm_kernel  #2  (GATHER_A=False, APPLY_WEIGHT=True)
      │    c2[row, H] = act_row @ w2[e].T * topk_weight(pair)
      ▼
    moe_combine (torch index_add):  out[t] += c2[pos(t, slot)]

Kernel body = the exercise (search ``YOUR CODE HERE``). Everything else —
alignment, reference, combine, tests, benchmark — is provided.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

import triton
import triton.language as tl

from einf.executors.torch.triton_ops import silu_and_mul_triton


# ---------------------------------------------------------------------------
# Alignment (torch reference-grade; not the exercise)
# ---------------------------------------------------------------------------


def moe_align_block_size(
    topk_ids: Tensor,
    block_m: int,
    num_experts: int,
) -> tuple[Tensor, Tensor, Tensor, int]:
    """Sort (token, slot) pairs by expert and pad each run to ``block_m``.

    Returns ``(sorted_ids, expert_ids, num_tokens_post_padded, total_padded)``:

    - ``sorted_ids`` [total_padded] int32: flat pair ids (``t * K + slot``)
      grouped by expert, each expert's run padded to a ``block_m`` multiple
      with ``-1`` sentinels. Rows of the grouped GEMMs index this 1:1.
    - ``expert_ids`` [total_padded // block_m] int32: expert per GEMM block.
    - ``num_tokens_post_padded`` [1] int32 device tensor, read by the kernel
      to early-exit over-provisioned blocks without a host sync.
    - ``total_padded`` int (host): real padded length for allocations.
    """
    flat = topk_ids.reshape(-1).to(torch.int32)
    num_pairs = flat.numel()
    device = flat.device

    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_m - 1) // block_m * block_m
    block_counts = padded // block_m

    exclusive_counts = torch.cumsum(counts, 0) - counts
    exclusive_padded = torch.cumsum(padded, 0) - padded

    # Stable sort groups pairs by expert preserving pair order. The rank of
    # the j-th sorted pair inside its expert's run is j minus the expert's
    # first flat position, computed from the sorted side (a rank computed
    # from the flat side would count all pairs, not same-expert pairs).
    order = torch.argsort(flat, stable=True)
    experts_sorted = flat[order]
    j = torch.arange(num_pairs, device=device)
    pos = exclusive_padded[experts_sorted] + (j - exclusive_counts[experts_sorted])

    sorted_ids = torch.full((int(padded.sum()),), -1, dtype=torch.int32, device=device)
    sorted_ids[pos] = order.to(torch.int32)

    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=device), block_counts
    )

    num_tokens_post_padded = padded.sum().reshape(1).to(torch.int32)

    return (
        sorted_ids,
        expert_ids.to(torch.int32),
        num_tokens_post_padded,
        int(padded.sum()),
    )


# ---------------------------------------------------------------------------
# The exercise: one grouped-GEMM kernel serving both expert GEMMs
# ---------------------------------------------------------------------------


@triton.jit
def moe_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    weights_ptr,
    num_tokens_post_padded_ptr,
    k_pairs,
    h_dim,
    n_dim,
    stride_a_row,
    stride_b_expert,
    stride_b_row,
    stride_c_row,
    GATHER_A: tl.constexpr,
    APPLY_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program computes one ``BLOCK_M x BLOCK_N`` tile of one GEMM block.

    Grid: ``(grid_m, cdiv(n_dim, BLOCK_N))`` where ``grid_m`` is
    over-provisioned as ``cdiv(T*K, BLOCK_M) + num_experts``; programs whose
    tile starts past ``num_tokens_post_padded`` exit immediately.

    Arg    ----
    a_ptr: GEMM A operand. GEMM #1: ``hidden`` [T, H] gathered by token;
        GEMM #2: ``act`` [P, I] read contiguously in sorted order.
    b_ptr: expert weights [E, N, H_dim], contiguous last dim; the expert
        for this block comes from ``expert_ids``.
    c_ptr: output [P, n_dim], rows in sorted order.
    sorted_ids_ptr: [P] int32 pair ids (``t * K + slot``), ``-1`` sentinel.
    weights_ptr: [T*K] topk weights, only read when ``APPLY_WEIGHT``.
    k_pairs: K (pairs decode as ``token = pair_id // k_pairs``).
    h_dim / n_dim: reduction and output column counts.

    Body outline (the exercise):

    1. ``pid_m = tl.program_id(0)``, ``pid_n = tl.program_id(1)``. Load
       ``num_tokens_post_padded`` and early-return when
       ``pid_m * BLOCK_M >= post_padded``.
    2. ``expert = tl.load(expert_ids_ptr + pid_m)``.
    3. ``rows_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)``; load
       ``pair_ids = tl.load(sorted_ids_ptr + rows_local)`` (no mask — the
       array is already padded to BLOCK_M multiples); row validity is
       ``row_mask = pair_ids >= 0``.
    4. A row indices: when ``GATHER_A``, decode
       ``a_rows = tl.where(row_mask, pair_ids // k_pairs, 0)``; otherwise
       ``a_rows = rows_local`` (the sorted activation buffer is contiguous).
    5. ``n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)``;
       ``n_mask = n < n_dim``.
    6. ``acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)``.
    7. Reduction loop ``for h0 in range(0, h_dim, BLOCK_H)``:
       - ``hs = h0 + tl.arange(0, BLOCK_H)``;
         ``h_mask = hs < h_dim`` (h_dim need not be a multiple of BLOCK_H)
       - A tile ``[BLOCK_M, BLOCK_H]``:
         ``tl.load(a_ptr + a_rows[:, None] * stride_a_row + hs[None, :],
         mask=row_mask[:, None] & h_mask[None, :], other=0.0)``
       - B tile ``[BLOCK_N, BLOCK_H]`` from expert ``expert``:
         ``tl.load(b_ptr + expert * stride_b_expert + n[:, None] * stride_b_row
         + hs[None, :], mask=n_mask[:, None] & h_mask[None, :], other=0.0)``
       - ``acc = tl.dot(a, tl.trans(b), acc)`` — the accumulate form;
         BLOCK_M/BLOCK_N/BLOCK_H must all be >= 16 for tensor cores.
    8. When ``APPLY_WEIGHT``: gather this tile's routing weights
       ``w = tl.load(weights_ptr + pair_ids, mask=row_mask, other=0.0)``
       and scale ``acc = acc * w[:, None]`` (fp32 — A/B tiles cast to fp32
       by the dot accumulator).
    9. Store: ``tl.store(c_ptr + rows_local[:, None] * stride_c_row +
       n[None, :], acc, mask=row_mask[:, None] & n_mask[None, :])``.
       Sentinel rows are never stored — their garbage never propagates
       because every consumer re-masks with ``row_mask``.
    """

    pid_m, pid_n = tl.program_id(0), tl.program_id(1)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    sgrow = BLOCK_M * pid_m

    if num_tokens_post_padded <= sgrow:
        return

    expert = tl.load(expert_ids_ptr + pid_m)

    grows = sgrow + tl.arange(0, BLOCK_M)

    pair_ids = tl.load(sorted_ids_ptr + grows)

    row_masks = pair_ids != -1

    if GATHER_A:
        a_rows = tl.where(row_masks, pair_ids // k_pairs, 0)
    else:
        a_rows = grows

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < n_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for h in range(0, h_dim, BLOCK_H):
        hs = h + tl.arange(0, BLOCK_H)
        h_mask = hs < h_dim
        a_tile = tl.load(
            a_ptr + a_rows[:, None] * stride_a_row + hs[None, :], 
            mask=row_masks[:, None] & h_mask[None, :], 
            other=0.0
        )
        b_tile = tl.load(
            b_ptr + expert * stride_b_expert + n[:, None] * stride_b_row + hs[None, :],
            mask=n_mask[:, None] & h_mask[None, :],
            other=0.0
        )
        acc = tl.dot(a_tile, tl.trans(b_tile), acc)

    if APPLY_WEIGHT:
        w = tl.load(weights_ptr + pair_ids, mask=row_masks, other=0.0)
        acc = acc * w[:, None]
    
    tl.store(c_ptr + grows[:, None] * stride_c_row + n[None, :], acc, mask=row_masks[:, None] & n_mask[None, :]) 

# ---------------------------------------------------------------------------
# Combine (torch; not the exercise)
# ---------------------------------------------------------------------------


def moe_combine(
    c2: Tensor,
    sorted_ids: Tensor,
    topk_weights: Tensor,
) -> Tensor:
    """Scatter-sum scaled expert outputs back to token rows."""
    valid = sorted_ids >= 0
    pairs = sorted_ids[valid].long()
    num_tokens = topk_weights.shape[0]
    k_pairs = topk_weights.shape[1]
    tokens = pairs // k_pairs

    # fp32 accumulation across the K expert contributions (vLLM finalize
    # does the same); cast back to the pipeline dtype at the end.
    out = torch.zeros(
        (num_tokens, c2.shape[1]), dtype=torch.float32, device=c2.device
    )
    # c2 is allocated at the over-provisioned grid size; only the first
    # total_padded rows (= sorted_ids length) back the sorted layout.
    out.index_add_(0, tokens, c2[: sorted_ids.shape[0]][valid].float())
    return out.to(c2.dtype)


# ---------------------------------------------------------------------------
# Reference (torch eager; also the benchmark baseline)
# ---------------------------------------------------------------------------


def moe_expert_mlp_reference(
    hidden: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    w13: Tensor,
    w2: Tensor,
    *,
    quantize_intermediates: bool = False,
) -> Tensor:
    """Per-expert eager loop: the exact math the fused path must reproduce.

    With ``quantize_intermediates=True`` the intermediate GEMM outputs and
    activations are rounded to the input dtype between stages, mirroring the
    fused pipeline's bf16 storage. Comparisons against the fused path must
    use this mode — a full-fp32 reference would demand impossible precision
    from a bf16 pipeline.
    """
    dtype = hidden.dtype
    num_tokens, h_dim = hidden.shape
    k_pairs = topk_ids.shape[1]
    num_experts, n2, _ = w13.shape
    inter = n2 // 2

    out = torch.zeros(
        (num_tokens, h_dim), dtype=torch.float32, device=hidden.device
    )
    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_weights.reshape(-1).float()
    tokens = torch.arange(num_tokens, device=hidden.device).repeat_interleave(k_pairs)

    for expert in range(num_experts):
        sel = flat_ids == expert

        if not sel.any():
            continue

        x = hidden[tokens[sel]].float()
        h = x @ w13[expert].float().T

        if quantize_intermediates:
            h = h.to(dtype).float()

        act = F.silu(h[..., :inter]) * h[..., inter:]

        if quantize_intermediates:
            act = act.to(dtype).float()

        y = act @ w2[expert].float().T

        if quantize_intermediates:
            y = y.to(dtype).float()

        out.index_add_(0, tokens[sel], y * flat_w[sel][:, None])

    return out.to(dtype) if quantize_intermediates else out.to(dtype)

    return out


# ---------------------------------------------------------------------------
# Fused top-level path
# ---------------------------------------------------------------------------


def moe_expert_mlp_triton(
    hidden: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    w13: Tensor,
    w2: Tensor,
    *,
    block_m: int = 16,
    block_n: int = 32,
    block_h: int = 32,
) -> Tensor:
    """Full expert MLP: align -> grouped GEMM #1 -> silu -> grouped GEMM #2 -> combine."""
    num_tokens, h_dim = hidden.shape
    k_pairs = topk_ids.shape[1]
    num_experts, n2, _ = w13.shape

    hidden = hidden.contiguous()
    w13 = w13.contiguous()
    w2 = w2.contiguous()

    # tl.dot requires both operands in one dtype; the intermediate buffers
    # carry the input dtype through, so all four tensors must agree.
    assert hidden.dtype == w13.dtype == w2.dtype, (
        "hidden, w13, w2 must share one dtype for the fused MoE path"
    )

    sorted_ids, expert_ids, post_padded, total_padded = moe_align_block_size(
        topk_ids, block_m, num_experts
    )

    # Over-provisioned block count: each expert wastes at most one block on
    # padding. The kernel early-exits past num_tokens_post_padded, so no
    # host sync is needed for the grid.
    grid_m = (num_tokens * k_pairs + block_m - 1) // block_m + num_experts

    c1 = torch.empty(
        (grid_m * block_m, n2), dtype=hidden.dtype, device=hidden.device
    )
    grid1 = (grid_m, triton.cdiv(n2, block_n))
    moe_grouped_gemm_kernel[grid1](
        hidden,
        w13,
        c1,
        sorted_ids,
        expert_ids,
        topk_weights.reshape(-1),
        post_padded,
        k_pairs,
        h_dim,
        n2,
        hidden.stride(0),
        w13.stride(0),
        w13.stride(1),
        c1.stride(0),
        GATHER_A=True,
        APPLY_WEIGHT=False,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        num_warps=4,
    )

    # Garbage in sentinel/over-provisioned rows is fine: GEMM #2 re-masks
    # every A-row load with row_mask, so it never enters the dot.
    act = silu_and_mul_triton(c1)

    c2 = torch.empty(
        (grid_m * block_m, h_dim), dtype=hidden.dtype, device=hidden.device
    )
    grid2 = (grid_m, triton.cdiv(h_dim, block_n))
    moe_grouped_gemm_kernel[grid2](
        act,
        w2,
        c2,
        sorted_ids,
        expert_ids,
        topk_weights.reshape(-1),
        post_padded,
        k_pairs,
        act.shape[1],
        h_dim,
        act.stride(0),
        w2.stride(0),
        w2.stride(1),
        c2.stride(0),
        GATHER_A=False,
        APPLY_WEIGHT=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_H=block_h,
        num_warps=4,
    )

    return moe_combine(c2, sorted_ids, topk_weights)
