"""Triton fused ``silu_and_mul`` with autotuned launch configuration.

``out = silu(gate) * up`` over ``gate_up: [num_tokens, 2 * hidden]`` (see
``silu_and_mul_reference`` for the exact math).

The kernel body was written as the first hand-written Triton exercise; the
launch configuration is now autotuned: ``triton.autotune`` sweeps
``(BLOCK, num_warps)`` configs on the first call for each distinct
``(num_tokens, hidden)`` pair and caches the winner. Small batches favour
small blocks (more programs to fill the GPU), large batches favour large
blocks (fewer programs, less launch overhead) — one static config cannot
serve both, which is exactly what the sweep resolves.

Autotune caveats worth remembering:
- The sweep runs on the first call per key and is included in that call's
  latency. Serving code must warm the expected batch sizes at init (dummy
  calls), or the first real step at a new batch size pays the sweep.
- Keying on ``num_tokens`` means every distinct batch size re-tunes once.
  For production, keying on ``hidden`` only plus fixed batch buckets is the
  usual compromise.

Dispatch is env-gated: ``EINF_TRITON_SILU=1`` routes
``fused_ops.silu_and_mul`` through this kernel; FlashInfer delegation and
the CPU fallback remain the defaults.
"""

from __future__ import annotations

import os

import torch
from torch import Tensor

import triton
import triton.language as tl


def triton_silu_and_mul_enabled(gate_up: Tensor) -> bool:
    """Env gate: opt-in Triton path, CUDA only, bf16/fp16/fp32 inputs."""
    return (
        os.environ.get("EINF_TRITON_SILU") == "1"
        and gate_up.is_cuda
        and gate_up.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and gate_up.ndim == 2
        and gate_up.shape[-1] % 2 == 0
    )


def _silu_configs() -> list[triton.Config]:
    return [
        triton.Config({"BLOCK": block}, num_warps=warps)
        for block in (128, 256, 512, 1024, 2048, 4096)
        for warps in (2, 4, 8)
    ]


@triton.autotune(configs=_silu_configs(), key=["num_tokens", "hidden"])
@triton.jit
def silu_and_mul_kernel(
    gate_up_ptr,
    out_ptr,
    stride_gu_row,
    stride_out_row,
    num_tokens,
    hidden,
    BLOCK: tl.constexpr,
):
    """One program computes ``BLOCK`` output elements of one token row.

    Grid: ``(num_tokens, cdiv(hidden, BLOCK))`` — token rows on axis 0
    (the large dimension), column blocks on axis 1 (axis 1 caps at 65535).
    ``num_tokens`` is unused in the body but is the autotune cache key for
    the batch dimension.
    """

    row, col = tl.program_id(0), tl.program_id(1)

    cols = col * BLOCK + tl.arange(0, BLOCK)

    mask = cols < hidden

    gate_offs = row * stride_gu_row + cols
    up_offs = gate_offs + hidden

    gate = tl.load(gate_up_ptr + gate_offs, mask=mask, other=0.0)
    up = tl.load(gate_up_ptr + up_offs, mask=mask, other=0.0)

    gate_f32 = gate.to(tl.float32)
    silu = gate_f32 * tl.sigmoid(gate_f32)
    res = silu.to(gate.dtype) * up

    out_offs = row * stride_out_row + cols
    tl.store(out_ptr + out_offs, res, mask=mask)


def silu_and_mul_triton(gate_up: Tensor) -> Tensor:
    """Launch wrapper: allocate the output and start the autotuned kernel."""
    num_tokens, two_hidden = gate_up.shape
    hidden = two_hidden // 2
    out = torch.empty(
        (num_tokens, hidden),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )

    grid = lambda meta: (num_tokens, triton.cdiv(hidden, meta["BLOCK"]))  # noqa: E731
    silu_and_mul_kernel[grid](
        gate_up,
        out,
        gate_up.stride(0),
        out.stride(0),
        num_tokens,
        hidden,
    )
    return out


def silu_and_mul_reference(gate_up: Tensor) -> Tensor:
    """Pure-torch reference: the exact math the kernel must reproduce."""
    hidden = gate_up.shape[-1] // 2
    gate = gate_up[..., :hidden].float()
    up = gate_up[..., hidden:].float()
    return (gate * torch.sigmoid(gate) * up).to(dtype=gate_up.dtype)


# ---------------------------------------------------------------------------
# Stage 2: fused MoE routing (DeepSeek-V3 MoEGate semantics)
# ---------------------------------------------------------------------------
#
# One fused pass computes, per token row of router logits:
#
#     scores = softmax(logits)                        # fp32, max-stabilized
#     biased = scores + bias                          # optional bias correction
#     top-k experts are picked by *biased* scores
#     weights = scores gathered at the picked ids     # unbiased values
#     weights = weights / sum(weights) * scaling      # optional renorm
#
# The subtle part (and the exercise): the top-k selection runs k argmax
# rounds inside one program, keeping the selected ids and raw weights in
# *registers* via ``tl.where(slot_vec == i, ...)`` under
# ``tl.static_range`` — Triton cannot index-assign into a vector, so you
# build the k-length vectors by masked replacement per round.


@triton.jit
def moe_topk_softmax_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    stride_logits_row,
    stride_weights_row,
    stride_ids_row,
    num_experts,
    scaling,
    HAS_BIAS: tl.constexpr,
    RENORM: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program routes one token row.

    Grid: ``(num_tokens,)``. ``BLOCK_E`` must be
    ``next_power_of_2(num_experts)`` and ``BLOCK_K`` must be
    ``next_power_of_2(TOP_K)`` — every Triton block shape is power-of-two
    only. ``TOP_K`` stays the *exact* expert count: it drives the
    ``static_range`` round count and the store mask, while ``BLOCK_K`` is
    only the vector width. Padding lanes (``slot >= TOP_K``) never receive
    a weight, so renorm's ``tl.sum`` is unaffected by them.

    Args
    ----
    logits_ptr: [num_tokens, num_experts] router logits (any float dtype).
    bias_ptr: [num_experts] ``e_score_correction_bias``; only read when
        ``HAS_BIAS`` is true (pass any tensor when absent).
    weights_ptr / ids_ptr: [num_tokens, TOP_K] outputs (fp32 / int32).

    Body outline (the exercise):

    1. ``row = tl.program_id(0)``; expert vector
       ``e = tl.arange(0, BLOCK_E)`` with ``mask = e < num_experts``.
    2. Load the logits row with ``other=float("-inf")``, cast fp32.
    3. Softmax with max stabilization:
       ``m = tl.max(x, 0)`` -> ``p = tl.exp(x - m)`` -> scores = ``p /
       tl.sum(p, 0)``. Padded lanes stay -inf/0 throughout — check that.
    4. ``sel = scores + bias`` when ``HAS_BIAS`` else ``sel = scores``.
    5. Selection loop ``for i in tl.static_range(TOP_K)``:
       - ``idx = tl.argmax(sel, 0)`` (ties pick the smallest index, same as
         ``torch.topk``)
       - blank it out: ``sel = tl.where(e == idx, float("-inf"), sel)``
       - raw weight from the *unbiased* scores:
         ``w = tl.sum(tl.where(e == idx, scores, 0.0), 0)``
       - accumulate into register vectors: with
         ``slot = tl.arange(0, BLOCK_K)`` do
         ``w_vec = tl.where(slot == i, w, w_vec)`` and the same for ``idx``
         into ``id_vec`` (initialize both as ``(BLOCK_K,)`` zeros before
         the loop). Only lanes ``i < TOP_K`` are ever written — padding
         lanes stay zero.
    6. After the loop: if ``RENORM``, ``w_vec = w_vec / tl.sum(w_vec, 0)``;
       then ``w_vec = w_vec * scaling``. The sum needs no mask: padding
       lanes are still zero.
    7. Store ``w_vec`` to ``weights_ptr + row * stride_weights_row + slot``
       and ``id_vec`` (cast to ``ids_ptr.dtype.element_ty``) to
       ``ids_ptr + row * stride_ids_row + slot``, both with
       ``mask=slot < TOP_K`` — BLOCK_K may over-pad TOP_K (e.g. k=6 -> 8).
    """
    row = tl.program_id(0)

    e = tl.arange(0, BLOCK_E)

    mask = e < num_experts

    logits_off = row * stride_logits_row + e
    logits = tl.load(logits_ptr + logits_off, mask=mask, other=float("-inf"))

    bias = tl.load(bias_ptr + e, mask=mask, other=float("-inf"))

    # fp32 math regardless of the logits dtype (bf16 in serving), matching
    # the reference's ``.float()`` before softmax.
    logits_f = logits.to(tl.float32)

    m = tl.max(logits_f, 0)
    p = tl.exp(logits_f - m)
    scores = p / tl.sum(p, 0)

    vals = (scores + bias) if HAS_BIAS else scores

    topk_ids = tl.zeros((BLOCK_K,), dtype=tl.int32)
    topk_weights = tl.zeros((BLOCK_K,), dtype=tl.float32)
    slot = tl.arange(0, BLOCK_K)

    for i in tl.static_range(TOP_K):
        m_idx = tl.argmax(vals, axis=0)
        vals = tl.where(e == m_idx, float("-inf"), vals)
        w = tl.sum(tl.where(e == m_idx, scores, 0.0), 0)
        topk_ids = tl.where(slot == i, m_idx, topk_ids)
        topk_weights = tl.where(slot == i, w, topk_weights)

    if RENORM:
        topk_weights = topk_weights / tl.sum(topk_weights, 0)

    topk_weights = topk_weights * scaling

    # BLOCK_K over-pads TOP_K (k=6 -> 8): padding lanes stayed zero through
    # renorm, but they must not reach memory.
    store_mask = slot < TOP_K

    weight_offs = row * stride_weights_row + slot
    ids_offs = row * stride_ids_row + slot

    tl.store(weights_ptr + weight_offs, topk_weights, mask=store_mask)
    tl.store(ids_ptr + ids_offs, topk_ids, mask=store_mask)


def moe_topk_softmax_triton(
    logits: Tensor,
    bias: Tensor | None,
    *,
    top_k: int,
    scaling: float = 1.0,
    renorm: bool = True,
) -> tuple[Tensor, Tensor]:
    """Launch wrapper: route ``logits`` [num_tokens, num_experts]."""
    num_tokens, num_experts = logits.shape
    if top_k > num_experts:
        raise ValueError("top_k cannot exceed num_experts")

    weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=logits.device
    )
    ids = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=logits.device
    )

    block_e = triton.next_power_of_2(num_experts)
    block_k = triton.next_power_of_2(top_k)

    moe_topk_softmax_kernel[(num_tokens,)](
        logits,
        bias if bias is not None else logits,
        weights,
        ids,
        logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        num_experts,
        scaling,
        HAS_BIAS=bias is not None,
        RENORM=renorm,
        TOP_K=top_k,
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return weights, ids


def moe_topk_softmax_reference(
    logits: Tensor,
    bias: Tensor | None,
    *,
    top_k: int,
    scaling: float = 1.0,
    renorm: bool = True,
) -> tuple[Tensor, Tensor]:
    """Pure-torch reference: DeepSeek-V3 ``MoEGate`` forward semantics."""
    scores = torch.softmax(logits.float(), dim=-1)
    biased = scores + bias.float() if bias is not None else scores
    ids = torch.topk(biased, top_k, dim=-1).indices
    weights = scores.gather(-1, ids)
    if renorm:
        weights = weights / weights.sum(-1, keepdim=True)
    weights = weights * scaling
    return weights, ids.to(torch.int32)
