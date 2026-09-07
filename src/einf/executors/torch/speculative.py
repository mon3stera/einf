"""Speculative decoding: draft verification with accept/reject semantics.

Pipeline position (v1 = chain, EAGLE roadmap):

    draft loop  ->  verify forward (target, K+1 positions)  ->  accept/reject  ->  commit

The verify attention runs on FlashInver paged prefill: chain uses the
causal fast path, trees swap in ``packed_custom_mask`` (verified in
benchmarks/probe_spec_verify.py). This module owns the accept/reject
math, which is pure logic and fully CPU-testable.

Contract (shared by greedy and sampling):

    draft_tokens  [B, K]     int64, drafts proposed by the draft model
    target_logits [B, K+1, V] target logits; ``target_logits[b, i]`` is the
                             distribution over draft slot ``i`` (the token
                             that follows prefix + drafts[:i]); slot K is
                             the bonus position after all K drafts.

    returns (accepted_tokens [B, K+1] int64 padded with -1,
             accepted_len [B] int64 = accepted drafts + 1 correction/bonus)

Distribution invariant: with sampling, the committed tokens follow the
TARGET distribution exactly, regardless of how bad the draft is. The
greedy variant is exact only under greedy decoding.

EAGLE notes: EAGLE-3 changes the draft-head training, not this module;
EAGLE-2 replaces chain masks with tree masks (verify kernel unchanged)
and needs token-tree verification, not this sequential scan.
"""

from __future__ import annotations

import torch


def chain_verify_mask(k_len: int, prefix_len: int) -> torch.Tensor:
    """Build the [k_len, prefix_len + k_len] boolean chain verify mask.

    Row i (draft slot i) attends to the whole prefix and drafts 0..i.
    Equivalent to the causal fast path; kept explicit as the mask-builder
    seam where EAGLE-2 tree masks will plug in.
    """
    causal = torch.tril(torch.ones(k_len, k_len, dtype=torch.bool))
    prefix = torch.ones(k_len, prefix_len, dtype=torch.bool)
    return torch.cat([prefix, causal], dim=-1)


def accept_reject_greedy(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy accept/reject: EXERCISE (implement me).

    Sequential scan per batch row: accept draft slot i while
    ``draft_tokens[b, i] == argmax(target_logits[b, i])``. At the first
    mismatch (or after all K acceptances) append ``argmax(target_logits[b, L])``
    as the correction (mismatch) or bonus (all accepted) token.

    Returns (accepted_tokens [B, K+1] int64, -1 padded; accepted_len [B]).
    """

    B, T = draft_tokens.shape

    device = draft_tokens.device

    # [B, T]
    tids = torch.argmax(target_logits, dim=-1)

    # [B, T + 1]
    is_accept = draft_tokens == tids[:, :T]

    # [B]
    num_accepted = torch.cumprod(is_accept, dim=1).sum(dim=1)

    # [B, T + 1]
    out = torch.zeros((B, T + 1), dtype=torch.int64, device=device)
    out.fill_(-1)

    accept_len = num_accepted + 1

    mask = torch.arange(T + 1, device=device).unsqueeze(0) < accept_len.unsqueeze(1)

    out[mask] = tids[mask]

    return out, accept_len


def accept_reject_sampling(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stochastic accept/reject: EXERCISE (implement me).

    Sequential rejection sampling per batch row, per draft slot i:

        p = softmax(target_logits[b, i])          # target distribution
        d = draft_probs[b, i]                     # draft distribution
        accept draft x with probability min(1, p[x] / d[x])

    On rejection, sample the correction token from the residual
    distribution ``norm(max(p - d, 0))`` — the FULL draft distribution is
    subtracted, independent of which x was rejected; this is what makes
    ``min(d, p) + max(p - d, 0) = p`` close the invariant. Then stop the
    scan. If all K drafts are accepted, sample the bonus token from
    ``softmax(target_logits[b, K])``.

    Assume ``d[x] > 0`` for every proposed x (multinomial guarantees it).
    Guard the degenerate residual (all-zero) by falling back to p.

    Returns (accepted_tokens [B, K+1] int64, -1 padded; accepted_len [B]).
    """

    B, K = draft_tokens.shape
    V = target_logits.shape[-1]
    device = draft_tokens.device

    target_probs = torch.softmax(target_logits, dim=-1)  # [B, K + 1, V]
    p = target_probs[:, :K, :]                          # [B, K, V]
    p_bonus = target_probs[:, K:K+1, :]                 # [B, 1, V]
    d = draft_probs                                     # [B, K, V]

    draft_tokens_exp = draft_tokens.unsqueeze(-1)       # [B, K, 1]
    p_x = torch.gather(p, dim=-1, index=draft_tokens_exp).squeeze(-1)  # [B, K]
    d_x = torch.gather(d, dim=-1, index=draft_tokens_exp).squeeze(-1)  # [B, K]

    u = torch.rand((B, K), device=device, dtype=p_x.dtype, generator=generator)
    is_accept = u < torch.clamp(p_x / d_x, max=1.0)     # [B, K]

    num_accepted = is_accept.long().cumprod(dim=1).sum(dim=1)  # [B]
    accepted_len = num_accepted + 1                            # [B]

    res = torch.clamp(p - d, min=0.0)                          # [B, K, V]
    res_sum = res.sum(dim=-1, keepdim=True)                    # [B, K, 1]
    res_dist = torch.where(res_sum > 0, res / torch.clamp(res_sum, min=1e-12), p)

    candidate_dists = torch.cat([res_dist, p_bonus], dim=1)    # [B, K + 1, V]

    gather_idx = num_accepted.view(B, 1, 1).expand(-1, 1, V)   # [B, 1, V]
    selected_dist = torch.gather(candidate_dists, dim=1, index=gather_idx).squeeze(1) # [B, V]

    next_token = torch.multinomial(selected_dist, num_samples=1, generator=generator).squeeze(-1) # [B]

    out = torch.full((B, K + 1), -1, dtype=torch.int64, device=device)

    positions = torch.arange(K, device=device).unsqueeze(0)    # [1, K]
    mask_draft = positions < num_accepted.unsqueeze(1)         # [B, K]
    out[:, :K] = torch.where(mask_draft, draft_tokens, out[:, :K])

    out.scatter_(dim=1, index=num_accepted.unsqueeze(1), src=next_token.unsqueeze(1))

    return out, accepted_len
    


def accept_reject_greedy_reference(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference greedy accept/reject (slow, per-row loop)."""
    B, K = draft_tokens.shape
    out = torch.full((B, K + 1), -1, dtype=torch.int64)
    accepted = torch.zeros(B, dtype=torch.int64)

    for b in range(B):
        picks = target_logits[b].argmax(dim=-1)
        L = 0
        while L < K and draft_tokens[b, L] == picks[L]:
            out[b, L] = draft_tokens[b, L]
            L += 1

        out[b, L] = picks[L]
        accepted[b] = L + 1

    return out, accepted


def accept_reject_sampling_reference(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference rejection sampling (slow, per-row loop, float64)."""
    B, K = draft_tokens.shape
    out = torch.full((B, K + 1), -1, dtype=torch.int64)
    accepted = torch.zeros(B, dtype=torch.int64)
    rand_device = generator.device if generator is not None else draft_tokens.device

    for b in range(B):
        L = 0
        while L < K:
            p = torch.softmax(target_logits[b, L].double(), dim=-1)
            x = draft_tokens[b, L].item()
            d = draft_probs[b, L].double()
            r = torch.rand((), generator=generator, device=rand_device).item()

            if r * d[x] <= p[x]:
                out[b, L] = x
                L += 1
                continue

            residual = (p - d).clamp_min(0.0)
            total = residual.sum()

            if total > 0:
                pick = torch.multinomial(residual / total, 1, generator=generator)
            else:
                pick = torch.multinomial(p, 1, generator=generator)

            out[b, L] = int(pick.item())
            break

        if L == K:
            p_bonus = torch.softmax(target_logits[b, K].double(), dim=-1)
            bonus = torch.multinomial(p_bonus, 1, generator=generator)
            out[b, K] = int(bonus.item())
            accepted[b] = K + 1
        else:
            accepted[b] = L + 1

    return out, accepted
