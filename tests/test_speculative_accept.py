"""Tests for speculative accept/reject semantics.

Reference tests pin the math; exercise tests (xfail until implemented)
compare the student implementations against the references. The
distribution invariant test is the acceptance gate for sampling.
"""

from __future__ import annotations

import pytest
import torch

from einf.executors.torch.speculative import (
    accept_reject_greedy,
    accept_reject_greedy_reference,
    accept_reject_sampling,
    accept_reject_sampling_reference,
    chain_verify_mask,
)


def _rand_logits(B, K, V, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(B, K + 1, V, generator=gen) * 3.0


# --------------------------------------------------------------------------
# chain mask
# --------------------------------------------------------------------------


def test_chain_verify_mask_structure():
    mask = chain_verify_mask(k_len=3, prefix_len=2)
    assert mask.shape == (3, 5)
    assert mask.dtype == torch.bool
    assert mask[:, :2].all()
    assert mask[0].tolist() == [True, True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True, False]
    assert mask[2].tolist() == [True, True, True, True, True]


# --------------------------------------------------------------------------
# greedy reference semantics
# --------------------------------------------------------------------------


def test_greedy_reference_all_accepted():
    draft = torch.tensor([[7, 3, 5]])
    logits = torch.full((1, 4, 10), -10.0)
    logits[0, 0, 7] = 5.0
    logits[0, 1, 3] = 5.0
    logits[0, 2, 5] = 5.0
    logits[0, 3, 9] = 5.0

    out, n = accept_reject_greedy_reference(draft, logits)
    assert out[0].tolist() == [7, 3, 5, 9]
    assert n[0].item() == 4


def test_greedy_reference_first_mismatch():
    draft = torch.tensor([[7, 3, 5]])
    logits = torch.full((1, 4, 10), -10.0)
    logits[0, 0, 7] = 5.0
    logits[0, 1, 4] = 5.0  # draft says 3, target says 4 -> mismatch at slot 1
    logits[0, 2, 5] = 5.0
    logits[0, 3, 9] = 5.0

    out, n = accept_reject_greedy_reference(draft, logits)
    assert out[0].tolist() == [7, 4, -1, -1]
    assert n[0].item() == 2


def test_greedy_reference_immediate_mismatch():
    draft = torch.tensor([[7, 3, 5]])
    logits = torch.full((1, 4, 10), -10.0)
    logits[0, 0, 1] = 5.0

    out, n = accept_reject_greedy_reference(draft, logits)
    assert out[0].tolist() == [1, -1, -1, -1]
    assert n[0].item() == 1


def test_greedy_reference_batch_independence():
    draft = torch.tensor([[7, 3, 5], [2, 2, 2]])
    logits = _rand_logits(2, 3, 10, seed=11)
    out, n = accept_reject_greedy_reference(draft, logits)

    for b in range(2):
        ref_out, ref_n = accept_reject_greedy_reference(
            draft[b : b + 1], logits[b : b + 1]
        )
        assert out[b].tolist() == ref_out[0].tolist()
        assert n[b].item() == ref_n[0].item()


# --------------------------------------------------------------------------
# sampling reference semantics
# --------------------------------------------------------------------------


def test_sampling_reference_accepts_when_draft_equals_target():
    gen = torch.Generator().manual_seed(3)
    V = 8
    K = 4
    p = torch.softmax(torch.randn(V, generator=gen), dim=-1)
    draft = torch.multinomial(p.expand(K, V), 1, generator=gen).reshape(1, K)
    draft_probs = p.expand(1, K, V).clone()
    logits = torch.log(p).expand(1, K + 1, V).clone()

    out, n = accept_reject_sampling_reference(draft, draft_probs, logits, generator=gen)
    assert n[0].item() == K + 1
    assert out[0, :K].tolist() == draft[0].tolist()


def test_sampling_reference_never_accepts_zero_target_mass():
    gen = torch.Generator().manual_seed(5)
    draft = torch.tensor([[0]])
    draft_probs = torch.ones(1, 1, 4) / 4.0
    logits = torch.full((1, 2, 4), -20.0)
    logits[0, 0, 1] = 20.0  # target puts all mass on token 1, draft proposed 0

    out, n = accept_reject_sampling_reference(draft, draft_probs, logits, generator=gen)
    assert out[0, 0].item() == 1  # correction from residual = target argmax
    assert n[0].item() == 1


def test_sampling_reference_distribution_invariant():
    """Output marginal must equal the target distribution exactly."""
    gen = torch.Generator().manual_seed(7)
    d = torch.tensor([0.1, 0.6, 0.3])
    p = torch.tensor([0.5, 0.25, 0.25])
    V = 3
    N = 20000

    draft = torch.multinomial(d.expand(N, V), 1, generator=gen)
    draft_probs = d.expand(N, 1, V).clone()
    logits = torch.log(p.expand(N, 2, V).clone())

    out, _ = accept_reject_sampling_reference(draft, draft_probs, logits, generator=gen)
    empirical = torch.bincount(out[:, 0] % V, minlength=V).double() / N
    tv = 0.5 * (empirical - p).abs().sum().item()
    assert tv < 0.02, f"TV={tv:.4f}; empirical={empirical.tolist()} target={p.tolist()}"


def test_sampling_reference_statistical_matches_target_on_logits():
    gen = torch.Generator().manual_seed(13)
    V, N = 6, 20000
    target = torch.softmax(torch.randn(V, generator=gen) * 2.0, dim=-1)
    draft = torch.softmax(torch.randn(V, generator=gen) * 2.0, dim=-1)

    picks = torch.multinomial(draft.expand(N, V), 1, generator=gen)
    out, _ = accept_reject_sampling_reference(
        picks,
        draft.expand(N, 1, V).clone(),
        torch.log(target.expand(N, 2, V).clone()),
        generator=gen,
    )
    empirical = torch.bincount(out[:, 0] % V, minlength=V).double() / N
    tv = 0.5 * (empirical - target).abs().sum().item()
    assert tv < 0.03, f"TV={tv:.4f}"


# --------------------------------------------------------------------------
# exercises: student implementations (xpess when correct)
# --------------------------------------------------------------------------


@pytest.mark.xfail(reason="练习：greedy accept/reject 未实现（实现正确后自动 XPASS）", strict=False)
def test_exercise_greedy_matches_reference():
    gen = torch.Generator().manual_seed(21)
    for trial in range(8):
        B, K, V = 3, 4, 32
        draft = torch.randint(0, V, (B, K), generator=gen)
        logits = _rand_logits(B, K, V, seed=100 + trial)
        out, n = accept_reject_greedy(draft, logits)
        ref_out, ref_n = accept_reject_greedy_reference(draft, logits)
        assert out.tolist() == ref_out.tolist()
        assert n.tolist() == ref_n.tolist()


@pytest.mark.xfail(reason="练习：rejection sampling 未实现（实现正确后自动 XPASS）", strict=False)
def test_exercise_sampling_matches_reference():
    gen = torch.Generator().manual_seed(23)
    for trial in range(8):
        B, K, V = 2, 3, 16
        draft_probs = torch.softmax(torch.randn(B, K, V, generator=gen), dim=-1)
        draft = torch.multinomial(
            draft_probs.reshape(B * K, V), 1, generator=gen
        ).reshape(B, K)
        logits = _rand_logits(B, K, V, seed=200 + trial)

        g1 = torch.Generator().manual_seed(31)
        g2 = torch.Generator().manual_seed(31)
        out, n = accept_reject_sampling(draft, draft_probs, logits, generator=g1)
        ref_out, ref_n = accept_reject_sampling_reference(
            draft, draft_probs, logits, generator=g2
        )
        assert out.tolist() == ref_out.tolist()
        assert n.tolist() == ref_n.tolist()


@pytest.mark.xfail(reason="练习：sampling 分布不变量（实现正确后自动 XPASS）", strict=False)
def test_exercise_sampling_distribution_invariant():
    gen = torch.Generator().manual_seed(29)
    d = torch.tensor([0.1, 0.6, 0.3])
    p = torch.tensor([0.5, 0.25, 0.25])
    V, N = 3, 20000

    draft = torch.multinomial(d.expand(N, V), 1, generator=gen)
    out, _ = accept_reject_sampling(
        draft,
        d.expand(N, 1, V).clone(),
        torch.log(p.expand(N, 1, V).clone()),
        generator=gen,
    )
    empirical = torch.bincount(out[:, 0] % V, minlength=V).double() / N
    tv = 0.5 * (empirical - p).abs().sum().item()
    assert tv < 0.02, f"TV={tv:.4f}"
