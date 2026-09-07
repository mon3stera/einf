"""CPU tests for the speculative engine's bookkeeping.

The deterministic runner maps each input token to a one-hot next-token
logit, so plain greedy decoding produces (t+1, t+2, ...) — a closed
form the speculative engine must reproduce exactly, including cache
context advancement across accept/reject boundaries.
"""

from __future__ import annotations

import torch

from einf.executors.torch.input import ModelInput
from einf.executors.torch.model_runner import DeterministicModelRunner
from einf.executors.torch.spec_runner import SpeculativeEngine

VOCAB = 64


def _deterministic() -> DeterministicModelRunner:
    return DeterministicModelRunner(vocab_size=VOCAB)


def _empty_input(tokens: list[int]) -> ModelInput:
    n = len(tokens)
    device = torch.device("cpu")
    return ModelInput(
        input_token_ids=torch.tensor(tokens, device=device, dtype=torch.long),
        position=torch.zeros(n, dtype=torch.long),
        slot_mapping=torch.zeros(n, dtype=torch.long),
        query_start_loc=torch.tensor([0, n], dtype=torch.long),
        block_tables=torch.zeros((1, 1), dtype=torch.long),
        context_lens=torch.tensor([n], dtype=torch.long),
        query_start_loc_host=(0, n),
        context_lens_host=(n,),
    )


def test_spec_greedy_matches_plain_loop():
    runner = _deterministic()
    prompt = [5, 9, 2]
    max_new = 12

    engine = SpeculativeEngine(
        runner,
        runner,
        device=torch.device("cpu"),
        num_spec_tokens=4,
    )
    spec_ids, stats = engine.generate(
        prompt, max_new_len=max_new, greedy=True
    )

    # closed form: successive successors of the last prompt token
    expected = [(prompt[-1] + 1 + i) % VOCAB for i in range(max_new)]
    assert spec_ids == expected

    # self-drafting with identical weights accepts everything
    assert stats.proposed == stats.accepted
    assert stats.committed == stats.accepted + stats.steps
    # every step commits the full K+1
    per_step_full = stats.accepted + stats.steps
    assert per_step_full >= max_new - 4  # final step may truncate at max_new_len


def test_spec_greedy_long_run_consistency():
    runner = _deterministic()
    engine = SpeculativeEngine(
        runner,
        runner,
        device=torch.device("cpu"),
        block_len=4,
        num_blocks=16,
        num_spec_tokens=3,
    )
    spec_ids, _ = engine.generate(
        [0, 1], max_new_len=40, greedy=True
    )
    expected = [(1 + 1 + i) % VOCAB for i in range(40)]
    assert spec_ids == expected


def test_spec_stops_on_eos():
    runner = _deterministic()
    prompt = [10, 20]
    eos = (20 + 3) % VOCAB  # third generated token

    engine = SpeculativeEngine(
        runner,
        runner,
        device=torch.device("cpu"),
        num_spec_tokens=4,
    )
    spec_ids, stats = engine.generate(
        prompt, max_new_len=32, eos_token_id=eos, greedy=True
    )

    assert spec_ids == [(20 + 1 + i) % VOCAB for i in range(3)]
    assert spec_ids[-1] == eos
    assert len(spec_ids) < 32


def test_spec_sampling_matches_greedy_for_one_hot():
    """One-hot logits make multinomial degenerate to argmax, so the
    sampling path must reproduce the greedy sequence exactly."""
    runner = _deterministic()
    prompt = [3, 1, 4]
    gen = torch.Generator().manual_seed(0)

    greedy_engine = SpeculativeEngine(
        runner, runner, device=torch.device("cpu"), num_spec_tokens=3
    )
    greedy_ids, _ = greedy_engine.generate(prompt, max_new_len=10, greedy=True)

    sample_engine = SpeculativeEngine(
        runner, runner, device=torch.device("cpu"), num_spec_tokens=3
    )
    sample_ids, _ = sample_engine.generate(
        prompt, max_new_len=10, greedy=False, generator=gen
    )
    assert sample_ids == greedy_ids
