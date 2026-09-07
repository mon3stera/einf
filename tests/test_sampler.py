from __future__ import annotations

import torch

from einf.executors.torch.sampler import Sampler, SamplingBatch


def _batch(*, temperatures: list[float], vocab_size: int, device: torch.device) -> SamplingBatch:
    n = len(temperatures)
    if all(temperature == 0 for temperature in temperatures):
        return SamplingBatch(is_all_greedy=True)
    return SamplingBatch(
        temperatures=torch.tensor(temperatures, device=device, dtype=torch.float32),
        top_ks=torch.full((n,), vocab_size, device=device, dtype=torch.int32),
        top_ps=torch.ones(n, device=device, dtype=torch.float32),
        min_ps=torch.zeros(n, device=device, dtype=torch.float32),
        seeds=torch.arange(n, device=device, dtype=torch.int64),
        offsets=torch.zeros(n, device=device, dtype=torch.int64),
    )


def test_all_greedy_is_argmax() -> None:
    sampler = Sampler()
    logits = torch.tensor(
        [
            [0.1, 4.0, 0.2],
            [9.0, 1.0, 3.0],
        ]
    )
    batch = SamplingBatch(is_all_greedy=True)
    assert batch.temperatures is None
    output = sampler.sample(logits, batch)
    assert output.token_ids.tolist() == [1, 0]
    assert output.logprobs is None


def test_random_sampling_is_seeded() -> None:
    sampler = Sampler()
    logits = torch.tensor([[1.0, 1.0, 1.0], [0.0, 5.0, 0.0]])
    batch = _batch(temperatures=[1.0, 1.0], vocab_size=3, device=logits.device)
    first = sampler.sample(logits, batch).token_ids
    second = sampler.sample(logits, batch).token_ids
    assert first.tolist() == second.tolist()
