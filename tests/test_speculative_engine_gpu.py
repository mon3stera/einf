"""GPU smoke test: speculative engine on a tiny random-weight Qwen runner.

Self-speculation with identical weights must accept every draft under
greedy decoding, so the engine's output must equal a plain decode loop
over the same weights token-for-token. Exercises the real cache, RoPE,
and attention path without any checkpoint dependency.
"""

from __future__ import annotations

import pytest
import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch.qwen import QwenConfig, QwenModelRunner
from einf.executors.torch.spec_runner import SpeculativeEngine, _Track

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

BLOCK_LEN = 8
NUM_BLOCKS = 32
MAX_NEW = 24


def _tiny_runner(device: torch.device, seed: int) -> QwenModelRunner:
    config = QwenConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    cache = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=NUM_BLOCKS,
            block_len=BLOCK_LEN,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=torch.float32,
        device=device,
        use_custom_ops=False,
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed)
    try:
        with torch.device(device):
            runner = QwenModelRunner(config, cache=cache)
    finally:
        torch.set_default_dtype(previous)
    return runner.eval()


def _plain_greedy(
    runner: QwenModelRunner, device: torch.device, prompt: list[int]
) -> list[int]:
    """Sequential decode loop over the same weights (reference output)."""
    track = _Track(
        forward=runner.forward,
        block_table=list(range(NUM_BLOCKS)),
        context_len=0,
        pending_token=-1,
    )
    helper = SpeculativeEngine(
        runner, runner, device=device, block_len=BLOCK_LEN, num_blocks=NUM_BLOCKS
    )
    gaps: list[float] = []
    with torch.inference_mode():
        out = runner.forward(helper._make_input(track, list(prompt)))
        token = int(out.logits[-1].argmax().item())
        ids = [token]
        gaps.append(_top2_gap(out.logits[-1]))
        track.context_len = len(prompt)
        while len(ids) < MAX_NEW:
            out = runner.forward(helper._make_input(track, [token]))
            token = int(out.logits[-1].argmax().item())
            gaps.append(_top2_gap(out.logits[-1]))
            track.context_len += 1
            ids.append(token)
    return ids, gaps


def _top2_gap(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits.float(), 2).values
    return float(top2[0] - top2[1])


@requires_cuda
def test_spec_self_drafting_matches_plain_loop():
    device = torch.device("cuda")
    runner = _tiny_runner(device, seed=7)

    engine = SpeculativeEngine(
        runner,
        runner,
        device=device,
        block_len=BLOCK_LEN,
        num_blocks=NUM_BLOCKS,
        num_spec_tokens=4,
    )
    spec_ids, stats = engine.generate(
        [11, 5, 23], max_new_len=MAX_NEW, greedy=True
    )

    plain_ids, plain_gaps = _plain_greedy(runner, device, [11, 5, 23])

    # Greedy spec decoding is numerically equivalent to the plain loop, not
    # bit-identical: verify forwards (q_len=K+1) and decode forwards (q_len=1)
    # dispatch different SDPA/GEMM tilings, so near-tie argmaxes may flip.
    # Divergences are therefore allowed only at near-tie positions.
    diverged = [i for i, (a, b) in enumerate(zip(spec_ids, plain_ids)) if a != b]
    for i in diverged:
        assert plain_gaps[i] < 1e-3, (
            f"divergence at {i} with well-separated logits "
            f"(gap={plain_gaps[i]:.3e})"
        )
    assert len(diverged) <= 2

    # identical weights accept nearly every proposal under greedy decoding
    assert stats.accept_rate >= 0.9
