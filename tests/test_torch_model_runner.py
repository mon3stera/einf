import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.config import ModelConfig
from einf.executors.torch.input import ModelInput
from einf.executors.torch.model_runner import (
    DeterministicModelRunner,
    ReferenceModelRunner,
)
from einf.scheduler import ScheduledBatch, ScheduledRequest, WorkType


def make_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    *,
    block_table: tuple[int, ...],
    need_sample: bool = False,
    work_type: WorkType = WorkType.PREFILL,
    start_position: int = 0,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        start_position=start_position,
        block_table=block_table,
        work_type=work_type,
        need_sample=need_sample,
    )


def make_config(*, num_layers: int = 2) -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        num_layers=num_layers,
        hidden_size=16,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=4,
        eos_token_id=15,
    )


def make_storage(config: ModelConfig) -> TorchKVCacheStorage:
    return TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_layers,
            num_blocks=6,
            block_len=2,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        ),
        dtype=torch.float32,
        device="cpu",
    )


def make_model_input(
    *requests: ScheduledRequest,
) -> ModelInput:
    return ModelInput.from_batch(
        ScheduledBatch(step_id=1, requests=requests),
        block_len=2,
        device=torch.device("cpu"),
    )


def test_deterministic_model_runner_returns_one_known_logit_row_per_token() -> None:
    model_input = make_model_input(
        make_request(
            "request",
            (0, 4, 9),
            block_table=(0, 1),
            need_sample=True,
        )
    )
    runner = DeterministicModelRunner(vocab_size=10)

    output = runner.forward(model_input)

    assert output.logits.shape == (3, 10)
    assert output.logits.device == torch.device("cpu")
    assert torch.equal(
        output.logits.argmax(dim=1),
        torch.tensor([1, 5, 0], dtype=torch.long),
    )


def test_reference_model_runner_returns_packed_logits_and_writes_cache() -> None:
    torch.manual_seed(0)
    config = make_config()
    storage = make_storage(config)
    runner = ReferenceModelRunner(config, cache=storage).eval()
    model_input = make_model_input(
        make_request("a", (1, 2), block_table=(0,)),
        make_request("b", (3,), block_table=(2,)),
    )

    with torch.no_grad():
        output = runner(model_input)

    assert output.logits.shape == (3, config.vocab_size)
    assert torch.isfinite(output.logits).all()
    for layer_idx in range(config.num_layers):
        K, V = storage.read_slots(
            layer_idx,
            model_input.slot_mapping,
        )
        assert K.shape == (3, config.num_kv_heads, config.head_dim)
        assert V.shape == K.shape
        assert torch.isfinite(K).all()
        assert torch.isfinite(V).all()


def test_reference_model_runner_cached_decode_matches_full_recompute() -> None:
    torch.manual_seed(1)
    config = make_config()
    cached_runner = ReferenceModelRunner(
        config,
        cache=make_storage(config),
    ).eval()
    full_runner = ReferenceModelRunner(
        config,
        cache=make_storage(config),
    ).eval()
    full_runner.load_state_dict(cached_runner.state_dict())

    prefill_input = make_model_input(
        make_request("request", (1, 2, 3), block_table=(2, 0))
    )
    decode_input = make_model_input(
        make_request(
            "request",
            (4,),
            block_table=(2, 0),
            work_type=WorkType.DECODE,
            start_position=3,
        )
    )
    full_input = make_model_input(
        make_request(
            "request",
            (1, 2, 3, 4),
            block_table=(2, 0),
        )
    )

    with torch.no_grad():
        cached_runner(prefill_input)
        decode_output = cached_runner(decode_input)
        full_output = full_runner(full_input)

    torch.testing.assert_close(
        decode_output.logits[0],
        full_output.logits[-1],
        rtol=1e-5,
        atol=1e-6,
    )


def test_reference_model_runner_packed_requests_match_separate_execution() -> None:
    torch.manual_seed(2)
    config = make_config(num_layers=1)
    mixed_runner = ReferenceModelRunner(
        config,
        cache=make_storage(config),
    ).eval()
    request_a_runner = ReferenceModelRunner(
        config,
        cache=make_storage(config),
    ).eval()
    request_b_runner = ReferenceModelRunner(
        config,
        cache=make_storage(config),
    ).eval()
    request_a_runner.load_state_dict(mixed_runner.state_dict())
    request_b_runner.load_state_dict(mixed_runner.state_dict())

    request_a = make_request("a", (5, 6), block_table=(0,))
    request_b = make_request("b", (7,), block_table=(2,))

    with torch.no_grad():
        mixed_output = mixed_runner(
            make_model_input(request_a, request_b)
        )
        separate_output = torch.cat(
            (
                request_a_runner(make_model_input(request_a)).logits,
                request_b_runner(make_model_input(request_b)).logits,
            ),
            dim=0,
        )

    torch.testing.assert_close(
        mixed_output.logits,
        separate_output,
        rtol=1e-5,
        atol=1e-6,
    )
