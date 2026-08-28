import pytest
import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch.input import ModelInput
from einf.executors.torch.qwen import QwenConfig, QwenModelRunner, choose_num_splits
from einf.scheduler import ScheduledBatch, ScheduledRequest, WorkType


def make_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )


def make_cache(config: QwenConfig) -> TorchKVCacheStorage:
    return TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_hidden_layers,
            num_blocks=4,
            block_len=2,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        ),
        dtype=torch.float32,
        device="cpu",
        use_custom_ops=False,
    )


def make_input(
    token_ids: tuple[int, ...],
    *,
    start_position: int,
    block_table: tuple[int, ...],
    block_len: int = 2,
    device: torch.device = torch.device("cpu"),
) -> ModelInput:
    batch = ScheduledBatch(
        step_id=0,
        requests=(
            ScheduledRequest(
                request_id="request",
                input_token_ids=token_ids,
                work_type=(
                    WorkType.PREFILL
                    if len(token_ids) > 1
                    else WorkType.DECODE
                ),
                start_position=start_position,
                block_table=block_table,
                need_sample=True,
            ),
        ),
    )
    return ModelInput.from_batch(batch, block_len=block_len, device=device)


def test_qwen_cached_decode_matches_full_recompute() -> None:
    torch.manual_seed(0)
    config = make_config()
    cached_runner = QwenModelRunner(config, cache=make_cache(config)).eval()
    full_runner = QwenModelRunner(config, cache=make_cache(config)).eval()
    full_runner.load_state_dict(cached_runner.state_dict())

    cached_runner(make_input((1, 2, 3), start_position=0, block_table=(0, 1)))
    cached_decode = cached_runner(
        make_input((4,), start_position=3, block_table=(0, 1))
    ).logits[-1]
    full_decode = full_runner(
        make_input((1, 2, 3, 4), start_position=0, block_table=(0, 1))
    ).logits[-1]

    torch.testing.assert_close(cached_decode, full_decode, rtol=1e-5, atol=1e-5)


def test_qwen_mixed_packed_requests_match_separate_execution() -> None:
    torch.manual_seed(1)
    config = make_config()
    mixed_runner = QwenModelRunner(config, cache=make_cache(config)).eval()
    request_a_runner = QwenModelRunner(config, cache=make_cache(config)).eval()
    request_b_runner = QwenModelRunner(config, cache=make_cache(config)).eval()
    request_a_runner.load_state_dict(mixed_runner.state_dict())
    request_b_runner.load_state_dict(mixed_runner.state_dict())

    mixed_batch = ScheduledBatch(
        step_id=0,
        requests=(
            ScheduledRequest(
                request_id="a",
                input_token_ids=(1, 2),
                work_type=WorkType.PREFILL,
                start_position=0,
                block_table=(0,),
                need_sample=True,
            ),
            ScheduledRequest(
                request_id="b",
                input_token_ids=(3,),
                work_type=WorkType.PREFILL,
                start_position=0,
                block_table=(2,),
                need_sample=True,
            ),
        ),
    )
    mixed_input = ModelInput.from_batch(
        mixed_batch,
        block_len=2,
        device=torch.device("cpu"),
    )
    mixed_logits = mixed_runner(mixed_input).logits

    request_a_logits = request_a_runner(
        make_input((1, 2), start_position=0, block_table=(0,))
    ).logits
    request_b_logits = request_b_runner(
        make_input((3,), start_position=0, block_table=(2,))
    ).logits

    torch.testing.assert_close(
        mixed_logits,
        torch.cat((request_a_logits, request_b_logits)),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_qwen_flash_attention_matches_eager_attention() -> None:
    torch.manual_seed(2)
    config = QwenConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    geometry = KVCacheGeometry(
        num_layers=config.num_hidden_layers,
        num_blocks=4,
        block_len=16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    eager_runner = QwenModelRunner(
        config,
        cache=TorchKVCacheStorage(
            geometry,
            dtype=torch.float32,
            device="cuda",
            use_custom_ops=True,
        ),
    ).to(device="cuda", dtype=torch.float32).eval()
    flash_runner = QwenModelRunner(
        config,
        cache=TorchKVCacheStorage(
            geometry,
            dtype=torch.float32,
            device="cuda",
            use_custom_ops=True,
        ),
        use_flash_attention=True,
    ).to(device="cuda", dtype=torch.float32).eval()
    flash_runner.load_state_dict(eager_runner.state_dict())

    batch = ScheduledBatch(
        step_id=0,
        requests=(
            ScheduledRequest(
                request_id="request",
                input_token_ids=tuple(range(1, 20)),
                work_type=WorkType.PREFILL,
                start_position=0,
                block_table=(0, 1),
                need_sample=True,
            ),
        ),
    )
    model_input = ModelInput.from_batch(
        batch,
        block_len=16,
        device=torch.device("cuda"),
    )

    with torch.inference_mode():
        eager_logits = eager_runner(model_input).logits
        flash_logits = flash_runner(model_input).logits

    torch.testing.assert_close(flash_logits, eager_logits, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-4, 1e-4),
        (torch.bfloat16, 2e-2, 3e-1),
    ),
)
def test_qwen_split_kv_decode_matches_eager_attention(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    torch.manual_seed(3)
    config = QwenConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    geometry = KVCacheGeometry(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_len=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    eager_runner = QwenModelRunner(
        config,
        cache=TorchKVCacheStorage(
            geometry,
            dtype=dtype,
            device="cuda",
            use_custom_ops=True,
        ),
    ).to(device="cuda", dtype=dtype).eval()
    paged_runner = QwenModelRunner(
        config,
        cache=TorchKVCacheStorage(
            geometry,
            dtype=dtype,
            device="cuda",
            use_custom_ops=True,
        ),
        use_paged_decode_attention=True,
    ).to(device="cuda", dtype=dtype).eval()
    paged_runner.load_state_dict(eager_runner.state_dict())

    block_table = tuple(range(8))
    prefill_ids = tuple((index % 29) + 3 for index in range(31))
    prefill_input = make_input(
        prefill_ids,
        start_position=0,
        block_table=block_table,
        block_len=4,
        device=torch.device("cuda"),
    )
    decode_input = make_input(
        (7,),
        start_position=31,
        block_table=block_table,
        block_len=4,
        device=torch.device("cuda"),
    )

    with torch.inference_mode():
        eager_runner(prefill_input)
        paged_runner(prefill_input)
        eager_logits = eager_runner(decode_input).logits
        paged_logits = paged_runner(decode_input).logits

    torch.testing.assert_close(paged_logits, eager_logits, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("num_logical_blocks", "expected"),
    [
        (1, 1),
        (8, 8),
        (34, 16),
        (64, 16),
        (128, 32),
        (256, 64),
        (512, 64),
        (1024, 64),
    ],
)
def test_choose_num_splits_matches_measured_optima(
    num_logical_blocks: int,
    expected: int,
) -> None:
    """Pins the optima measured in benchmarks/micro-paged-decode-2026-08-23.md."""
    assert choose_num_splits(num_logical_blocks, 64) == expected


def test_choose_num_splits_respects_operator_precondition() -> None:
    for num_logical_blocks in range(1, 400):
        num_splits = choose_num_splits(num_logical_blocks, 64)
        assert 1 <= num_splits <= num_logical_blocks
        assert num_splits <= 64


def test_choose_num_splits_honours_max_splits() -> None:
    assert choose_num_splits(1024, 8) == 8
    assert choose_num_splits(1024, 1) == 1
