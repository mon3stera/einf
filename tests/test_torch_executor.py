import torch

from einf import SamplingParams, SamplingPlan
from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.config import ModelConfig
from einf.executors.torch import (
    DeterministicModelRunner,
    ReferenceModelRunner,
    TorchExecutor,
)
from einf.scheduler import ScheduledBatch, ScheduledRequest, WorkType


def make_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    *,
    block_table: tuple[int, ...],
    need_sample: bool,
    work_type: WorkType,
) -> ScheduledRequest:
    sampling_plan = (
        SamplingPlan(SamplingParams(), 0)
        if need_sample
        else None
    )
    return ScheduledRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        start_position=0,
        block_table=block_table,
        work_type=work_type,
        need_sample=need_sample,
        sampling_plan=sampling_plan,
    )


def test_torch_executor_handles_mixed_sampling_and_eos() -> None:
    batch = ScheduledBatch(
        step_id=12,
        requests=(
            make_request(
                "intermediate-prefill",
                (3, 4),
                block_table=(0,),
                need_sample=False,
                work_type=WorkType.PREFILL,
            ),
            make_request(
                "final-prefill",
                (8, 9),
                block_table=(1,),
                need_sample=True,
                work_type=WorkType.PREFILL,
            ),
            make_request(
                "decode",
                (5,),
                block_table=(2,),
                need_sample=True,
                work_type=WorkType.DECODE,
            ),
        ),
    )
    executor = TorchExecutor(
        model_runner=DeterministicModelRunner(vocab_size=16),
        block_len=2,
        eos_token_id=10,
        device=torch.device("cpu"),
    )

    result = executor.execute(batch)

    assert result.step_id == 12
    assert result.request_results[0].request_id == "intermediate-prefill"
    assert result.request_results[0].generated_token_ids == []
    assert result.request_results[0].cached_len_delta == 2
    assert result.request_results[0].is_eos is False

    assert result.request_results[1].request_id == "final-prefill"
    assert result.request_results[1].generated_token_ids == [10]
    assert result.request_results[1].cached_len_delta == 2
    assert result.request_results[1].is_eos is True

    assert result.request_results[2].request_id == "decode"
    assert result.request_results[2].generated_token_ids == [6]
    assert result.request_results[2].cached_len_delta == 1
    assert result.request_results[2].is_eos is False


def test_torch_executor_runs_reference_model_runner() -> None:
    torch.manual_seed(3)
    config = ModelConfig(
        vocab_size=16,
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=4,
        eos_token_id=15,
    )
    storage = TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=config.num_layers,
            num_blocks=2,
            block_len=2,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
        ),
        dtype=torch.float32,
        device="cpu",
    )
    runner = ReferenceModelRunner(config, cache=storage).eval()
    executor = TorchExecutor(
        model_runner=runner,
        block_len=2,
        eos_token_id=config.eos_token_id,
        device=torch.device("cpu"),
    )
    batch = ScheduledBatch(
        step_id=13,
        requests=(
            make_request(
                "reference",
                (1, 2),
                block_table=(0,),
                need_sample=True,
                work_type=WorkType.PREFILL,
            ),
        ),
    )

    result = executor.execute(batch)

    request_result = result.request_results[0]
    assert request_result.request_id == "reference"
    assert request_result.cached_len_delta == 2
    assert len(request_result.generated_token_ids) == 1
    token_id = request_result.generated_token_ids[0]
    assert 0 <= token_id < config.vocab_size
    assert request_result.is_eos is (
        token_id == config.eos_token_id
    )
