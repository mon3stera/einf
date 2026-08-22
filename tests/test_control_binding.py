import pytest

from einf import (
    BatchPlan,
    CompletionReason,
    ExecutionResult,
    RequestExecutionResult,
    RequestSpec,
    RequestState,
    SamplingParams,
    SamplingPlan,
    ScheduledBatch,
    ScheduledRequest,
    Scheduler,
    WorkType,
)
from einf.executors import FakeExecutor


def make_scheduler(*, policy: str = "fcfs") -> Scheduler:
    return Scheduler(
        policy=policy,
        num_blocks=4,
        block_len=2,
        max_batch_len=4,
        max_prefill_chunk_len=2,
    )


def test_native_value_contract_and_aliases() -> None:
    sampling_plan = SamplingPlan(SamplingParams(), 0)
    request = ScheduledRequest(
        request_id="req-1",
        input_token_ids=(1, 2),
        work_type=WorkType.PREFILL,
        start_position=0,
        block_table=(3,),
        need_sample=True,
        sampling_plan=sampling_plan,
    )
    plan = BatchPlan(step_id=7, requests=(request,))

    assert ScheduledBatch is BatchPlan
    assert isinstance(plan, ScheduledBatch)
    assert plan.requests[0] == request
    assert request.input_token_ids == [1, 2]
    assert request.block_table == [3]
    assert request.work_type is WorkType.PREFILL
    assert request.sampling_plan == sampling_plan
    assert WorkType.PREFILL is not WorkType.DECODE
    assert RequestState.FINISHED is RequestState.FINISHED
    assert CompletionReason.EOS is CompletionReason.EOS

    with pytest.raises(AttributeError):
        plan.step_id = 8


def test_sampling_params_defaults_validation_and_request_round_trip() -> None:
    greedy = SamplingParams()
    assert greedy.is_greedy
    assert greedy.temperature == 0.0
    assert greedy.top_k is None
    assert greedy.top_p == 1.0
    assert greedy.min_p == 0.0

    random = SamplingParams(
        0.75,
        top_k=50,
        top_p=0.95,
        min_p=0.05,
        seed=123,
        stop_token_ids=(2, 3),
        num_logprobs=4,
    )
    assert not random.is_greedy
    assert random.temperature == pytest.approx(0.75)
    assert random.top_k == 50
    assert random.stop_token_ids == [2, 3]

    default_spec = RequestSpec("default", (1,), 1)
    assert default_spec.sampling_params == greedy
    random_spec = RequestSpec("random", (1,), 1, random)
    assert random_spec.sampling_params == random

    with pytest.raises(ValueError, match="temperature"):
        SamplingParams(float("nan"))
    with pytest.raises(ValueError, match="top_k"):
        SamplingParams(1.0, top_k=0)
    with pytest.raises(ValueError, match="positive temperature"):
        SamplingParams(0.0, top_p=0.9)
    assert SamplingParams(1.0, seed=2**63 - 1).seed == 2**63 - 1
    with pytest.raises(ValueError, match="seed"):
        SamplingParams(1.0, seed=2**63)
    with pytest.raises(ValueError, match="sampling_plan"):
        ScheduledRequest(
            "bad",
            (1,),
            WorkType.PREFILL,
            0,
            (0,),
            False,
            SamplingPlan(greedy, 0),
        )


def test_native_scheduler_round_trip_and_request_view() -> None:
    scheduler = make_scheduler()
    request_id = scheduler.submit(
        RequestSpec(
            request_id="req-1",
            prompt_token_ids=(10, 20, 30),
            max_new_len=2,
        )
    )
    assert request_id == "req-1"
    assert scheduler.request(request_id).state is RequestState.WAITING
    assert scheduler.request(request_id).sample_index == 0

    first_batch = scheduler.schedule()
    assert first_batch is not None
    assert first_batch.requests[0].sampling_plan is None
    executor = FakeExecutor()
    scheduler.apply_result(executor.execute(first_batch))
    assert scheduler.request(request_id).sample_index == 0

    second_batch = scheduler.schedule()
    assert second_batch is not None
    assert second_batch.requests[0].sampling_plan is not None
    assert second_batch.requests[0].sampling_plan.sample_index == 0
    scheduler.apply_result(executor.execute(second_batch))
    assert scheduler.request(request_id).sample_index == 1

    while (batch := scheduler.schedule()) is not None:
        scheduler.apply_result(executor.execute(batch))

    request = scheduler.request(request_id)
    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [31, 32]
    assert request.completion_reason is CompletionReason.LENGTH
    assert scheduler.block_table(request_id) == []
    assert scheduler.free_blocks() == 4


def test_native_scheduler_rejects_protocol_errors() -> None:
    scheduler = make_scheduler()
    scheduler.submit(RequestSpec("req-1", (10,), 1))
    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit(RequestSpec("req-1", (20,), 1))

    batch = scheduler.schedule()
    assert batch is not None
    with pytest.raises(ValueError, match="still outstanding"):
        scheduler.schedule()
    with pytest.raises(RuntimeError, match="in flight"):
        scheduler.cancel("req-1")

    bad = ExecutionResult(
        step_id=batch.step_id,
        request_results=(
            RequestExecutionResult(
                request_id="req-1",
                generated_token_ids=(11,),
                cached_len_delta=0,
                is_eos=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="cached length delta mismatch"):
        scheduler.apply_result(bad)


def test_native_scheduler_validates_constructor_and_request_spec() -> None:
    with pytest.raises(ValueError, match="unknown scheduling policy"):
        Scheduler(
            policy="random",
            num_blocks=1,
            block_len=1,
            max_batch_len=1,
            max_prefill_chunk_len=1,
        )
    with pytest.raises(ValueError, match="prompt"):
        make_scheduler().submit(RequestSpec("empty", (), 1))
    with pytest.raises(ValueError, match="max_new_len"):
        make_scheduler().submit(RequestSpec("zero", (1,), 0))
