import pytest

from einf.execution_plan import BatchPlan, ScheduledBatch, ScheduledRequest, WorkType


def test_batch_plan_is_native_immutable_and_aliased() -> None:
    request = ScheduledRequest(
        request_id="req-1",
        input_token_ids=(1, 2),
        work_type=WorkType.PREFILL,
        start_position=0,
        block_table=(3,),
        need_sample=True,
    )
    plan = BatchPlan(step_id=4, requests=(request,))

    assert ScheduledBatch is BatchPlan
    assert isinstance(plan, ScheduledBatch)
    assert plan.requests[0].block_table == [3]

    with pytest.raises(AttributeError):
        plan.step_id = 5


def test_execution_plan_uses_singleton_work_kind() -> None:
    assert WorkType.PREFILL is not WorkType.DECODE
    assert ScheduledRequest(
        request_id="req-1",
        input_token_ids=(7,),
        work_type=WorkType.DECODE,
        start_position=8,
        block_table=(2, 9),
        need_sample=True,
    ).work_type is WorkType.DECODE
