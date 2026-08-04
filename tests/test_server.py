from einf.cache.manager import KVCacheManager
from einf.cache.pool import BlockPool
from einf.execution import (
    ExecutionResult,
    Executor,
    FakeExecutor,
    RequestExecutionResult,
)
from einf.lib import LLMServer
from einf.request import CompletionReason, RequestSpec, RequestState
from einf.scheduler import (
    DecodeFirstPolicy,
    FCFSPolicy,
    Policy,
    ScheduledBatch,
    ScheduledRequest,
    Scheduler,
    WorkType,
)


def make_scheduler(
    *,
    num_blocks: int = 4,
    block_len: int = 4,
    max_batch_len: int = 8,
    max_prefill_chunk_len: int = 8,
    policy: Policy | None = None,
) -> tuple[Scheduler, KVCacheManager, BlockPool]:
    pool = BlockPool(num_blocks=num_blocks)
    cache_manager = KVCacheManager(
        pool,
        block_len=block_len,
    )
    scheduler = Scheduler(
        policy if policy is not None else FCFSPolicy(),
        cache_manager,
        max_batch_len=max_batch_len,
        max_prefill_chunk_len=max_prefill_chunk_len,
    )
    return scheduler, cache_manager, pool


def test_fake_executor_returns_one_result_per_scheduled_request() -> None:
    batch = ScheduledBatch(
        step_id=7,
        requests=(
            ScheduledRequest(
                request_id="req-1",
                input_token_ids=(10, 20, 30),
                work_type=WorkType.PREFILL,
                start_position=0,
                block_table=(0,),
                need_sample=False,
            ),
            ScheduledRequest(
                request_id="req-2",
                input_token_ids=(40,),
                work_type=WorkType.DECODE,
                start_position=3,
                block_table=(1,),
                need_sample=True,
            ),
        ),
    )

    result = FakeExecutor().execute(batch)

    assert result == ExecutionResult(
        step_id=7,
        request_results=(
            RequestExecutionResult(
                request_id="req-1",
                generated_token_ids=(),
                cached_len_delta=3,
                is_eos=False,
            ),
            RequestExecutionResult(
                request_id="req-2",
                generated_token_ids=(41,),
                cached_len_delta=1,
                is_eos=False,
            ),
        ),
    )


def test_schedule_builds_dynamic_batch_from_token_budget() -> None:
    scheduler, _, _ = make_scheduler(
        block_len=2,
        max_batch_len=2,
    )
    scheduler.submit(
        RequestSpec(
            request_id="req-1",
            prompt_token_ids=(10, 20),
            max_new_len=2,
        )
    )
    scheduler.submit(
        RequestSpec(
            request_id="req-2",
            prompt_token_ids=(30,),
            max_new_len=2,
        )
    )
    executor = FakeExecutor()

    first_batch = scheduler.schedule()
    assert first_batch is not None
    assert [request.request_id for request in first_batch.requests] == [
        "req-1"
    ]
    assert sum(
        len(request.input_token_ids)
        for request in first_batch.requests
    ) == 2

    scheduler.apply_result(executor.execute(first_batch))

    second_batch = scheduler.schedule()
    assert second_batch is not None
    assert [request.request_id for request in second_batch.requests] == [
        "req-1",
        "req-2",
    ]
    assert sum(
        len(request.input_token_ids)
        for request in second_batch.requests
    ) == 2


def test_schedule_grows_block_table_when_decode_crosses_block_boundary() -> None:
    scheduler, cache_manager, _ = make_scheduler(
        num_blocks=2,
        block_len=4,
        max_batch_len=4,
    )
    request_id = scheduler.submit(
        RequestSpec(
            request_id="req-1",
            prompt_token_ids=(10, 20, 30, 40),
            max_new_len=2,
        )
    )
    executor = FakeExecutor()

    prefill_batch = scheduler.schedule()
    assert prefill_batch is not None
    prefill = prefill_batch.requests[0]
    assert prefill.input_token_ids == (10, 20, 30, 40)
    assert prefill.start_position == 0
    assert prefill.block_table == (0,)

    scheduler.apply_result(executor.execute(prefill_batch))

    decode_batch = scheduler.schedule()
    assert decode_batch is not None
    decode = decode_batch.requests[0]
    assert decode.input_token_ids == (41,)
    assert decode.start_position == 4
    assert decode.block_table == (0, 1)
    assert cache_manager.translate(request_id, 4) == (1, 0)


def test_server_runs_multiple_requests_and_releases_cache() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=3,
        block_len=2,
        max_batch_len=3,
    )
    server = LLMServer(scheduler, FakeExecutor())
    first_id = scheduler.submit(
        RequestSpec(
            request_id="req-1",
            prompt_token_ids=(10, 20),
            max_new_len=2,
        )
    )
    second_id = scheduler.submit(
        RequestSpec(
            request_id="req-2",
            prompt_token_ids=(30,),
            max_new_len=2,
        )
    )

    server.run_until_idle()

    first = scheduler._get_request(first_id)
    second = scheduler._get_request(second_id)

    assert first.state is RequestState.FINISHED
    assert first.generated_token_ids == [21, 22]
    assert first.cached_len == 3
    assert first.completion_reason is CompletionReason.LENGTH

    assert second.state is RequestState.FINISHED
    assert second.generated_token_ids == [31, 32]
    assert second.cached_len == 2
    assert second.completion_reason is CompletionReason.LENGTH

    assert cache_manager.block_table(first_id) == ()
    assert cache_manager.block_table(second_id) == ()
    assert pool.free_len() == 3
    assert len(scheduler._policy) == 0


def test_eos_wins_when_eos_and_length_happen_together() -> None:
    scheduler, cache_manager, pool = make_scheduler(num_blocks=1)
    request_id = scheduler.submit(
        RequestSpec(
            request_id="req-1",
            prompt_token_ids=(10, 20, 30),
            max_new_len=1,
        )
    )
    batch = scheduler.schedule()
    assert batch is not None

    scheduler.apply_result(
        ExecutionResult(
            step_id=batch.step_id,
            request_results=(
                RequestExecutionResult(
                    request_id=request_id,
                    generated_token_ids=(31,),
                    cached_len_delta=3,
                    is_eos=True,
                ),
            ),
        )
    )

    request = scheduler._get_request(request_id)
    assert request.state is RequestState.FINISHED
    assert request.completion_reason is CompletionReason.EOS
    assert cache_manager.block_table(request_id) == ()
    assert pool.free_len() == 1
    assert len(scheduler._policy) == 0



def test_decode_first_prioritizes_decode_over_earlier_prefill() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=5,
        block_len=2,
        max_batch_len=3,
        max_prefill_chunk_len=2,
        policy=DecodeFirstPolicy(),
    )
    prefill_id = scheduler.submit(
        RequestSpec(
            request_id="prefill",
            prompt_token_ids=(10, 20, 30, 40, 50),
            max_new_len=1,
        )
    )
    decode_id = scheduler.submit(
        RequestSpec(
            request_id="decode",
            prompt_token_ids=(100,),
            max_new_len=3,
        )
    )
    executor = FakeExecutor()

    first_batch = scheduler.schedule()
    assert first_batch is not None
    assert [request.request_id for request in first_batch.requests] == [
        prefill_id,
        decode_id,
    ]
    assert [request.work_type for request in first_batch.requests] == [
        WorkType.PREFILL,
        WorkType.PREFILL,
    ]
    assert [request.need_sample for request in first_batch.requests] == [
        False,
        True,
    ]
    scheduler.apply_result(executor.execute(first_batch))

    second_batch = scheduler.schedule()
    assert second_batch is not None
    assert [request.request_id for request in second_batch.requests] == [
        decode_id,
        prefill_id,
    ]
    assert [request.work_type for request in second_batch.requests] == [
        WorkType.DECODE,
        WorkType.PREFILL,
    ]
    assert [request.input_token_ids for request in second_batch.requests] == [
        (101,),
        (30, 40),
    ]
    scheduler.apply_result(executor.execute(second_batch))

    third_batch = scheduler.schedule()
    assert third_batch is not None
    assert [request.request_id for request in third_batch.requests] == [
        decode_id,
        prefill_id,
    ]
    assert [request.input_token_ids for request in third_batch.requests] == [
        (102,),
        (50,),
    ]
    assert [request.need_sample for request in third_batch.requests] == [
        True,
        True,
    ]
    scheduler.apply_result(executor.execute(third_batch))

    decode = scheduler._get_request(decode_id)
    prefill = scheduler._get_request(prefill_id)
    assert decode.state is RequestState.FINISHED
    assert decode.generated_token_ids == [101, 102, 103]
    assert prefill.state is RequestState.FINISHED
    assert prefill.generated_token_ids == [51]
    assert cache_manager.block_table(decode_id) == ()
    assert cache_manager.block_table(prefill_id) == ()
    assert pool.free_len() == 5


def test_new_waiting_request_joins_existing_decode_next_iteration() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=4,
        block_len=2,
        max_batch_len=3,
        max_prefill_chunk_len=2,
        policy=DecodeFirstPolicy(),
    )
    decode_id = scheduler.submit(
        RequestSpec(
            request_id="decode",
            prompt_token_ids=(100,),
            max_new_len=3,
        )
    )
    executor = FakeExecutor()

    first_batch = scheduler.schedule()
    assert first_batch is not None
    assert [request.request_id for request in first_batch.requests] == [decode_id]
    scheduler.apply_result(executor.execute(first_batch))

    prefill_id = scheduler.submit(
        RequestSpec(
            request_id="late-prefill",
            prompt_token_ids=(10, 20, 30, 40),
            max_new_len=1,
        )
    )
    assert scheduler._get_request(prefill_id).state is RequestState.WAITING

    mixed_batch = scheduler.schedule()
    assert mixed_batch is not None
    assert [request.request_id for request in mixed_batch.requests] == [
        decode_id,
        prefill_id,
    ]
    assert [request.work_type for request in mixed_batch.requests] == [
        WorkType.DECODE,
        WorkType.PREFILL,
    ]
    assert [request.input_token_ids for request in mixed_batch.requests] == [
        (101,),
        (10, 20),
    ]
    assert mixed_batch.requests[1].need_sample is False
    scheduler.apply_result(executor.execute(mixed_batch))

    final_batch = scheduler.schedule()
    assert final_batch is not None
    assert [request.request_id for request in final_batch.requests] == [
        decode_id,
        prefill_id,
    ]
    assert [request.input_token_ids for request in final_batch.requests] == [
        (102,),
        (30, 40),
    ]
    assert final_batch.requests[1].need_sample is True
    scheduler.apply_result(executor.execute(final_batch))

    decode = scheduler._get_request(decode_id)
    prefill = scheduler._get_request(prefill_id)
    assert decode.state is RequestState.FINISHED
    assert decode.generated_token_ids == [101, 102, 103]
    assert prefill.state is RequestState.FINISHED
    assert prefill.generated_token_ids == [41]
    assert cache_manager.block_table(decode_id) == ()
    assert cache_manager.block_table(prefill_id) == ()
    assert pool.free_len() == 4


def test_chunked_prefill_samples_only_after_final_chunk() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=3,
        block_len=2,
        max_batch_len=2,
        max_prefill_chunk_len=2,
    )
    request_id = scheduler.submit(
        RequestSpec(
            request_id="req-1",
            prompt_token_ids=(10, 20, 30, 40, 50),
            max_new_len=1,
        )
    )
    executor = FakeExecutor()

    first_batch = scheduler.schedule()
    assert first_batch is not None
    first_chunk = first_batch.requests[0]
    assert first_chunk.work_type is WorkType.PREFILL
    assert first_chunk.input_token_ids == (10, 20)
    assert first_chunk.start_position == 0
    assert first_chunk.need_sample is False

    first_result = executor.execute(first_batch)
    assert first_result.request_results[0].generated_token_ids == ()
    scheduler.apply_result(first_result)
    request = scheduler._get_request(request_id)
    assert request.cached_len == 2
    assert request.generated_token_ids == []

    second_batch = scheduler.schedule()
    assert second_batch is not None
    second_chunk = second_batch.requests[0]
    assert second_chunk.input_token_ids == (30, 40)
    assert second_chunk.start_position == 2
    assert second_chunk.need_sample is False

    second_result = executor.execute(second_batch)
    assert second_result.request_results[0].generated_token_ids == ()
    scheduler.apply_result(second_result)
    assert request.cached_len == 4
    assert request.generated_token_ids == []

    final_batch = scheduler.schedule()
    assert final_batch is not None
    final_chunk = final_batch.requests[0]
    assert final_chunk.input_token_ids == (50,)
    assert final_chunk.start_position == 4
    assert final_chunk.need_sample is True

    final_result = executor.execute(final_batch)
    assert final_result.request_results[0].generated_token_ids == (51,)
    scheduler.apply_result(final_result)

    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [51]
    assert request.cached_len == 5
    assert request.completion_reason is CompletionReason.LENGTH
    assert cache_manager.block_table(request_id) == ()
    assert pool.free_len() == 3


def test_waiting_cancel_is_skipped_before_admission() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=1,
        block_len=2,
        max_batch_len=2,
    )
    cancelled_id = scheduler.submit(
        RequestSpec(
            request_id="cancelled",
            prompt_token_ids=(10, 20),
            max_new_len=1,
        )
    )
    live_id = scheduler.submit(
        RequestSpec(
            request_id="live",
            prompt_token_ids=(30, 40),
            max_new_len=1,
        )
    )

    scheduler.cancel(cancelled_id)
    batch = scheduler.schedule()

    assert batch is not None
    assert [request.request_id for request in batch.requests] == [live_id]
    assert scheduler._get_request(cancelled_id).state is RequestState.CANCELLED
    assert cache_manager.block_table(cancelled_id) == ()

    scheduler.apply_result(FakeExecutor().execute(batch))
    assert pool.free_len() == 1


def test_running_cancel_releases_cache_and_skips_stale_policy_entry() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=1,
        block_len=2,
        max_batch_len=2,
    )
    request_id = scheduler.submit(
        RequestSpec(
            request_id="req-1",
            prompt_token_ids=(10, 20),
            max_new_len=2,
        )
    )
    executor = FakeExecutor()

    first_batch = scheduler.schedule()
    assert first_batch is not None
    scheduler.apply_result(executor.execute(first_batch))
    assert cache_manager.block_table(request_id) == (0,)

    scheduler.cancel(request_id)

    assert scheduler._get_request(request_id).state is RequestState.CANCELLED
    assert cache_manager.block_table(request_id) == ()
    assert pool.free_len() == 1
    assert scheduler.schedule() is None



class FailingExecutor(Executor):
    def execute(self, batch: ScheduledBatch) -> ExecutionResult:
        raise RuntimeError("injected execution failure")


def test_executor_failure_fails_batch_releases_cache_and_continues() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=2,
        block_len=2,
        max_batch_len=4,
    )
    server = LLMServer(scheduler, FailingExecutor())
    request_ids = []

    for index in range(3):
        request_ids.append(
            scheduler.submit(
                RequestSpec(
                    request_id=f"req-{index}",
                    prompt_token_ids=(index * 10 + 1, index * 10 + 2),
                    max_new_len=2,
                )
            )
        )

    assert server.run_once() is True

    for request_id in request_ids[:2]:
        request = scheduler._get_request(request_id)
        assert request.state is RequestState.FAILED
        assert request.error == "RuntimeError: injected execution failure"
        assert cache_manager.block_table(request_id) == ()

    assert scheduler._get_request(request_ids[2]).state is RequestState.WAITING
    assert pool.free_len() == 2

    assert server.run_once() is True
    last_request = scheduler._get_request(request_ids[2])
    assert last_request.state is RequestState.FAILED
    assert last_request.error == "RuntimeError: injected execution failure"
    assert cache_manager.block_table(request_ids[2]) == ()
    assert pool.free_len() == 2
    assert server.run_once() is False


def test_single_request_cache_requirement_beyond_capacity_fails() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=1,
        block_len=2,
        max_batch_len=2,
        max_prefill_chunk_len=2,
        policy=FCFSPolicy(),
    )
    server = LLMServer(scheduler, FakeExecutor())
    request_id = scheduler.submit(
        RequestSpec(
            request_id="too-large",
            prompt_token_ids=(10, 20),
            max_new_len=2,
        )
    )

    assert server.run_once() is True
    assert server.run_once() is False

    request = scheduler._get_request(request_id)
    assert request.state is RequestState.FAILED
    assert request.error == "Insufficient memory to fulfill request too-large"
    assert cache_manager.block_table(request_id) == ()
    assert len(scheduler._policy) == 0
    assert list(scheduler._waiting) == []
    assert pool.free_len() == 1


def test_fcfs_preemption_keeps_older_request_running() -> None:
    scheduler, cache_manager, pool = make_scheduler(
        num_blocks=2,
        block_len=2,
        max_batch_len=4,
        max_prefill_chunk_len=2,
        policy=FCFSPolicy(),
    )
    executor = FakeExecutor()
    older_id = scheduler.submit(
        RequestSpec(
            request_id="older",
            prompt_token_ids=(10, 20),
            max_new_len=2,
        )
    )
    newer_id = scheduler.submit(
        RequestSpec(
            request_id="newer",
            prompt_token_ids=(30, 40),
            max_new_len=2,
        )
    )

    first_batch = scheduler.schedule()
    assert first_batch is not None
    assert [item.request_id for item in first_batch.requests] == [
        older_id,
        newer_id,
    ]
    scheduler.apply_result(executor.execute(first_batch))

    second_batch = scheduler.schedule()
    assert second_batch is not None
    assert [item.request_id for item in second_batch.requests] == [older_id]
    assert scheduler._get_request(newer_id).state is RequestState.WAITING
    assert cache_manager.block_table(newer_id) == ()
    assert [item.request_id for item in scheduler._waiting] == [newer_id]
    assert [item.request_id for item in scheduler._policy.candidates()] == [
        older_id
    ]

    scheduler.apply_result(executor.execute(second_batch))
    assert scheduler._get_request(older_id).state is RequestState.FINISHED
    assert cache_manager.block_table(older_id) == ()
    assert pool.free_len() == 2
