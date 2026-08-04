from copy import deepcopy

import pytest

from einf.request import (
    AdvanceResult,
    CompletionReason,
    InvalidStateTransition,
    Request,
    RequestInvariantViolation,
    RequestSpec,
    RequestState,
)


def create_request(*, max_new_len: int = 2) -> Request:
    spec = RequestSpec(
        request_id="req-1",
        prompt_token_ids=(10, 20, 30),
        max_new_len=max_new_len,
    )
    return Request.create(spec, arrival_index=7)


def create_running_request(*, max_new_len: int = 2) -> Request:
    request = create_request(max_new_len=max_new_len)
    request.admit()
    return request


def create_terminal_request(state: RequestState) -> Request:
    request = create_request(max_new_len=4)

    if state is RequestState.CANCELLED:
        request.cancel()
        return request

    request.admit()
    if state is RequestState.FINISHED:
        request.advance(
            AdvanceResult(
                generated_token_ids=[40],
                cached_len_delta=len(request.prompt_token_ids),
                completion_reason=CompletionReason.EOS,
            )
        )
    elif state is RequestState.FAILED:
        request.fail("executor failed")
    else:
        raise ValueError(f"unsupported terminal state: {state}")

    return request


def test_create_request_starts_waiting() -> None:
    request = create_request()

    assert request.request_id == "req-1"
    assert request.prompt_token_ids == (10, 20, 30)
    assert request.arrival_index == 7
    assert request.max_new_len == 2
    assert request.generated_token_ids == []
    assert request.state is RequestState.WAITING
    assert request.cached_len == 0
    assert request.completion_reason is None
    assert request.error is None


def test_admit_transitions_waiting_request_to_running() -> None:
    request = create_request()

    request.admit()

    assert request.state is RequestState.RUNNING
    assert request.generated_token_ids == []
    assert request.cached_len == 0
    assert request.completion_reason is None
    assert request.error is None


def test_prefill_advance_updates_progress_and_keeps_request_running() -> None:
    request = create_running_request()

    request.advance(
        AdvanceResult(
            generated_token_ids=[40],
            cached_len_delta=len(request.prompt_token_ids),
            completion_reason=None,
        )
    )

    assert request.state is RequestState.RUNNING
    assert request.generated_token_ids == [40]
    assert request.cached_len == 3
    assert request.completion_reason is None
    assert request.error is None


def test_decode_advance_can_finish_request_for_length() -> None:
    request = create_running_request(max_new_len=2)
    request.advance(
        AdvanceResult(
            generated_token_ids=[40],
            cached_len_delta=len(request.prompt_token_ids),
            completion_reason=None,
        )
    )

    request.advance(
        AdvanceResult(
            generated_token_ids=[41],
            cached_len_delta=1,
            completion_reason=CompletionReason.LENGTH,
        )
    )

    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [40, 41]
    assert request.cached_len == 4
    assert request.completion_reason is CompletionReason.LENGTH
    assert request.error is None


def test_advance_can_finish_request_for_eos() -> None:
    request = create_running_request(max_new_len=4)

    request.advance(
        AdvanceResult(
            generated_token_ids=[40],
            cached_len_delta=len(request.prompt_token_ids),
            completion_reason=CompletionReason.EOS,
        )
    )

    assert request.state is RequestState.FINISHED
    assert request.generated_token_ids == [40]
    assert request.cached_len == 3
    assert request.completion_reason is CompletionReason.EOS
    assert request.error is None


def test_running_request_can_fail() -> None:
    request = create_running_request()

    request.fail("executor failed")

    assert request.state is RequestState.FAILED
    assert request.error == "executor failed"
    assert request.completion_reason is None


@pytest.mark.parametrize("running", [False, True])
def test_waiting_or_running_request_can_be_cancelled(running: bool) -> None:
    request = create_running_request() if running else create_request()

    request.cancel()

    assert request.state is RequestState.CANCELLED
    assert request.completion_reason is None
    assert request.error is None


def test_cancelling_cancelled_request_is_idempotent() -> None:
    request = create_request()
    request.cancel()
    before = deepcopy(request)

    request.cancel()

    assert request == before


@pytest.mark.parametrize(
    "terminal_state",
    [RequestState.FINISHED, RequestState.FAILED],
)
def test_finished_or_failed_request_rejects_cancel(
    terminal_state: RequestState,
) -> None:
    request = create_terminal_request(terminal_state)
    before = deepcopy(request)

    with pytest.raises(InvalidStateTransition):
        request.cancel()

    assert request == before


@pytest.mark.parametrize(
    "terminal_state",
    [
        RequestState.FINISHED,
        RequestState.CANCELLED,
        RequestState.FAILED,
    ],
)
def test_terminal_request_rejects_late_advance_without_mutation(
    terminal_state: RequestState,
) -> None:
    request = create_terminal_request(terminal_state)
    before = deepcopy(request)

    with pytest.raises(InvalidStateTransition):
        request.advance(
            AdvanceResult(
                generated_token_ids=[41],
                cached_len_delta=1,
                completion_reason=None,
            )
        )

    assert request == before


def test_waiting_request_rejects_failure_without_mutation() -> None:
    request = create_request()
    before = deepcopy(request)

    with pytest.raises(InvalidStateTransition):
        request.fail("executor failed")

    assert request == before


def test_running_request_rejects_second_admission_without_mutation() -> None:
    request = create_running_request()
    before = deepcopy(request)

    with pytest.raises(InvalidStateTransition):
        request.admit()

    assert request == before


def test_invalid_advance_does_not_partially_update_request() -> None:
    request = create_running_request(max_new_len=1)
    before = deepcopy(request)

    with pytest.raises(RequestInvariantViolation):
        request.advance(
            AdvanceResult(
                generated_token_ids=[40, 41],
                cached_len_delta=3,
                completion_reason=CompletionReason.LENGTH,
            )
        )

    assert request == before
