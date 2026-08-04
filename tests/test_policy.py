from einf.scheduler import (
    DecodeFirstPolicy,
    FCFSPolicy,
    RequestBundle,
    WorkType,
)


def bundle(request_id: str, work_type: WorkType) -> RequestBundle:
    return RequestBundle(
        request_id=request_id,
        work_type=work_type,
    )


def test_fcfs_policy_candidates_are_persistent() -> None:
    policy = FCFSPolicy()
    first = bundle("first", WorkType.PREFILL)
    second = bundle("second", WorkType.PREFILL)

    policy.add(first)
    policy.add(second)

    assert policy.candidates() == (first, second)
    assert policy.candidates() == (first, second)
    assert len(policy) == 2


def test_fcfs_policy_updates_without_changing_order() -> None:
    policy = FCFSPolicy()
    first = bundle("first", WorkType.PREFILL)
    second = bundle("second", WorkType.DECODE)
    updated_first = bundle("first", WorkType.DECODE)

    policy.add(first)
    policy.add(second)

    assert policy.update(updated_first) is True
    assert policy.candidates() == (updated_first, second)
    assert policy.pop_victim() == second
    assert policy.remove("first") == updated_first
    assert policy.pop_victim() is None


def test_decode_first_policy_orders_persistent_candidates() -> None:
    policy = DecodeFirstPolicy()
    prefill_1 = bundle("prefill-1", WorkType.PREFILL)
    decode_1 = bundle("decode-1", WorkType.DECODE)
    prefill_2 = bundle("prefill-2", WorkType.PREFILL)
    decode_2 = bundle("decode-2", WorkType.DECODE)

    for candidate in (prefill_1, decode_1, prefill_2, decode_2):
        policy.add(candidate)

    expected = (decode_1, decode_2, prefill_1, prefill_2)
    assert policy.candidates() == expected
    assert policy.candidates() == expected
    assert len(policy) == 4


def test_decode_first_policy_updates_work_type_in_arrival_order() -> None:
    policy = DecodeFirstPolicy()
    first = bundle("first", WorkType.PREFILL)
    second = bundle("second", WorkType.DECODE)

    policy.add(first)
    policy.add(second)
    assert policy.candidates() == (second, first)

    updated_first = bundle("first", WorkType.DECODE)
    assert policy.update(updated_first) is True
    assert policy.candidates() == (updated_first, second)


def test_decode_first_policy_preempts_in_reverse_priority_order() -> None:
    policy = DecodeFirstPolicy()
    prefill_1 = bundle("prefill-1", WorkType.PREFILL)
    decode_1 = bundle("decode-1", WorkType.DECODE)
    prefill_2 = bundle("prefill-2", WorkType.PREFILL)
    decode_2 = bundle("decode-2", WorkType.DECODE)

    for candidate in (prefill_1, decode_1, prefill_2, decode_2):
        policy.add(candidate)

    assert policy.pop_victim() == prefill_2
    assert policy.pop_victim() == prefill_1
    assert policy.pop_victim() == decode_2
    assert policy.pop_victim() == decode_1
    assert policy.pop_victim() is None
