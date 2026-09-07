import torch

from einf.executors.torch.flashinfer_attn import build_paged_kv_csr
from einf.executors.torch.input import ModelInput, ModelInputPool
from einf.scheduler import ScheduledBatch, ScheduledRequest, WorkType


def make_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    start_position: int,
    block_table: tuple[int, ...],
    *,
    work_type: WorkType = WorkType.PREFILL,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        start_position=start_position,
        block_table=block_table,
        work_type=work_type,
        need_sample=False,
    )


def assert_long_cpu_tensor(actual: torch.Tensor, expected: list[int]) -> None:
    assert actual.dtype is torch.long
    assert actual.device == torch.device("cpu")
    assert torch.equal(
        actual,
        torch.tensor(expected, dtype=torch.long),
    )


def test_from_batch_packs_requests_in_batch_order() -> None:
    batch = ScheduledBatch(
        step_id=7,
        requests=(
            make_request(
                "prefill",
                (10, 11),
                start_position=1,
                block_table=(3, 1),
            ),
            make_request(
                "decode",
                (20,),
                start_position=0,
                block_table=(4,),
                work_type=WorkType.DECODE,
            ),
            make_request(
                "later-decode",
                (30,),
                start_position=3,
                block_table=(2, 5),
                work_type=WorkType.DECODE,
            ),
        ),
    )

    model_input = ModelInput.from_batch(
        batch,
        block_len=2,
        device=torch.device("cpu"),
    )

    assert_long_cpu_tensor(
        model_input.input_token_ids,
        [10, 11, 20, 30],
    )
    assert_long_cpu_tensor(
        model_input.position,
        [1, 2, 0, 3],
    )
    assert_long_cpu_tensor(
        model_input.slot_mapping,
        [7, 2, 8, 11],
    )
    assert_long_cpu_tensor(
        model_input.query_start_loc,
        [0, 2, 3, 4],
    )
    assert_long_cpu_tensor(
        model_input.context_lens,
        [3, 1, 4],
    )
    assert torch.equal(
        model_input.block_tables,
        torch.tensor(
            [
                [3, 1],
                [4, -1],
                [2, 5],
            ],
            dtype=torch.long,
        ),
    )


def test_from_batch_maps_only_current_chunk_across_block_boundary() -> None:
    batch = ScheduledBatch(
        step_id=8,
        requests=(
            make_request(
                "chunk",
                (40, 41, 42),
                start_position=3,
                block_table=(5, 2),
            ),
        ),
    )

    model_input = ModelInput.from_batch(
        batch,
        block_len=4,
        device=torch.device("cpu"),
    )

    assert_long_cpu_tensor(
        model_input.input_token_ids,
        [40, 41, 42],
    )
    assert_long_cpu_tensor(
        model_input.position,
        [3, 4, 5],
    )
    assert_long_cpu_tensor(
        model_input.slot_mapping,
        [23, 8, 9],
    )
    assert_long_cpu_tensor(
        model_input.query_start_loc,
        [0, 3],
    )
    assert_long_cpu_tensor(
        model_input.context_lens,
        [6],
    )
    assert torch.equal(
        model_input.block_tables,
        torch.tensor([[5, 2]], dtype=torch.long),
    )
    assert model_input.block_tables.shape == (1, 2)
    assert model_input.slot_mapping.numel() == len(
        batch.requests[0].input_token_ids
    )


def test_query_start_loc_describes_each_packed_request_slice() -> None:
    batch = ScheduledBatch(
        step_id=9,
        requests=(
            make_request("a", (1, 2, 3), 0, (0, 1)),
            make_request("b", (4,), 0, (2,)),
            make_request("c", (5, 6), 0, (3,)),
        ),
    )

    model_input = ModelInput.from_batch(
        batch,
        block_len=2,
        device=torch.device("cpu"),
    )

    expected_request_tokens = (
        torch.tensor([1, 2, 3], dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([5, 6], dtype=torch.long),
    )

    for index, expected in enumerate(expected_request_tokens):
        start = model_input.query_start_loc[index].item()
        end = model_input.query_start_loc[index + 1].item()
        assert torch.equal(
            model_input.input_token_ids[start:end],
            expected,
        )

    assert_long_cpu_tensor(
        model_input.context_lens,
        [3, 1, 2],
    )
    assert torch.equal(
        model_input.block_tables,
        torch.tensor(
            [
                [0, 1],
                [2, -1],
                [3, -1],
            ],
            dtype=torch.long,
        ),
    )
    assert model_input.query_start_loc.numel() == len(batch.requests) + 1
    assert model_input.query_start_loc[-1].item() == model_input.input_token_ids.numel()


def test_host_metadata_matches_device_tensors() -> None:
    """Gate 6.1: the per-layer loop reads these instead of syncing on the device.

    The refactor is only correct if the host copies carry exactly what the device
    tensors carry, so pin that invariant here.
    """
    batch = ScheduledBatch(
        step_id=0,
        requests=(
            make_request("a", (1, 2, 3), 0, (0, 1)),
            make_request("b", (4,), 5, (2, 3, 4), work_type=WorkType.DECODE),
            make_request("c", (6, 7), 2, (5, 6)),
        ),
    )

    model_input = ModelInput.from_plan(batch, block_len=2, device=torch.device("cpu"))

    assert model_input.query_start_loc_host == tuple(
        model_input.query_start_loc.tolist()
    )
    assert model_input.context_lens_host == tuple(model_input.context_lens.tolist())
    assert len(model_input.query_start_loc_host) == len(batch.requests) + 1
    assert len(model_input.context_lens_host) == len(batch.requests)
    assert all(type(value) is int for value in model_input.query_start_loc_host)
    assert all(type(value) is int for value in model_input.context_lens_host)


POOL_BATCH = ScheduledBatch(
    step_id=0,
    requests=(
        make_request("a", (1, 2, 3), 0, (0, 1)),
        make_request("b", (4,), 5, (2, 3, 4), work_type=WorkType.DECODE),
        make_request("c", (6, 7), 2, (5, 6)),
    ),
)

FIELDS = (
    "input_token_ids",
    "position",
    "slot_mapping",
    "query_start_loc",
    "block_tables",
    "context_lens",
)


def test_model_input_pool_matches_from_plan() -> None:
    pool = ModelInputPool(torch.device("cpu"))
    pooled = pool.build(POOL_BATCH, block_len=2)
    reference = ModelInput.from_plan(POOL_BATCH, block_len=2, device=torch.device("cpu"))

    for field in FIELDS:
        assert torch.equal(getattr(pooled, field), getattr(reference, field)), field
    assert pooled.query_start_loc_host == reference.query_start_loc_host
    assert pooled.context_lens_host == reference.context_lens_host


def test_model_input_pool_builds_flashinfer_csr() -> None:
    pool = ModelInputPool(torch.device("cpu"))
    pooled = pool.build(POOL_BATCH, block_len=2)
    reference = ModelInput.from_plan(POOL_BATCH, block_len=2, device=torch.device("cpu"))

    qo_indptr, kv_indptr, kv_indices, last_page_len = pooled.flashinfer_csr
    reference_indptr, reference_indices, reference_last_page = build_paged_kv_csr(
        reference.block_tables,
        reference.context_lens,
        block_len=2,
    )

    assert qo_indptr.dtype is torch.int32
    assert qo_indptr.tolist() == reference.query_start_loc.tolist()
    assert kv_indptr.tolist() == reference_indptr.tolist()
    assert kv_indices.tolist() == reference_indices.tolist()
    assert last_page_len.tolist() == reference_last_page.tolist()


def test_model_input_pool_reuses_buffers_across_shapes() -> None:
    pool = ModelInputPool(torch.device("cpu"))
    pool.build(POOL_BATCH, block_len=2)
    assert pool._allocations == 1

    bigger = ScheduledBatch(
        step_id=1,
        requests=(
            make_request("x", tuple(range(10)), 7, (9, 8, 7, 6, 5)),
            make_request("y", (11,), 15, (5, 4, 3, 2), work_type=WorkType.DECODE),
        ),
    )
    pooled_second = pool.build(bigger, block_len=4)
    reference = ModelInput.from_plan(bigger, block_len=4, device=torch.device("cpu"))

    # The second shape fits the first allocation, so the buffers are reused.
    assert pool._allocations == 1
    for field in FIELDS:
        assert torch.equal(getattr(pooled_second, field), getattr(reference, field)), field

    largest = ScheduledBatch(
        step_id=2,
        requests=(make_request("z", tuple(range(80)), 0, tuple(range(39, -1, -1))),),
    )
    pooled_third = pool.build(largest, block_len=2)
    reference_third = ModelInput.from_plan(largest, block_len=2, device=torch.device("cpu"))

    # Growing past capacity replaces the buffers but keeps the values correct.
    assert pool._allocations == 2
    for field in FIELDS:
        assert torch.equal(getattr(pooled_third, field), getattr(reference_third, field)), field


def test_model_input_pool_rejects_short_block_table() -> None:
    pool = ModelInputPool(torch.device("cpu"))
    batch = ScheduledBatch(
        step_id=2,
        requests=(
            # context_len 8 needs 4 pages at block_len 2; the table has one.
            make_request("short", (1,), 7, (7,)),
        ),
    )

    try:
        pool.build(batch, block_len=2)
    except ValueError as error:
        assert "shorter than the paged context" in str(error)
    else:
        raise AssertionError("expected a ValueError for a short block table")
