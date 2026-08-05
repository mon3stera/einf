import torch

from einf.executors.torch.input import ModelInput
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
