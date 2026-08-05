import torch

from einf.executors.torch.input import ModelInput
from einf.executors.torch.model_runner import DeterministicModelRunner
from einf.scheduler import ScheduledBatch, ScheduledRequest, WorkType


def make_request(
    request_id: str,
    input_token_ids: tuple[int, ...],
    *,
    block_table: tuple[int, ...],
    need_sample: bool,
    work_type: WorkType,
) -> ScheduledRequest:
    return ScheduledRequest(
        request_id=request_id,
        input_token_ids=input_token_ids,
        start_position=0,
        block_table=block_table,
        work_type=work_type,
        need_sample=need_sample,
    )


def test_deterministic_model_runner_returns_one_known_logit_row_per_token() -> None:
    batch = ScheduledBatch(
        step_id=1,
        requests=(
            make_request(
                "request",
                (0, 4, 9),
                block_table=(0, 1),
                need_sample=True,
                work_type=WorkType.PREFILL,
            ),
        ),
    )
    model_input = ModelInput.from_batch(
        batch,
        block_len=2,
        device=torch.device("cpu"),
    )
    runner = DeterministicModelRunner(vocab_size=10)

    output = runner.forward(model_input)

    assert output.logits.shape == (3, 10)
    assert output.logits.device == torch.device("cpu")
    assert torch.equal(
        output.logits.argmax(dim=1),
        torch.tensor([1, 5, 0], dtype=torch.long),
    )
