from einf.execution import ExecutionResult, Executor, RequestExecutionResult

from einf.executors.torch.input import ModelInput
from einf.scheduler import ScheduledBatch
import torch

class TorchExecutor(Executor):
    def __init__(self, *, model_runner, block_len: int, eos_token_id: int, device: torch.device):
        self._model_runner = model_runner
        self._block_len = block_len
        self._eos_token_id = eos_token_id
        self._device = device

    def execute(self, batch: ScheduledBatch) -> ExecutionResult:
        input = ModelInput.from_batch(batch, block_len=self._block_len, device=self._device)

        output = self._model_runner.forward(input)

        results = []

        for index, request in enumerate(batch.requests):
            generated_token_ids = ()
            is_eos = False

            if request.need_sample:
                last_query_index = input.query_start_loc[index + 1].item() - 1
                token_id = output.logits[last_query_index].argmax().item()
                generated_token_ids = (token_id, )
                is_eos = token_id == self._eos_token_id

            results.append(
                RequestExecutionResult(
                    request_id=request.request_id,
                    generated_token_ids=generated_token_ids,
                    cached_len_delta=len(request.input_token_ids),
                    is_eos=is_eos
                )
            )

        return ExecutionResult(
            step_id=batch.step_id,
            request_results=tuple(results),
        )
