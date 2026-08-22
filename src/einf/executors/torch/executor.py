from einf.execution import ExecutionResult, Executor, RequestExecutionResult

from einf.executors.torch.input import ModelInput
from einf.execution_plan import BatchPlan
from einf.executors.torch.sampler import Sampler, SamplingBatch

import torch

class TorchExecutor(Executor):
    def __init__(self, *, model_runner, block_len: int, eos_token_id: int, device: torch.device):
        self._model_runner = model_runner
        self._block_len = block_len
        self._eos_token_id = eos_token_id
        self._device = device
        self._sampler = Sampler()

    def execute(self, batch: BatchPlan) -> ExecutionResult:
        input = ModelInput.from_batch(batch, block_len=self._block_len, device=self._device)

        output = self._model_runner.forward(input)

        results = []

        sampling_requests = [
            request
            for request in batch.requests
            if request.sampling_plan is not None
        ]

        if sampling_requests:
            sampling_request_indices = [
                index
                for index, request in enumerate(batch.requests)
                if request.sampling_plan is not None
            ]

            vocab_size = output.logits[0].shape[0]

            device = output.logits.device

            temperatures = torch.tensor(
                [r.sampling_plan.params.temperature for r in sampling_requests],
                device=device,
                dtype=torch.float32,
            )

            top_ks = torch.tensor(
                [
                    r.sampling_plan.params.top_k
                    if r.sampling_plan.params.top_k is not None
                    else vocab_size
                    for r in sampling_requests
                ],
                device=device,
                dtype=torch.int32,
            )

            top_ps = torch.tensor(
                [r.sampling_plan.params.top_p for r in sampling_requests],
                device=device,
                dtype=torch.float32,
            )

            min_ps = torch.tensor(
                [r.sampling_plan.params.min_p for r in sampling_requests],
                device=device,
                dtype=torch.float32,
            )

            seeds = torch.tensor(
                [r.sampling_plan.params.seed for r in sampling_requests],
                device=device,
                dtype=torch.int64,
            )

            offsets = torch.tensor(
                [r.sampling_plan.sample_index for r in sampling_requests],
                device=device,
                dtype=torch.int64,
            )

            sample_batch = SamplingBatch(
                temperatures=temperatures,
                top_ks=top_ks,
                top_ps=top_ps,
                min_ps=min_ps,
                seeds=seeds,
                offsets=offsets,
            )

            logits_indices = input.query_start_loc[1:] - 1
            sample_logits_indices = logits_indices[sampling_request_indices]
            logits = output.logits[sample_logits_indices]

            output = self._sampler.sample(logits, sample_batch)

            sampled_tokens = dict(
                zip(
                    sampling_request_indices,
                    output.token_ids.cpu().tolist(),
                    strict=True
                )
            )
        else:
            sampled_tokens = {}

        for index, request in enumerate(batch.requests):
            token_id = sampled_tokens.get(index)

            if token_id is None:
                generated_token_ids = ()
                is_eos = False
            else:
                generated_token_ids = (token_id, )
                stop_ids = set(request.sampling_plan.params.stop_token_ids)
                is_eos = (token_id == self._eos_token_id or token_id in stop_ids)

            results.append(
                RequestExecutionResult(
                    request_id=request.request_id,
                    generated_token_ids=generated_token_ids,
                    cached_len_delta=len(request.input_token_ids),
                    is_eos=is_eos,
                )
            )

        return ExecutionResult(
            step_id=batch.step_id,
            request_results=tuple(results),
        )
