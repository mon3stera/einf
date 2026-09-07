from einf.execution import ExecutionResult, Executor, RequestExecutionResult

from einf.executors.torch.decode_graph import is_decode_only_plan
from einf.executors.torch.input import ModelInputPool
from einf.execution_plan import BatchPlan
from einf.executors.torch.sampler import Sampler, SamplingBatch

import torch
from torch import Tensor

_MIN_SAMPLE_CAP = 64


class TorchExecutor(Executor):
    def __init__(self, *, model_runner, block_len: int, eos_token_id: int, device: torch.device):
        self._model_runner = model_runner
        self._block_len = block_len
        self._eos_token_id = eos_token_id
        self._device = device
        self._sampler = Sampler()
        self._input_pool = ModelInputPool(device)
        self._host_tokens: Tensor | None = None
        self._host_f32: Tensor | None = None
        self._host_i64: Tensor | None = None
        self._gpu_f32: Tensor | None = None
        self._gpu_i64: Tensor | None = None

    def execute(self, batch: BatchPlan) -> ExecutionResult:
        output = None
        model_input = None
        try_plan = getattr(self._model_runner, "try_decode_cuda_graph_plan", None)
        if try_plan is not None:
            output = try_plan(batch)
        if output is None:
            model_input = self._input_pool.build(batch, block_len=self._block_len)
            # Host-side gate: the pool already carries the plan's lengths, so
            # the device-wide ``is_decode_only`` re-check (a torch.equal sync)
            # is only paid by batches that can actually use a decode graph.
            if is_decode_only_plan(batch):
                try_graph = getattr(self._model_runner, "try_decode_cuda_graph", None)
                if try_graph is not None:
                    output = try_graph(model_input)
            if output is None:
                output = self._model_runner.forward(model_input)

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

            sample_count = len(sampling_requests)
            self._ensure_token_buffer(sample_count)
            is_all_greedy = all(
                request.sampling_plan.params.temperature == 0
                for request in sampling_requests
            )
            if is_all_greedy:
                sample_batch = SamplingBatch(is_all_greedy=True)
            else:
                self._ensure_param_buffers(sample_count)
                sample_batch = self._pack_sampling_params(
                    sampling_requests,
                    vocab_size=output.logits[0].shape[0],
                )

            if model_input is None:
                logits = output.logits[sampling_request_indices]
            else:
                logits_indices = model_input.query_start_loc[1:] - 1
                sample_logits_indices = logits_indices[sampling_request_indices]
                logits = output.logits[sample_logits_indices]

            output = self._sampler.sample(logits, sample_batch)
            sampled_tokens = dict(
                zip(
                    sampling_request_indices,
                    self._copy_tokens_to_host(output.token_ids),
                    strict=True,
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

    def _ensure_token_buffer(self, batch_size: int) -> None:
        if self._host_tokens is not None and self._host_tokens.numel() >= batch_size:
            return
        cap = max(batch_size, _MIN_SAMPLE_CAP)
        pin = self._device.type == "cuda"
        self._host_tokens = torch.empty(cap, dtype=torch.long, pin_memory=pin)

    def _ensure_param_buffers(self, batch_size: int) -> None:
        if self._host_f32 is not None and self._host_f32.size(0) >= batch_size:
            return
        cap = max(batch_size, _MIN_SAMPLE_CAP)
        pin = self._device.type == "cuda"
        self._host_f32 = torch.empty((cap, 3), dtype=torch.float32, pin_memory=pin)
        self._host_i64 = torch.empty((cap, 3), dtype=torch.int64, pin_memory=pin)
        self._gpu_f32 = torch.empty((cap, 3), dtype=torch.float32, device=self._device)
        self._gpu_i64 = torch.empty((cap, 3), dtype=torch.int64, device=self._device)

    def _pack_sampling_params(
        self,
        sampling_requests,
        *,
        vocab_size: int,
    ) -> SamplingBatch:
        n = len(sampling_requests)
        host_f32 = self._host_f32.numpy()
        host_i64 = self._host_i64.numpy()
        for index, request in enumerate(sampling_requests):
            params = request.sampling_plan.params
            host_f32[index, 0] = params.temperature
            host_f32[index, 1] = params.top_p
            host_f32[index, 2] = params.min_p
            host_i64[index, 0] = (
                params.top_k if params.top_k is not None else vocab_size
            )
            host_i64[index, 1] = params.seed
            host_i64[index, 2] = request.sampling_plan.sample_index
        non_blocking = self._device.type == "cuda"
        self._gpu_f32[:n].copy_(self._host_f32[:n], non_blocking=non_blocking)
        self._gpu_i64[:n].copy_(self._host_i64[:n], non_blocking=non_blocking)
        return SamplingBatch(
            temperatures=self._gpu_f32[:n, 0],
            top_ks=self._gpu_i64[:n, 0],
            top_ps=self._gpu_f32[:n, 1],
            min_ps=self._gpu_f32[:n, 2],
            seeds=self._gpu_i64[:n, 1],
            offsets=self._gpu_i64[:n, 2],
        )

    def _copy_tokens_to_host(self, token_ids: Tensor) -> list[int]:
        n = int(token_ids.numel())
        non_blocking = token_ids.is_cuda
        self._host_tokens[:n].copy_(token_ids, non_blocking=non_blocking)
        if non_blocking:
            torch.cuda.current_stream().synchronize()
        return self._host_tokens[:n].tolist()
