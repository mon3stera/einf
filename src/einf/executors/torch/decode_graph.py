from __future__ import annotations

import bisect

import torch
from torch import Tensor

from einf.executors.torch.input import ModelInput, is_decode_only_plan
from einf.executors.torch.output import ModelOutput


DECODE_GRAPH_BUCKETS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    10,
    12,
    14,
    16,
    20,
    24,
    28,
    32,
    40,
    48,
    56,
    64,
)
MAX_DECODE_GRAPH_BATCH = DECODE_GRAPH_BUCKETS[-1]


def select_decode_graph_bucket(batch: int) -> int | None:
    if batch <= 0 or batch > MAX_DECODE_GRAPH_BATCH:
        return None
    return DECODE_GRAPH_BUCKETS[bisect.bisect_left(DECODE_GRAPH_BUCKETS, batch)]


class DecodeCudaGraph:
    """Capture the whole decode forward (embed through lm_head) per batch bucket.

    FlashInfer ``plan()`` stays outside the graph. Attention ``run()`` is recorded
    as part of each layer, together with RMSNorm, QKV, RoPE, write_slots and MLP.
    """

    def __init__(
        self,
        *,
        runner: object,
        device: torch.device,
        block_len: int,
        num_blocks: int,
        dummy_block: int,
    ) -> None:
        self.runner = runner
        self.device = device
        self.block_len = block_len
        self.num_blocks = num_blocks
        self.dummy_block = dummy_block
        self.dummy_slot = dummy_block * block_len
        self._inputs: dict[int, ModelInput] = {}
        self._logits: dict[int, Tensor] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._failed: set[int] = set()
        self._errors: dict[int, BaseException] = {}
        self._host_tokens: Tensor | None = None
        self._host_position: Tensor | None = None
        self._host_slots: Tensor | None = None
        self._host_context: Tensor | None = None
        self._host_pages: Tensor | None = None
        self._host_kv_indptr: Tensor | None = None
        self._host_kv_indices: Tensor | None = None
        self._host_last_page: Tensor | None = None
        self._csr_ready = False
        self._csr_nnz = 0

    def try_replay(self, model_input: ModelInput) -> ModelOutput | None:
        # Trust the builder's is_decode_only assertion instead of re-deriving
        # eligibility from device tensors: a torch.equal / torch.any check on
        # a device tensor forces a cudaStreamSynchronize, and the speculative
        # engine replays five times per step. Builders stamp the flag from
        # host-side state, so the checks cost nothing.
        if not model_input.is_decode_only:
            return None
        batch = int(model_input.context_lens.numel())
        bucket = select_decode_graph_bucket(batch)
        if bucket is None or bucket in self._failed:
            return None
        self._fill(bucket, model_input, batch)
        self._csr_ready = False
        return self._replay_bucket(bucket, batch)

    def try_replay_plan(self, plan: object) -> ModelOutput | None:
        packed = self.pack_plan(plan)
        if packed is None:
            return None
        bucket, batch = packed
        return self._replay_bucket(bucket, batch)

    def pack_plan(self, plan: object) -> tuple[int, int] | None:
        """Pack a decode-only plan into the captured static buffers.

        Writes Python lists into pinned host staging, then one ``copy_`` per
        field into the graph tensors. Returns ``(bucket, batch)`` or ``None``
        if this plan cannot use a decode graph.
        """
        if not is_decode_only_plan(plan):
            return None
        batch = len(plan.requests)
        bucket = select_decode_graph_bucket(batch)
        if bucket is None or bucket in self._failed:
            return None
        if not self._fill_from_plan(bucket, plan, batch):
            return None
        self._csr_ready = True
        return bucket, batch

    def _ensure_captured(self, bucket: int) -> None:
        if bucket not in self._graphs:
            self._capture(bucket)

    def _static_input(self, bucket: int) -> ModelInput:
        cached = self._inputs.get(bucket)
        if cached is not None:
            return cached
        device = self.device
        tokens = torch.zeros(bucket, dtype=torch.long, device=device)
        position = torch.zeros(bucket, dtype=torch.long, device=device)
        slot_mapping = torch.full(
            (bucket,),
            self.dummy_slot,
            dtype=torch.long,
            device=device,
        )
        query_start_loc = torch.arange(bucket + 1, dtype=torch.long, device=device)
        block_tables = torch.full(
            (bucket, self.num_blocks),
            -1,
            dtype=torch.long,
            device=device,
        )
        block_tables[:, 0] = self.dummy_block
        context_lens = torch.ones(bucket, dtype=torch.long, device=device)
        model_input = ModelInput(
            input_token_ids=tokens,
            position=position,
            slot_mapping=slot_mapping,
            query_start_loc=query_start_loc,
            block_tables=block_tables,
            context_lens=context_lens,
            query_start_loc_host=tuple(range(bucket + 1)),
            context_lens_host=tuple(1 for _ in range(bucket)),
        )
        self._inputs[bucket] = model_input
        return model_input

    def _ensure_host(self) -> None:
        if self._host_tokens is not None:
            return
        pin = self.device.type == "cuda"
        rows = MAX_DECODE_GRAPH_BATCH
        kwargs = {"dtype": torch.long, "pin_memory": pin}
        self._host_tokens = torch.zeros(rows, **kwargs)
        self._host_position = torch.zeros(rows, **kwargs)
        self._host_slots = torch.zeros(rows, **kwargs)
        self._host_context = torch.zeros(rows, **kwargs)
        self._host_pages = torch.empty((rows, self.num_blocks), **kwargs)
        csr_kwargs = {"dtype": torch.int32, "pin_memory": pin}
        self._host_kv_indptr = torch.zeros(rows + 1, **csr_kwargs)
        self._host_kv_indices = torch.zeros(rows * self.num_blocks, **csr_kwargs)
        self._host_last_page = torch.zeros(rows, **csr_kwargs)

    def _copy_host(self, bucket: int) -> None:
        dst = self._static_input(bucket)
        non_blocking = dst.input_token_ids.is_cuda
        dst.input_token_ids.copy_(self._host_tokens[:bucket], non_blocking=non_blocking)
        dst.position.copy_(self._host_position[:bucket], non_blocking=non_blocking)
        dst.slot_mapping.copy_(self._host_slots[:bucket], non_blocking=non_blocking)
        dst.context_lens.copy_(self._host_context[:bucket], non_blocking=non_blocking)
        dst.block_tables.copy_(self._host_pages[:bucket], non_blocking=non_blocking)

    def _fill_from_plan(self, bucket: int, plan: object, batch: int) -> bool:
        self._ensure_host()
        tokens = self._host_tokens.numpy()
        position = self._host_position.numpy()
        slots = self._host_slots.numpy()
        context = self._host_context.numpy()
        pages = self._host_pages.numpy()
        indptr = self._host_kv_indptr.numpy()
        kv_indices = self._host_kv_indices.numpy()
        last_page = self._host_last_page.numpy()
        pages[:bucket].fill(-1)
        pages[:bucket, 0] = self.dummy_block
        block_len = self.block_len
        dummy = self.dummy_block
        nnz = 0
        indptr[0] = 0
        for index, request in enumerate(plan.requests):
            token_ids = request.input_token_ids
            pos = int(request.start_position)
            table = request.block_table
            logical = pos // block_len
            context_len = pos + 1
            n_pages = (context_len + block_len - 1) // block_len
            if (
                logical < 0
                or logical >= len(table)
                or len(table) > self.num_blocks
                or n_pages > len(table)
            ):
                return False
            physical = int(table[logical])
            if physical == dummy:
                return False
            tokens[index] = int(token_ids[0])
            position[index] = pos
            slots[index] = physical * block_len + (pos % block_len)
            context[index] = context_len
            remainder = context_len % block_len
            last_page[index] = block_len if remainder == 0 else remainder
            for page_idx, block_id in enumerate(table):
                block_id = int(block_id)
                if block_id == dummy:
                    return False
                pages[index, page_idx] = block_id
            for page_idx in range(n_pages):
                kv_indices[nnz] = int(table[page_idx])
                nnz += 1
            indptr[index + 1] = nnz
        if batch < bucket:
            tokens[batch:bucket] = 0
            position[batch:bucket] = 0
            slots[batch:bucket] = self.dummy_slot
            context[batch:bucket] = 1
            for index in range(batch, bucket):
                kv_indices[nnz] = dummy
                nnz += 1
                last_page[index] = 1
                indptr[index + 1] = nnz
        self._copy_host(bucket)
        self._csr_nnz = nnz
        return True

    def _fill(self, bucket: int, src: ModelInput, batch: int) -> None:
        dst = self._static_input(bucket)
        dst.input_token_ids[:batch].copy_(src.input_token_ids)
        dst.position[:batch].copy_(src.position)
        dst.slot_mapping[:batch].copy_(src.slot_mapping)
        dst.context_lens[:batch].copy_(src.context_lens)
        dst.block_tables.fill_(-1)
        dst.block_tables[:, 0] = self.dummy_block
        pages = min(src.block_tables.size(1), dst.block_tables.size(1))
        dst.block_tables[:batch, :pages].copy_(src.block_tables[:, :pages])
        if batch < bucket:
            dst.input_token_ids[batch:].zero_()
            dst.position[batch:].zero_()
            dst.slot_mapping[batch:] = self.dummy_slot
            dst.context_lens[batch:] = 1

    def _replay_bucket(self, bucket: int, batch: int) -> ModelOutput | None:
        try:
            self._ensure_captured(bucket)
            host_csr = None
            if self._csr_ready:
                host_csr = (
                    self._host_kv_indptr[: bucket + 1],
                    self._host_kv_indices[: self._csr_nnz],
                    self._host_last_page[:bucket],
                )
            self.runner.flashinfer.plan_decode_bucket(
                bucket,
                self._inputs[bucket],
                host_csr=host_csr,
            )
            self._graphs[bucket].replay()
        except Exception as exc:
            self._failed.add(bucket)
            self._errors[bucket] = exc
            return None
        return ModelOutput(logits=self._logits[bucket][:batch])

    def _capture(self, bucket: int) -> None:
        static_input = self._static_input(bucket)
        self.runner.flashinfer.ensure_decode_wrapper(bucket)
        runner = self.runner
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                runner.flashinfer.plan_decode_bucket(bucket, static_input)
                output = runner.forward_compute(static_input)
        torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            runner.flashinfer.plan_decode_bucket(bucket, static_input)
            with torch.cuda.graph(graph):
                output = runner.forward_compute(static_input)
        self._graphs[bucket] = graph
        self._logits[bucket] = output.logits
