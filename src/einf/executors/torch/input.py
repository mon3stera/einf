
from dataclasses import dataclass

from einf.execution_plan import BatchPlan
import torch
from torch import Tensor


def is_decode_only_plan(plan: object) -> bool:
    """Host-side decode-graph eligibility check over the plan's Python lists.

    Lives here rather than in ``decode_graph`` so the pool can stamp the
    verdict into the ModelInput it builds without a circular import.
    """
    requests = getattr(plan, "requests", ())
    return bool(requests) and all(len(request.input_token_ids) == 1 for request in requests)


@dataclass(frozen=True, slots=True)
class ModelInput:
    input_token_ids: Tensor
    position: Tensor
    slot_mapping: Tensor
    query_start_loc: Tensor
    block_tables: Tensor
    context_lens: Tensor
    # Host-side copies of the two metadata arrays that the per-layer attention
    # loop needs as Python ints. They are built here from the plan's own lists,
    # so producing them is free. Reading the device tensors instead forced a
    # cudaStreamSynchronize three times per request per layer, which is 1152
    # synchronisations per step at concurrency 16 (Gate 6.1). The device tensors
    # stay because ops and index arithmetic still consume them.
    query_start_loc_host: tuple[int, ...]
    context_lens_host: tuple[int, ...]
    # FlashInfer CSR built on the host into pinned staging (qo_indptr,
    # kv_indptr, kv_indices, last_page_len), matching the decode-graph plan
    # strategy: indptr and last_page_len stay on pinned CPU so FlashInfer's
    # plan() copies them H2D, while packed indices are staged into a device
    # buffer to avoid plan()'s synchronous H2D for CPU page lists. ``None``
    # makes plan() fall back to the device-side ``build_paged_kv_csr``.
    flashinfer_csr: tuple[Tensor, Tensor, Tensor, Tensor] | None = None
    # Builder-asserted decode-graph eligibility: every query is a single
    # token AND no block table touches the graph's dummy page. ``try_replay``
    # trusts this instead of re-deriving it from device tensors, which would
    # force a cudaStreamSynchronize per check per forward — the same pattern
    # Gate 6.1 removed from the scheduler path via ``is_decode_only_plan``.
    # Only builders that construct inputs from host-side state (the pool and
    # the speculative engine) may set it.
    is_decode_only: bool = False

    @classmethod
    def from_plan(cls, plan: BatchPlan, *, block_len: int, device: torch.device) -> "ModelInput":
        input_token_ids = []
        position = []
        slot_mapping = []
        query_start_loc = [0]
        context_lens = []
        block_tables = []

        for request in plan.requests:
            input_token_ids.extend(request.input_token_ids)

            position.extend([request.start_position + i for i in range(len(request.input_token_ids))])

            for t in range(len(request.input_token_ids)):
                absolute_position = t + request.start_position

                logical_block_idx = absolute_position // block_len
                slot_idx = absolute_position % block_len

                physical_flatten_slot_idx = request.block_table[logical_block_idx] * block_len + slot_idx

                slot_mapping.append(physical_flatten_slot_idx)

            query_start_loc.append(query_start_loc[-1] + len(request.input_token_ids))
            context_lens.append(request.start_position + len(request.input_token_ids))
            block_tables.append(
                torch.tensor(
                    request.block_table,
                    device=device,
                    dtype=torch.long,
                )
            )

        return cls(
            input_token_ids=torch.tensor(input_token_ids, device=device, dtype=torch.long),
            position=torch.tensor(position, device=device, dtype=torch.long),
            slot_mapping=torch.tensor(slot_mapping, device=device, dtype=torch.long),
            query_start_loc=torch.tensor(query_start_loc, device=device, dtype=torch.long),
            block_tables=torch.nn.utils.rnn.pad_sequence(block_tables, batch_first=True, padding_value=-1),
            context_lens=torch.tensor(context_lens, device=device, dtype=torch.long),
            query_start_loc_host=tuple(query_start_loc),
            context_lens_host=tuple(context_lens),
        )

    @classmethod
    def from_batch(cls, batch: BatchPlan, *, block_len: int, device: torch.device) -> "ModelInput":
        """Backward-compatible alias for the cross-language plan boundary."""
        return cls.from_plan(batch, block_len=block_len, device=device)


class ModelInputPool:
    """Reusable pinned staging and device buffers for per-step ModelInput values.

    ``from_plan`` allocates every tensor fresh each step, including one pageable
    H2D copy per request block table, and the FlashInfer CSR path re-derives its
    indices on the device with ``torch.any`` checks and mask compaction that
    synchronise. The pool writes the same values from the plan's Python lists
    into pinned numpy staging and issues one non-blocking copy per field, so a
    mixed step reuses its buffers exactly like the decode-graph path does.

    Buffers grow on demand; growing replaces the tensors, so a previously built
    ``ModelInput`` must not outlive the next ``build`` call that exceeds its
    capacity. Returned device tensors are stream-ordered views: callers enqueue
    their kernels on the same stream the copies ran on.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._pin = device.type == "cuda"
        self._batch_cap = 0
        self._pages_cap = 0
        self._tokens_cap = 0
        self._nnz_cap = 0
        self._copies: torch.cuda.Event | None = None
        self._allocations = 0
        self._host_tokens: Tensor | None = None
        self._host_position: Tensor | None = None
        self._host_slots: Tensor | None = None
        self._host_qsl: Tensor | None = None
        self._host_context: Tensor | None = None
        self._host_pages: Tensor | None = None
        self._host_qo_indptr: Tensor | None = None
        self._host_kv_indptr: Tensor | None = None
        self._host_last_page: Tensor | None = None
        self._host_kv_indices: Tensor | None = None
        self._d_tokens: Tensor | None = None
        self._d_position: Tensor | None = None
        self._d_slots: Tensor | None = None
        self._d_qsl: Tensor | None = None
        self._d_context: Tensor | None = None
        self._d_pages: Tensor | None = None
        self._d_kv_indices: Tensor | None = None

    def build(self, plan: BatchPlan, *, block_len: int) -> ModelInput:
        requests = plan.requests
        batch = len(requests)
        if batch == 0:
            raise ValueError("cannot build ModelInput for an empty batch")

        packed_len = 0
        pages_needed = 0
        nnz = 0
        for request in requests:
            token_count = len(request.input_token_ids)
            context_len = request.start_position + token_count
            page_count = (context_len + block_len - 1) // block_len
            if context_len < 1:
                raise ValueError("FlashInfer paged KV requires context_len >= 1")
            if page_count > len(request.block_table):
                raise ValueError("block_tables is shorter than the paged context")
            packed_len += token_count
            pages_needed = max(pages_needed, len(request.block_table))
            nnz += page_count

        self._wait_for_previous_copies()
        self._ensure(batch=batch, pages=pages_needed, tokens=packed_len, nnz=nnz)
        query_start_loc_host, context_lens_host = self._fill(
            requests,
            block_len=block_len,
            pages_needed=pages_needed,
        )

        non_blocking = self._pin
        self._d_tokens[:packed_len].copy_(self._host_tokens[:packed_len], non_blocking=non_blocking)
        self._d_position[:packed_len].copy_(self._host_position[:packed_len], non_blocking=non_blocking)
        self._d_slots[:packed_len].copy_(self._host_slots[:packed_len], non_blocking=non_blocking)
        self._d_qsl[: batch + 1].copy_(self._host_qsl[: batch + 1], non_blocking=non_blocking)
        self._d_context[:batch].copy_(self._host_context[:batch], non_blocking=non_blocking)
        self._d_pages[:batch, :pages_needed].copy_(
            self._host_pages[:batch, :pages_needed],
            non_blocking=non_blocking,
        )
        self._d_kv_indices[:nnz].copy_(self._host_kv_indices[:nnz], non_blocking=non_blocking)
        if self._pin:
            if self._copies is None:
                self._copies = torch.cuda.Event()
            self._copies.record()

        return ModelInput(
            input_token_ids=self._d_tokens[:packed_len],
            position=self._d_position[:packed_len],
            slot_mapping=self._d_slots[:packed_len],
            query_start_loc=self._d_qsl[: batch + 1],
            block_tables=self._d_pages[:batch, :pages_needed],
            context_lens=self._d_context[:batch],
            query_start_loc_host=query_start_loc_host,
            context_lens_host=context_lens_host,
            flashinfer_csr=(
                self._host_qo_indptr[: batch + 1],
                self._host_kv_indptr[: batch + 1],
                self._d_kv_indices[:nnz],
                self._host_last_page[:batch],
            ),
            is_decode_only=is_decode_only_plan(plan),
        )

    def _wait_for_previous_copies(self) -> None:
        # A step that samples nothing (all-prefill batch) never synchronises the
        # stream, so the host could refill the pinned staging while this step's
        # queued H2D copies of the previous step are still in flight.
        if self._copies is not None and not self._copies.query():
            self._copies.synchronize()

    def _ensure(self, *, batch: int, pages: int, tokens: int, nnz: int) -> None:
        if (
            batch <= self._batch_cap
            and pages <= self._pages_cap
            and tokens <= self._tokens_cap
            and nnz <= self._nnz_cap
        ):
            return

        self._batch_cap = max(batch, self._batch_cap * 2, 8)
        self._pages_cap = max(pages, self._pages_cap * 2, 8)
        self._tokens_cap = max(tokens, self._tokens_cap * 2, 64)
        self._nnz_cap = max(nnz, self._nnz_cap * 2, 64)
        self._allocations += 1

        pin = {"pin_memory": self._pin}
        self._host_tokens = torch.empty(self._tokens_cap, dtype=torch.long, **pin)
        self._host_position = torch.empty(self._tokens_cap, dtype=torch.long, **pin)
        self._host_slots = torch.empty(self._tokens_cap, dtype=torch.long, **pin)
        self._host_qsl = torch.empty(self._batch_cap + 1, dtype=torch.long, **pin)
        self._host_context = torch.empty(self._batch_cap, dtype=torch.long, **pin)
        self._host_pages = torch.empty((self._batch_cap, self._pages_cap), dtype=torch.long, **pin)
        self._host_qo_indptr = torch.empty(self._batch_cap + 1, dtype=torch.int32, **pin)
        self._host_kv_indptr = torch.empty(self._batch_cap + 1, dtype=torch.int32, **pin)
        self._host_last_page = torch.empty(self._batch_cap, dtype=torch.int32, **pin)
        self._host_kv_indices = torch.empty(self._nnz_cap, dtype=torch.int32, **pin)

        device = self.device
        self._d_tokens = torch.empty(self._tokens_cap, dtype=torch.long, device=device)
        self._d_position = torch.empty(self._tokens_cap, dtype=torch.long, device=device)
        self._d_slots = torch.empty(self._tokens_cap, dtype=torch.long, device=device)
        self._d_qsl = torch.empty(self._batch_cap + 1, dtype=torch.long, device=device)
        self._d_context = torch.empty(self._batch_cap, dtype=torch.long, device=device)
        self._d_pages = torch.empty((self._batch_cap, self._pages_cap), dtype=torch.long, device=device)
        self._d_kv_indices = torch.empty(self._nnz_cap, dtype=torch.int32, device=device)

    def _fill(
        self,
        requests,
        *,
        block_len: int,
        pages_needed: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        tokens_np = self._host_tokens.numpy()
        position_np = self._host_position.numpy()
        slots_np = self._host_slots.numpy()
        qsl_np = self._host_qsl.numpy()
        context_np = self._host_context.numpy()
        pages_np = self._host_pages.numpy()
        qo_indptr_np = self._host_qo_indptr.numpy()
        kv_indptr_np = self._host_kv_indptr.numpy()
        last_page_np = self._host_last_page.numpy()
        kv_indices_np = self._host_kv_indices.numpy()

        query_start_loc_host = [0]
        context_lens_host = []
        qsl_np[0] = 0
        qo_indptr_np[0] = 0
        kv_indptr_np[0] = 0
        offset = 0
        nnz_offset = 0
        for index, request in enumerate(requests):
            token_ids = request.input_token_ids
            start = request.start_position
            table = request.block_table
            token_count = len(token_ids)
            context_len = start + token_count
            page_count = (context_len + block_len - 1) // block_len

            tokens_np[offset : offset + token_count] = token_ids
            for i in range(token_count):
                absolute_position = start + i
                position_np[offset + i] = absolute_position
                slot = table[absolute_position // block_len] * block_len + absolute_position % block_len
                slots_np[offset + i] = slot
            offset += token_count

            query_start_loc_host.append(query_start_loc_host[-1] + token_count)
            qsl_np[index + 1] = query_start_loc_host[-1]
            qo_indptr_np[index + 1] = query_start_loc_host[-1]
            context_lens_host.append(context_len)
            context_np[index] = context_len
            remainder = context_len % block_len
            last_page_np[index] = block_len if remainder == 0 else remainder

            row = pages_np[index]
            row[:pages_needed] = -1
            row[: len(table)] = table

            for page_idx in range(page_count):
                block_id = table[page_idx]
                if block_id < 0:
                    raise ValueError("valid block table entries must be non-negative")
                kv_indices_np[nnz_offset + page_idx] = block_id
            nnz_offset += page_count
            kv_indptr_np[index + 1] = nnz_offset

        return tuple(query_start_loc_host), tuple(context_lens_host)
