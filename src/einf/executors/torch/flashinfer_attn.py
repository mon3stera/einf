from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from einf.executors.torch.decode_graph import MAX_DECODE_GRAPH_BATCH
from einf.executors.torch.input import ModelInput


def flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        return False
    return True


def build_paged_kv_csr(
    block_tables: Tensor,
    context_lens: Tensor,
    block_len: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert padded per-request block tables into FlashInfer CSR pages.

    ``block_tables`` is ``[batch, max_pages]`` with ``-1`` padding, matching
    ``ModelInput.block_tables``. FlashInfer wants a packed page list plus
    ``last_page_len`` in ``[1, block_len]``.
    """
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [batch, max_pages]")
    if context_lens.ndim != 1:
        raise ValueError("context_lens must have shape [batch]")
    if block_tables.size(0) != context_lens.numel():
        raise ValueError("block_tables batch dim must match context_lens")

    device = block_tables.device
    context_lens_i32 = context_lens.to(device=device, dtype=torch.int32)
    if torch.any(context_lens_i32 < 1):
        raise ValueError("FlashInfer paged KV requires context_len >= 1")

    num_pages = torch.div(
        context_lens_i32 + (block_len - 1),
        block_len,
        rounding_mode="floor",
    )
    if torch.any(num_pages > block_tables.size(1)):
        raise ValueError("block_tables is shorter than the paged context")

    batch = context_lens_i32.numel()
    indptr = torch.empty(batch + 1, dtype=torch.int32, device=device)
    indptr[0] = 0
    torch.cumsum(num_pages, dim=0, out=indptr[1:])

    page_ids = torch.arange(block_tables.size(1), device=device)
    valid = page_ids.unsqueeze(0) < num_pages.unsqueeze(1)
    indices = block_tables.to(dtype=torch.int32)[valid]
    if torch.any(indices < 0):
        raise ValueError("valid block table entries must be non-negative")

    last_page_len = context_lens_i32 % block_len
    last_page_len = torch.where(
        last_page_len == 0,
        torch.full_like(last_page_len, block_len),
        last_page_len,
    )
    return indptr, indices, last_page_len


@dataclass(slots=True)
class FlashInferPagedAttention:
    """FlashInfer attention with a prefill wrapper and per-bucket decode wrappers.

    Mixed / prefill steps use ``BatchPrefillWithPagedKVCacheWrapper``. Decode
    CUDA graphs use ``BatchDecodeWithPagedKVCacheWrapper`` (one token per
    request, no ``qo_indptr``). ``plan()`` is CPU-side and stays outside the
    graph; ``run()`` is the per-layer kernel.
    """

    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    sm_scale: float
    dtype: torch.dtype
    kv_dtype: torch.dtype
    workspace: Tensor
    wrapper: object
    max_batch: int = MAX_DECODE_GRAPH_BATCH
    indices: Tensor | None = None
    qo_indptr: Tensor | None = None
    kv_indptr: Tensor | None = None
    last_page_len: Tensor | None = None
    decode_wrappers: dict[int, object] = field(default_factory=dict)
    _active: object | None = None

    @classmethod
    def create(
        cls,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
        max_nnz: int,
        kv_dtype: torch.dtype | None = None,
        workspace_bytes: int = 128 * 1024 * 1024,
        max_batch: int = MAX_DECODE_GRAPH_BATCH,
    ) -> "FlashInferPagedAttention":
        import flashinfer

        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("num_qo_heads must be divisible by num_kv_heads")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if max_nnz <= 0:
            raise ValueError("max_nnz must be positive")
        workspace = torch.zeros(workspace_bytes, dtype=torch.uint8, device=device)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
        )
        return cls(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            sm_scale=1.0 / (head_dim ** 0.5),
            dtype=dtype,
            kv_dtype=dtype if kv_dtype is None else kv_dtype,
            workspace=workspace,
            wrapper=wrapper,
            max_batch=max_batch,
            indices=torch.zeros(max_nnz, dtype=torch.int32, device=device),
            qo_indptr=torch.zeros(max_batch + 1, dtype=torch.int32, device=device),
            kv_indptr=torch.zeros(max_batch + 1, dtype=torch.int32, device=device),
            last_page_len=torch.zeros(max_batch, dtype=torch.int32, device=device),
        )

    def _paged_csr(self, model_input: ModelInput) -> tuple[Tensor, Tensor, Tensor]:
        return build_paged_kv_csr(
            model_input.block_tables,
            model_input.context_lens,
            self.page_size,
        )

    def plan(self, model_input: ModelInput) -> None:
        self._active = self.wrapper
        csr = model_input.flashinfer_csr
        if csr is not None:
            qo_indptr, page_indptr, page_indices, last_page_len = csr
        else:
            qo_indptr = model_input.query_start_loc.to(dtype=torch.int32)
            page_indptr, page_indices, last_page_len = self._paged_csr(model_input)
        plan = getattr(self.wrapper, "plan", None) or getattr(
            self.wrapper, "begin_forward"
        )
        if plan is None:
            raise RuntimeError("FlashInfer wrapper has neither plan nor begin_forward")

        # A packed custom mask replaces the causal assumption entirely (the
        # phase-0 probe validated the packed bit convention): tree speculation
        # rows skip siblings, which causal cannot express. The mask covers
        # qo_len x kv_len bits, where kv_len counts TOKENS (context includes
        # the qo rows themselves — the cache is written before attention).
        plan_kwargs: dict[str, object] = {
            "causal": True,
            "sm_scale": self.sm_scale,
            "q_data_type": self.dtype,
            "kv_data_type": self.kv_dtype,
        }

        packed_mask = model_input.packed_mask

        if packed_mask is not None:
            if packed_mask.device != qo_indptr.device:
                raise ValueError("packed_mask must live on the plan device")
            qo_total = int(model_input.query_start_loc[-1])
            kv_total = int(model_input.context_lens.sum())
            expected_bits = qo_total * kv_total

            if packed_mask.numel() * 8 < expected_bits:
                raise ValueError(
                    f"packed_mask covers {packed_mask.numel() * 8} bits but "
                    f"the forward needs {expected_bits} "
                    f"(qo={qo_total} x kv={kv_total})"
                )
            plan_kwargs["causal"] = False
            plan_kwargs["packed_custom_mask"] = packed_mask

        plan(
            qo_indptr,
            page_indptr,
            page_indices,
            last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            **plan_kwargs,
        )

    def ensure_decode_wrapper(self, bucket: int) -> object:
        import flashinfer

        wrapper = self.decode_wrappers.get(bucket)
        if wrapper is not None:
            return wrapper
        if self.indices is None or self.kv_indptr is None or self.last_page_len is None:
            raise RuntimeError("FlashInfer CUDA graph buffers are not allocated")
        if bucket > self.max_batch:
            raise ValueError(f"decode graph bucket {bucket} exceeds max_batch")
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            use_tensor_cores=True,
            paged_kv_indptr_buffer=self.kv_indptr[: bucket + 1],
            paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=self.last_page_len[:bucket],
        )
        self.decode_wrappers[bucket] = wrapper
        return wrapper

    def plan_decode_bucket(
        self,
        bucket: int,
        model_input: ModelInput,
        *,
        host_csr: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> None:
        wrapper = self.ensure_decode_wrapper(bucket)
        self._active = wrapper
        if host_csr is not None:
            page_indptr, page_indices, last_page_len = host_csr
            if (
                self.indices is not None
                and page_indices.device != self.indices.device
            ):
                nnz = int(page_indices.numel())
                if nnz > self.indices.numel():
                    raise RuntimeError("decode CSR nnz exceeds FlashInfer indices buffer")
                self.indices[:nnz].copy_(page_indices, non_blocking=True)
                page_indices = self.indices[:nnz]
        else:
            page_indptr, page_indices, last_page_len = self._paged_csr(model_input)
        wrapper.plan(
            page_indptr,
            page_indices,
            last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            pos_encoding_mode="NONE",
            sm_scale=self.sm_scale,
            q_data_type=self.dtype,
            kv_data_type=self.kv_dtype,
        )

    def run(self, q: Tensor, k_cache: Tensor, v_cache: Tensor) -> Tensor:
        wrapper = self._active if self._active is not None else self.wrapper
        run = getattr(wrapper, "run", None) or getattr(wrapper, "forward")
        return run(q, (k_cache, v_cache))
