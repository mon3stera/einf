from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class KVCacheGeometry:
    num_layers: int
    num_blocks: int
    block_len: int
    num_kv_heads: int
    head_dim: int


class TorchKVCacheStorage:
    """Paged K/V cache with an optional FP8 (E4M3-FN) storage dtype.

    ``dtype`` is the working dtype of the attention inputs; ``kv_dtype`` is the
    on-device storage dtype of the cache and may be ``torch.float8_e4m3fn`` to
    halve KV memory and read bandwidth. FP8 storage is quantized on write by
    ``write_slots`` (post-RoPE values, satfinite) and is only consumable by the
    FlashInfer attention backend, which converts FP8 tiles back to the query
    dtype inside its kernels. ``k_scale``/``v_scale`` are per-tensor write-side
    scales; the default 1.0 matches vLLM's default FP8 KV configuration.
    """

    _FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

    def __init__(
        self,
        geometry: KVCacheGeometry,
        *,
        dtype: torch.dtype,
        device,
        use_custom_ops: bool | None = None,
        kv_dtype: torch.dtype | None = None,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        self.geometry = geometry
        self.kv_dtype = dtype if kv_dtype is None else kv_dtype
        if self.kv_dtype not in self._FLOAT_DTYPES + (torch.float8_e4m3fn,):
            raise ValueError(f"unsupported KV cache dtype: {self.kv_dtype}")
        if self.kv_dtype == torch.float8_e4m3fn and (k_scale != 1.0 or v_scale != 1.0):
            raise ValueError(
                "FP8 KV cache scale dequantization is not wired through the "
                "FlashInfer plan API yet; use unit scale"
            )
        self.k_scale = k_scale
        self.v_scale = v_scale
        self.K = torch.empty(
            (
                geometry.num_layers,
                geometry.num_blocks,
                geometry.block_len,
                geometry.num_kv_heads,
                geometry.head_dim,
            ),
            dtype=self.kv_dtype,
            device=device,
        )
        self.V = torch.empty_like(self.K, dtype=self.kv_dtype, device=device)
        if use_custom_ops is None:
            use_custom_ops = self.K.is_cuda
        if use_custom_ops and not self.K.is_cuda:
            raise ValueError("custom KV Cache ops require a CUDA device")
        self.use_custom_ops = use_custom_ops

    def _layer_cache(self, cache: Tensor, layer_idx: int) -> Tensor:
        return cache[layer_idx]

    def _layer_slots(self, cache: Tensor, layer_idx: int) -> Tensor:
        return cache[layer_idx].view(
            self.geometry.num_blocks * self.geometry.block_len,
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
        )

    def layer_cache(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Return one layer's paged K/V cache views for direct Attention ops."""
        return (
            self._layer_cache(self.K, layer_idx),
            self._layer_cache(self.V, layer_idx),
        )

    def write_slots(
        self,
        layer_idx: int,
        slot_mapping: Tensor,
        K: Tensor,
        V: Tensor,
    ) -> None:
        if self.use_custom_ops:
            from einf.executors.torch.ops import write_slots_

            write_slots_(
                self._layer_cache(self.K, layer_idx),
                self._layer_cache(self.V, layer_idx),
                slot_mapping,
                K,
                V,
                k_scale=self.k_scale,
                v_scale=self.v_scale,
            )
            return

        if K.dtype != self.kv_dtype:
            K = K.to(self.kv_dtype)
            V = V.to(self.kv_dtype)
        K_slots = self._layer_slots(self.K, layer_idx)
        V_slots = self._layer_slots(self.V, layer_idx)
        if self.kv_dtype not in self._FLOAT_DTYPES:
            # torch's CPU scatter ops have no FP8 kernels; the byte view is
            # bit-identical because FP8 storage is one byte wide.
            K_slots = K_slots.view(torch.uint8)
            V_slots = V_slots.view(torch.uint8)
            K = K.view(torch.uint8)
            V = V.view(torch.uint8)
        K_slots.index_copy_(0, slot_mapping, K)
        V_slots.index_copy_(0, slot_mapping, V)

    def read_slots(
        self,
        layer_idx: int,
        slot_mapping: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self._assert_readable()
        K_slots = self._layer_slots(self.K, layer_idx)
        V_slots = self._layer_slots(self.V, layer_idx)
        return (
            K_slots.index_select(0, slot_mapping),
            V_slots.index_select(0, slot_mapping),
        )

    def _assert_readable(self) -> None:
        """FP8 bytes must be dequantized by the attention kernel that reads them."""
        if self.kv_dtype not in self._FLOAT_DTYPES:
            raise NotImplementedError(
                "reading an FP8 KV cache is only supported through the "
                "FlashInfer attention backend"
            )

    def gather_context(
        self,
        layer_idx: int,
        block_tables: Tensor | tuple[int, ...],
        context_len: int,
    ) -> tuple[Tensor, Tensor]:
        self._assert_readable()
        block_table = torch.as_tensor(
            block_tables,
            device=self.K.device,
            dtype=torch.long,
        )
        if self.use_custom_ops:
            from einf.executors.torch.ops import gather_context as gather_context_op

            return gather_context_op(
                self._layer_cache(self.K, layer_idx),
                self._layer_cache(self.V, layer_idx),
                block_table,
                context_len,
            )

        positions = torch.arange(
            context_len,
            device=self.K.device,
            dtype=torch.long,
        )
        logical_block_indices = positions // self.geometry.block_len
        slot_offsets = positions % self.geometry.block_len
        physical_block_indices = block_table[logical_block_indices]
        slot_mapping = (
            physical_block_indices * self.geometry.block_len
            + slot_offsets
        )

        return self.read_slots(
            layer_idx=layer_idx,
            slot_mapping=slot_mapping,
        )
