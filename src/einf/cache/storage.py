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
    def __init__(
        self,
        geometry: KVCacheGeometry,
        *,
        dtype: torch.dtype,
        device,
        use_custom_ops: bool | None = None,
    ) -> None:
        self.geometry = geometry
        self.K = torch.empty(
            (
                geometry.num_layers,
                geometry.num_blocks,
                geometry.block_len,
                geometry.num_kv_heads,
                geometry.head_dim,
            ),
            dtype=dtype,
            device=device,
        )
        self.V = torch.empty_like(self.K, dtype=dtype, device=device)
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
            )
            return

        K_slots = self._layer_slots(self.K, layer_idx)
        V_slots = self._layer_slots(self.V, layer_idx)
        K_slots.index_copy_(0, slot_mapping, K)
        V_slots.index_copy_(0, slot_mapping, V)

    def read_slots(
        self,
        layer_idx: int,
        slot_mapping: Tensor,
    ) -> tuple[Tensor, Tensor]:
        K_slots = self._layer_slots(self.K, layer_idx)
        V_slots = self._layer_slots(self.V, layer_idx)
        return (
            K_slots.index_select(0, slot_mapping),
            V_slots.index_select(0, slot_mapping),
        )

    def gather_context(
        self,
        layer_idx: int,
        block_tables: Tensor | tuple[int, ...],
        context_len: int,
    ) -> tuple[Tensor, Tensor]:
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
