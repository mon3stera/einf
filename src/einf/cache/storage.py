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
    def __init__(self, geometry: KVCacheGeometry, *, dtype: torch.dtype, device):
        self.geometry = geometry
        self.K = torch.empty((
            geometry.num_layers,
            geometry.num_blocks,
            geometry.block_len,
            geometry.num_kv_heads,
            geometry.head_dim,
        ), dtype=dtype, device=device)
        self.V = torch.empty_like(self.K, dtype=dtype, device=device)

    def _layer_slots(self, cache: Tensor, layer_idx: int) -> Tensor:
        return cache[layer_idx].view(
            self.geometry.num_blocks * self.geometry.block_len,
            self.geometry.num_kv_heads,
            self.geometry.head_dim
        )

    def write_slots(self, layer_idx: int, slot_mapping: Tensor, K: Tensor, V: Tensor) -> None:
        K_slots = self._layer_slots(self.K, layer_idx)
        V_slots = self._layer_slots(self.V, layer_idx)
        K_slots.index_copy_(0, slot_mapping, K)
        V_slots.index_copy_(0, slot_mapping, V)

    def read_slots(self, layer_idx: int, slot_mapping: Tensor) -> tuple[Tensor, Tensor]:
        K_slots = self._layer_slots(self.K, layer_idx)
        V_slots = self._layer_slots(self.V, layer_idx)
        return (K_slots.index_select(0, slot_mapping), V_slots.index_select(0, slot_mapping))

    def gather_context(self, layer_idx: int, block_tables: tuple[int, ...], context_len: int) -> tuple[Tensor, Tensor]:
        slot_mapping = []

        for i in range(context_len):
            logical_block_idx = i // self.geometry.block_len
            slot_idx = i % self.geometry.block_len
            physical_flatten_slot_idx = block_tables[logical_block_idx] * self.geometry.block_len + slot_idx
            slot_mapping.append(physical_flatten_slot_idx)

        slot_mapping = torch.tensor(slot_mapping, device=self.K.device, dtype=torch.long)

        return self.read_slots(layer_idx, slot_mapping)
