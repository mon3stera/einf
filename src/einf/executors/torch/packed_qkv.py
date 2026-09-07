from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class PackedQKV(nn.Module):
    """Q, K, V projections packed into one GEMM.

    The three ``nn.Linear`` members keep HuggingFace-style names for checkpoints.
    ``pack()`` concatenates their weights into a graph-stable buffer; ``forward``
    runs a single ``F.linear`` and splits the result.
    """

    def __init__(
        self,
        hidden_size: int,
        q_size: int,
        k_size: int,
        v_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if min(hidden_size, q_size, k_size, v_size) <= 0:
            raise ValueError("PackedQKV sizes must be positive")
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        self.q_proj = nn.Linear(hidden_size, q_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, k_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, v_size, bias=bias)
        packed_out = q_size + k_size + v_size
        self.register_buffer(
            "packed_weight",
            torch.empty(packed_out, hidden_size),
            persistent=False,
        )
        if bias:
            self.register_buffer(
                "packed_bias",
                torch.empty(packed_out),
                persistent=False,
            )
        else:
            self.packed_bias = None
        self.register_buffer("qweight", None, persistent=False)
        self.pack()

    @staticmethod
    def remap_hf_keys(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Map ``self_attn.{q,k,v}_proj`` onto ``self_attn.qkv.{q,k,v}_proj``."""
        remapped: dict[str, Tensor] = {}
        for key, value in state_dict.items():
            if ".self_attn.qkv." in key:
                remapped[key] = value
                continue
            new_key = key
            for name in ("q_proj", "k_proj", "v_proj"):
                needle = f".self_attn.{name}."
                if needle in key:
                    new_key = key.replace(needle, f".self_attn.qkv.{name}.")
                    break
            remapped[new_key] = value
        return remapped

    def pack(self) -> None:
        packed = torch.cat(
            (
                self.q_proj.weight.detach(),
                self.k_proj.weight.detach(),
                self.v_proj.weight.detach(),
            ),
            dim=0,
        )
        if (
            self.packed_weight.shape != packed.shape
            or self.packed_weight.dtype != packed.dtype
            or self.packed_weight.device != packed.device
        ):
            self.register_buffer("packed_weight", packed.clone(), persistent=False)
        else:
            self.packed_weight.copy_(packed)
        if self.q_proj.bias is None:
            self.packed_bias = None
            return
        packed_bias = torch.cat(
            (
                self.q_proj.bias.detach(),
                self.k_proj.bias.detach(),
                self.v_proj.bias.detach(),
            ),
            dim=0,
        )
        current = self.packed_bias
        if (
            current is None
            or current.shape != packed_bias.shape
            or current.dtype != packed_bias.dtype
            or current.device != packed_bias.device
        ):
            self.register_buffer("packed_bias", packed_bias.clone(), persistent=False)
        else:
            current.copy_(packed_bias)

    def quantize_w4a16(self, *, group_size: int = 128) -> None:
        """One-way Marlin W4A16 transform; drops the BF16 master weights.

        Must run after ``pack()`` on a CUDA runner; ``load_state_dict`` can
        only be called again on a freshly constructed runner.
        """
        from einf.executors.torch.w4a16 import quantize_packed_weight

        packed_out = self.q_size + self.k_size + self.v_size
        qweight, scales = quantize_packed_weight(self.packed_weight.t(), group_size=group_size)
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("qkv_scales", scales, persistent=False)
        self.register_buffer(
            "marlin_workspace",
            torch.zeros(
                packed_out // 128 * 8,
                dtype=torch.int32,
                device=self.packed_weight.device,
            ),
            persistent=False,
        )
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.packed_weight = None

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.qweight is not None:
            from einf.executors.torch.w4a16 import marlin_linear

            projected = marlin_linear(
                hidden_states,
                self.qweight,
                self.qkv_scales,
                self.marlin_workspace,
                out_features=self.q_size + self.k_size + self.v_size,
                bias=self.packed_bias,
            )
        else:
            projected = F.linear(hidden_states, self.packed_weight, self.packed_bias)
        return torch.split(
            projected,
            (self.q_size, self.k_size, self.v_size),
            dim=-1,
        )
