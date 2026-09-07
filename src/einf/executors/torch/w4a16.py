"""W4A16 serving path backed by the vendored Marlin kernel.

v1 scope (linear weights only)
------------------------------
The four GEMM-projection linears of every decoder layer (qkv, o_proj,
gate_up, down) store their weights as Marlin-layout ``int32`` qweights plus
per-group FP16 scales; activations, KV cache, embeddings and the (tied) LM
head stay in the model dtype. Attention backends and the KV cache therefore
need no changes.

``quantize_runner_w4a16`` is a one-way post-load transform: it runs after a
BF16 checkpoint is loaded, packs the weights, and drops the BF16 masters so
memory actually halves. Re-loading a checkpoint requires a fresh runner.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from einf.executors.torch.dsl.w4a16_pack import (
    pack_w4a16_marlin,
    permute_scales_marlin,
)

_GROUP_SIZE = 128
_MAX_PAR = 8


def _workspace(numel: int, device) -> Tensor:
    return torch.zeros(numel, dtype=torch.int32, device=device)


def quantize_packed_weight(
    weight_t: Tensor,
    *,
    group_size: int = _GROUP_SIZE,
) -> tuple[Tensor, Tensor]:
    """Pack ``weight_t = linear.weight.T`` ([K, N]) for the Marlin CUDA kernel.

    Returns ``(qweight, fp16 scales)`` with the kernel-side scale permutation
    already applied.
    """
    qweight, scales = pack_w4a16_marlin(weight_t, group_size=group_size)
    return qweight, permute_scales_marlin(scales).to(torch.float16)


def marlin_linear(
    x: Tensor,
    qweight: Tensor,
    scales: Tensor,
    workspace: Tensor,
    *,
    out_features: int,
    bias: Tensor | None = None,
) -> Tensor:
    """``x @ dequant(qweight)`` in FP16, returned in the input's dtype."""
    from einf.executors.torch.ops import marlin_gemm

    input_2d = x.reshape(-1, x.shape[-1])
    out = torch.empty(
        (input_2d.shape[0], out_features),
        dtype=torch.float16,
        device=x.device,
    )
    marlin_gemm(
        input_2d.to(torch.float16),
        qweight,
        scales,
        out,
        workspace,
        group_size=_GROUP_SIZE,
        max_par=_MAX_PAR,
    )
    result = out.to(x.dtype)

    if bias is not None:
        result = result + bias
    return result


class MarlinW4A16Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` built from a trained weight."""

    def __init__(
        self,
        qweight: Tensor,
        scales: Tensor,
        workspace: Tensor,
        *,
        in_features: int,
        out_features: int,
        bias: Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", qweight, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("workspace", workspace, persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, group_size: int = _GROUP_SIZE) -> "MarlinW4A16Linear":
        in_features = linear.in_features
        out_features = linear.out_features
        qweight, scales = quantize_packed_weight(
            linear.weight.detach().t(),
            group_size=group_size,
        )
        workspace = _workspace(
            out_features // 128 * _MAX_PAR,
            linear.weight.device,
        )
        bias = None if linear.bias is None else linear.bias.detach()
        return cls(
            qweight,
            scales,
            workspace,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return marlin_linear(
            x,
            self.qweight,
            self.scales,
            self.workspace,
            out_features=self.out_features,
            bias=self.bias,
        )


def quantize_runner_w4a16(runner: nn.Module, *, group_size: int = _GROUP_SIZE) -> None:
    """One-way W4A16 transform of every decoder-layer projection GEMM."""
    for layer in runner.model.layers:
        layer.self_attn.qkv.quantize_w4a16(group_size=group_size)
        layer.self_attn.o_proj = MarlinW4A16Linear.from_linear(
            layer.self_attn.o_proj,
            group_size=group_size,
        )
        layer.mlp.quantize_w4a16(group_size=group_size)
        layer.mlp.down_proj = MarlinW4A16Linear.from_linear(
            layer.mlp.down_proj,
            group_size=group_size,
        )
