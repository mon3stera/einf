from __future__ import annotations

import torch
from torch import nn

from einf.executors.torch.packed_qkv import PackedQKV


def test_packed_qkv_matches_separate_linears() -> None:
    torch.manual_seed(0)
    hidden, q_size, k_size, v_size = 16, 12, 8, 8
    packed = PackedQKV(hidden, q_size, k_size, v_size, bias=True)
    x = torch.randn(5, hidden)
    q, k, v = packed(x)
    assert q.shape == (5, q_size)
    assert k.shape == (5, k_size)
    assert v.shape == (5, v_size)
    torch.testing.assert_close(q, packed.q_proj(x))
    torch.testing.assert_close(k, packed.k_proj(x))
    torch.testing.assert_close(v, packed.v_proj(x))


def test_packed_qkv_remap_hf_keys() -> None:
    keys = PackedQKV.remap_hf_keys(
        {
            "model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
            "model.layers.0.self_attn.k_proj.bias": torch.zeros(1),
            "model.layers.0.self_attn.qkv.v_proj.weight": torch.zeros(1),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(1),
        }
    )
    assert "model.layers.0.self_attn.qkv.q_proj.weight" in keys
    assert "model.layers.0.self_attn.qkv.k_proj.bias" in keys
    assert "model.layers.0.self_attn.qkv.v_proj.weight" in keys
    assert "model.layers.0.self_attn.o_proj.weight" in keys


def test_packed_qkv_no_bias() -> None:
    packed = PackedQKV(8, 8, 4, 4, bias=False)
    x = torch.randn(2, 8)
    q, k, v = packed(x)
    torch.testing.assert_close(q, nn.functional.linear(x, packed.q_proj.weight))
    assert packed.packed_bias is None
    assert k.shape[-1] == 4 and v.shape[-1] == 4
