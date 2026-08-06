from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from einf.cache.storage import TorchKVCacheStorage
from einf.executors.torch.input import ModelInput
from einf.executors.torch.model_runner import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    build_causal_mask,
    repeat_kv,
)
from einf.executors.torch.ops import flash_attention
from einf.executors.torch.output import ModelOutput


@dataclass(frozen=True, slots=True)
class QwenConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    hidden_act: str
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    bos_token_id: int
    eos_token_id: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, data: dict) -> "QwenConfig":
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            rms_norm_eps=data["rms_norm_eps"],
            hidden_act=data["hidden_act"],
            rope_theta=data["rope_theta"],
            max_position_embeddings=data["max_position_embeddings"],
            tie_word_embeddings=data["tie_word_embeddings"],
            bos_token_id=data["bos_token_id"],
            eos_token_id=data["eos_token_id"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "QwenConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(dtype=input_dtype)


class QwenMLP(nn.Module):
    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported Qwen activation: {config.hidden_act}")
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


class QwenAttention(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        layer_idx: int,
        *,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.use_flash_attention = use_flash_attention

        attention_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, attention_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=True)
        self.o_proj = nn.Linear(attention_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        model_input: ModelInput,
        cache: TorchKVCacheStorage,
    ) -> Tensor:
        packed_len = hidden_states.size(0)
        Q = self.q_proj(hidden_states).view(
            packed_len,
            self.num_attention_heads,
            self.head_dim,
        )
        K = self.k_proj(hidden_states).view(
            packed_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        V = self.v_proj(hidden_states).view(
            packed_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        Q, K = apply_rotary_pos_emb(
            Q,
            K,
            cos.to(dtype=Q.dtype),
            sin.to(dtype=Q.dtype),
        )
        cache.write_slots(
            self.layer_idx,
            model_input.slot_mapping,
            K.contiguous(),
            V.contiguous(),
        )

        request_outputs = []
        for request_idx in range(model_input.query_start_loc.numel() - 1):
            query_start = int(model_input.query_start_loc[request_idx].item())
            query_end = int(model_input.query_start_loc[request_idx + 1].item())
            q_len = query_end - query_start
            context_len = int(model_input.context_lens[request_idx].item())

            request_Q = Q[query_start:query_end]
            context_K, context_V = cache.gather_context(
                self.layer_idx,
                model_input.block_tables[request_idx],
                context_len,
            )

            if self.use_flash_attention:
                output = flash_attention(
                    request_Q.contiguous(),
                    context_K.contiguous(),
                    context_V.contiguous(),
                    context_len - q_len,
                    1.0 / math.sqrt(self.head_dim),
                )
                request_outputs.append(output.reshape(q_len, -1))
                continue

            request_Q = request_Q.transpose(0, 1)
            context_K = repeat_kv(
                context_K,
                self.num_key_value_groups,
            ).transpose(0, 1)
            context_V = repeat_kv(
                context_V,
                self.num_key_value_groups,
            ).transpose(0, 1)

            scores = (
                request_Q @ context_K.transpose(-1, -2)
            ) / math.sqrt(self.head_dim)
            scores = scores + build_causal_mask(
                q_len,
                context_len - q_len,
                device=hidden_states.device,
                dtype=scores.dtype,
            )
            probabilities = torch.softmax(scores.float(), dim=-1).to(Q.dtype)
            output = probabilities @ context_V
            request_outputs.append(
                output.transpose(0, 1).reshape(q_len, -1)
            )

        return self.o_proj(torch.cat(request_outputs, dim=0))


class QwenDecoderLayer(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        layer_idx: int,
        *,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = QwenAttention(
            config,
            layer_idx,
            use_flash_attention=use_flash_attention,
        )
        self.mlp = QwenMLP(config)
        self.input_layernorm = QwenRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.post_attention_layernorm = QwenRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        model_input: ModelInput,
        cache: TorchKVCacheStorage,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            model_input,
            cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class QwenBackbone(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        *,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            QwenDecoderLayer(
                config,
                layer_idx,
                use_flash_attention=use_flash_attention,
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = QwenRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_embedding = RotaryEmbedding(
            config.head_dim,
            base=config.rope_theta,
        )


class QwenModelRunner(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        *,
        cache: TorchKVCacheStorage,
        use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        geometry = cache.geometry
        if (
            geometry.num_layers != config.num_hidden_layers
            or geometry.num_kv_heads != config.num_key_value_heads
            or geometry.head_dim != config.head_dim
        ):
            raise ValueError("cache geometry must match Qwen configuration")
        if use_flash_attention and config.head_dim != 64:
            raise ValueError("FlashAttention v0 requires head_dim == 64")

        self.config = config
        self.cache = cache
        self.model = QwenBackbone(
            config,
            use_flash_attention=use_flash_attention,
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def load_checkpoint(self, path: str | Path) -> None:
        from safetensors.torch import load_file

        weights = load_file(
            str(path),
            device=str(self.model.embed_tokens.weight.device),
        )
        if self.config.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        self.load_state_dict(weights, strict=True)

    def forward(self, model_input: ModelInput) -> ModelOutput:
        hidden_states = self.model.embed_tokens(model_input.input_token_ids)
        cos, sin = self.model.rotary_embedding(model_input.position)
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                model_input,
                self.cache,
            )
        hidden_states = self.model.norm(hidden_states)
        return ModelOutput(logits=self.lm_head(hidden_states))
