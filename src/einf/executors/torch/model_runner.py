import math
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from einf.cache.storage import TorchKVCacheStorage
from einf.config import ModelConfig
from einf.executors.torch.input import ModelInput
from einf.executors.torch.output import ModelOutput


class DeterministicModelRunner:
    def __init__(self, *, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self._vocab_size = vocab_size

    def forward(self, model_input: ModelInput) -> ModelOutput:
        target_token_ids = (
            model_input.input_token_ids + 1
        ) % self._vocab_size
        logits = torch.full(
            (
                model_input.input_token_ids.numel(),
                self._vocab_size,
            ),
            float("-inf"),
            dtype=torch.float32,
            device=model_input.input_token_ids.device,
        )
        logits.scatter_(
            1,
            target_token_ids.unsqueeze(1),
            0.0,
        )
        return ModelOutput(logits=logits)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq = self.inv_freq.to(device=position_ids.device)
        freqs = position_ids[:, None].float() * inv_freq[None, :]
        embedding = torch.cat((freqs, freqs), dim=-1)
        return (
            embedding.cos().unsqueeze(1),
            embedding.sin().unsqueeze(1),
        )


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


def repeat_kv(x: Tensor, scale: int) -> Tensor:
    return torch.repeat_interleave(x, repeats=scale, dim=1)


def build_causal_mask(
    q_len: int,
    start_pos: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    total_kv_len = start_pos + q_len
    query_positions = torch.arange(
        start_pos,
        total_kv_len,
        device=device,
    ).unsqueeze(1)
    key_positions = torch.arange(
        total_kv_len,
        device=device,
    ).unsqueeze(0)
    allowed = query_positions >= key_positions

    mask = torch.full(
        (q_len, total_kv_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    return mask.masked_fill(allowed, 0.0)


def _sdpa_kernel_context(tensor: Tensor):
    if not tensor.is_cuda:
        return nullcontext()
    return sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    )


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    *,
    scale: float,
) -> Tensor:
    """Lower-right causal GQA attention.

    Q is ``[q_len, num_qo_heads, head_dim]``; K/V are
    ``[kv_len, num_kv_heads, head_dim]``. The returned tensor matches Q.
    Flash SDPA is preferred on CUDA; CPU uses the math kernel.
    """
    q_len = Q.size(0)
    kv_len = K.size(0)
    with _sdpa_kernel_context(Q):
        output = torch.nn.functional.scaled_dot_product_attention(
            Q.transpose(0, 1).unsqueeze(0),
            K.transpose(0, 1).unsqueeze(0),
            V.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_lower_right(q_len, kv_len),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )
    return output.squeeze(0).transpose(0, 1)


class SingleLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_kv_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_kv_heads"
            )

        self.layer_idx = layer_idx
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_attention_heads // num_kv_heads

        attention_size = num_attention_heads * head_dim
        kv_size = num_kv_heads * head_dim

        self.K = nn.Linear(hidden_size, kv_size)
        self.V = nn.Linear(hidden_size, kv_size)
        self.Q = nn.Linear(hidden_size, attention_size)
        self.O = nn.Linear(attention_size, hidden_size)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        slot_mapping: Tensor,
        block_tables: Tensor,
        query_start_loc: Tensor,
        context_lens: Tensor,
        cache: TorchKVCacheStorage,
    ) -> Tensor:
        packed_len = x.shape[0]

        Q = self.Q(x).view(
            packed_len,
            self.num_attention_heads,
            self.head_dim,
        )
        K = self.K(x).view(
            packed_len,
            self.num_kv_heads,
            self.head_dim,
        )
        V = self.V(x).view(
            packed_len,
            self.num_kv_heads,
            self.head_dim,
        )

        cos = cos.to(dtype=Q.dtype)
        sin = sin.to(dtype=Q.dtype)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        cache.write_slots(self.layer_idx, slot_mapping, K, V)

        request_outputs = []
        for request_idx in range(query_start_loc.numel() - 1):
            query_start = int(query_start_loc[request_idx].item())
            query_end = int(query_start_loc[request_idx + 1].item())
            q_len = query_end - query_start
            context_len = int(context_lens[request_idx].item())

            request_Q = Q[query_start:query_end].transpose(0, 1)
            context_K, context_V = cache.gather_context(
                layer_idx=self.layer_idx,
                block_tables=block_tables[request_idx],
                context_len=context_len,
            )
            context_K = repeat_kv(
                context_K,
                self.num_kv_groups,
            ).transpose(0, 1)
            context_V = repeat_kv(
                context_V,
                self.num_kv_groups,
            ).transpose(0, 1)

            scores = (
                request_Q @ context_K.transpose(-1, -2)
            ) / math.sqrt(self.head_dim)
            scores = scores + build_causal_mask(
                q_len,
                context_len - q_len,
                device=x.device,
                dtype=scores.dtype,
            )
            probabilities = torch.softmax(
                scores.float(),
                dim=-1,
            ).to(Q.dtype)
            attention_output = probabilities @ context_V
            request_outputs.append(
                attention_output.transpose(0, 1).reshape(q_len, -1)
            )

        return self.O(torch.cat(request_outputs, dim=0))


class ReferenceModelRunner(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        cache: TorchKVCacheStorage,
    ) -> None:
        super().__init__()
        geometry = cache.geometry
        if (
            geometry.num_layers != config.num_layers
            or geometry.num_kv_heads != config.num_kv_heads
            or geometry.head_dim != config.head_dim
        ):
            raise ValueError("cache geometry must match model configuration")

        self.cache = cache
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.rotary_embedding = RotaryEmbedding(
            config.head_dim,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [
                SingleLayer(
                    layer_idx=layer_idx,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_kv_heads=config.num_kv_heads,
                    head_dim=config.head_dim,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(self, model_input: ModelInput) -> ModelOutput:
        hidden_states = self.embedding(model_input.input_token_ids)
        cos, sin = self.rotary_embedding(model_input.position)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cos,
                sin,
                model_input.slot_mapping,
                model_input.block_tables,
                model_input.query_start_loc,
                model_input.context_lens,
                self.cache,
            )

        return ModelOutput(logits=self.lm_head(hidden_states))
