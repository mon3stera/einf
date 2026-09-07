from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.profiler import record_function

from einf.cache.storage import TorchKVCacheStorage
from einf.executors.torch.input import ModelInput
from einf.executors.torch.model_runner import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    scaled_dot_product_attention,
)
from einf.executors.torch.decode_graph import (
    DecodeCudaGraph,
    MAX_DECODE_GRAPH_BATCH,
)
from einf.executors.torch.flashinfer_attn import FlashInferPagedAttention
from einf.executors.torch.fused_ops import (
    apply_rope_inplace,
    fused_add_rmsnorm,
    rmsnorm,
    silu_and_mul,
    use_flashinfer_fused,
)
from einf.executors.torch.packed_qkv import PackedQKV
from einf.executors.torch.ops import (
    flash_attention,
    paged_decode_attention,
    paged_decode_attention_split_kv,
)
from einf.executors.torch.output import ModelOutput


def choose_num_splits(num_logical_blocks: int, max_splits: int) -> int:
    """Pick the Split-KV count for one single-Query decode request.

    Measured optima on an RTX 4090 with Qwen2.5-0.5B geometry
    (``benchmarks/micro-paged-decode-2026-08-23.md``), as logical blocks -> splits:
    8 -> 8, 34 -> 16, 64 -> 16, 128 -> 32, 256 -> 64, 512 -> 64, 1024 -> 64.

    The floor of 16 matters at short context, where ``// 4`` alone under-splits and
    the resulting grid of ``num_attention_heads * num_splits`` CTAs cannot fill the
    device: at context 544 it produced 112 CTAs for 128 SMs and left 1.31x on the
    table. The cap matters in the other direction, because the merge kernel is a
    single-warp serial loop over splits, so its cost grows linearly with them and
    eventually dominates the call.

    The result always lies in ``[1, num_logical_blocks]``, which is what the
    operator's ``num_splits <= num_logical_blocks`` precondition requires.
    """
    return min(max_splits, max(min(num_logical_blocks, 16), num_logical_blocks // 4))


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
        return rmsnorm(hidden_states, self.weight, self.eps)


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
        self.register_buffer(
            "gate_up_weight",
            torch.empty(config.intermediate_size * 2, config.hidden_size),
            persistent=False,
        )
        self.register_buffer("gate_up_qweight", None, persistent=False)
        self.pack_gate_up()

    def pack_gate_up(self) -> None:
        packed = torch.cat((self.gate_proj.weight.detach(), self.up_proj.weight.detach()), dim=0)
        if (
            self.gate_up_weight.shape != packed.shape
            or self.gate_up_weight.dtype != packed.dtype
            or self.gate_up_weight.device != packed.device
        ):
            self.gate_up_weight = packed.detach().clone()
        else:
            self.gate_up_weight.copy_(packed)

    def quantize_w4a16(self, *, group_size: int = 128) -> None:
        """One-way Marlin W4A16 transform of the packed gate/up GEMM."""
        from einf.executors.torch.w4a16 import quantize_packed_weight

        qweight, scales = quantize_packed_weight(self.gate_up_weight.t(), group_size=group_size)
        self.register_buffer("gate_up_qweight", qweight, persistent=False)
        self.register_buffer("gate_up_scales", scales, persistent=False)
        self.register_buffer(
            "marlin_workspace",
            torch.zeros(
                self.gate_up_weight.shape[0] // 128 * 8,
                dtype=torch.int32,
                device=self.gate_up_weight.device,
            ),
            persistent=False,
        )
        self.gate_proj = None
        self.up_proj = None
        self.gate_up_weight = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.gate_up_qweight is not None:
            from einf.executors.torch.w4a16 import marlin_linear

            gate_up = marlin_linear(
                hidden_states,
                self.gate_up_qweight,
                self.gate_up_scales,
                self.marlin_workspace,
                out_features=int(self.gate_up_qweight.shape[1] // 2),
            )
            return self.down_proj(silu_and_mul(gate_up))
        if use_flashinfer_fused(hidden_states):
            gate_up = torch.nn.functional.linear(hidden_states, self.gate_up_weight)
            return self.down_proj(silu_and_mul(gate_up))
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
        use_paged_decode_attention: bool = False,
        use_flashinfer_attention: bool = False,
        paged_decode_max_splits: int = 64,
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
        self.use_paged_decode_attention = use_paged_decode_attention
        self.use_flashinfer_attention = use_flashinfer_attention
        self.paged_decode_max_splits = paged_decode_max_splits
        self.rope_theta = config.rope_theta
        self.flashinfer: FlashInferPagedAttention | None = None

        attention_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.qkv = PackedQKV(
            config.hidden_size,
            attention_size,
            kv_size,
            kv_size,
            bias=True,
        )
        self.o_proj = nn.Linear(attention_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor | None,
        sin: Tensor | None,
        model_input: ModelInput,
        cache: TorchKVCacheStorage,
    ) -> Tensor:
        packed_len = hidden_states.size(0)
        with record_function("attn.qkv"):
            Q, K, V = self.qkv(hidden_states)
            Q = Q.view(packed_len, self.num_attention_heads, self.head_dim)
            K = K.view(packed_len, self.num_key_value_heads, self.head_dim)
            V = V.view(packed_len, self.num_key_value_heads, self.head_dim)

        if use_flashinfer_fused(Q):
            with record_function("attn.rope"):
                apply_rope_inplace(Q, K, model_input.position, rope_theta=self.rope_theta)
        else:
            with record_function("attn.rope"):
                Q, K = apply_rotary_pos_emb(
                    Q,
                    K,
                    cos.to(dtype=Q.dtype),
                    sin.to(dtype=Q.dtype),
                )
        with record_function("attn.write"):
            cache.write_slots(
                self.layer_idx,
                model_input.slot_mapping,
                K.contiguous(),
                V.contiguous(),
            )

        if self.flashinfer is not None:
            k_cache, v_cache = cache.layer_cache(self.layer_idx)
            with record_function("attn.flashinfer"):
                output = self.flashinfer.run(Q.contiguous(), k_cache, v_cache)
            with record_function("attn.o_proj"):
                return self.o_proj(output.reshape(packed_len, -1))

        request_outputs = []
        for request_idx in range(len(model_input.query_start_loc_host) - 1):
            query_start = model_input.query_start_loc_host[request_idx]
            query_end = model_input.query_start_loc_host[request_idx + 1]
            q_len = query_end - query_start
            context_len = model_input.context_lens_host[request_idx]

            request_Q = Q[query_start:query_end]
            if self.use_paged_decode_attention and q_len == 1:
                K_cache, V_cache = cache.layer_cache(self.layer_idx)
                num_logical_blocks = math.ceil(
                    context_len / cache.geometry.block_len
                )
                num_splits = choose_num_splits(
                    num_logical_blocks,
                    self.paged_decode_max_splits,
                )
                paged_args = (
                    request_Q[0].contiguous(),
                    K_cache,
                    V_cache,
                    model_input.block_tables[request_idx],
                    context_len,
                )
                if num_splits == 1:
                    output = paged_decode_attention(
                        *paged_args,
                        scale=1.0 / math.sqrt(self.head_dim),
                    )
                else:
                    output = paged_decode_attention_split_kv(
                        *paged_args,
                        num_splits=num_splits,
                        scale=1.0 / math.sqrt(self.head_dim),
                    )
                request_outputs.append(output.reshape(1, -1))
                continue

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

            output = scaled_dot_product_attention(
                request_Q,
                context_K,
                context_V,
                scale=1.0 / math.sqrt(self.head_dim),
            )
            request_outputs.append(output.reshape(q_len, -1))

        return self.o_proj(torch.cat(request_outputs, dim=0))


class QwenDecoderLayer(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        layer_idx: int,
        *,
        use_flash_attention: bool = False,
        use_paged_decode_attention: bool = False,
        use_flashinfer_attention: bool = False,
        paged_decode_max_splits: int = 64,
    ) -> None:
        super().__init__()
        self.self_attn = QwenAttention(
            config,
            layer_idx,
            use_flash_attention=use_flash_attention,
            use_paged_decode_attention=use_paged_decode_attention,
            use_flashinfer_attention=use_flashinfer_attention,
            paged_decode_max_splits=paged_decode_max_splits,
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
        residual: Tensor | None,
        cos: Tensor | None,
        sin: Tensor | None,
        model_input: ModelInput,
        cache: TorchKVCacheStorage,
    ) -> tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            with record_function("layer.input_norm"):
                hidden_states = self.input_layernorm(hidden_states)
        else:
            with record_function("layer.input_norm"):
                hidden_states, residual = fused_add_rmsnorm(
                    hidden_states,
                    residual,
                    self.input_layernorm.weight,
                    self.input_layernorm.eps,
                )
        hidden_states = self.self_attn(
            hidden_states,
            cos,
            sin,
            model_input,
            cache,
        )
        with record_function("layer.post_attn_norm"):
            hidden_states, residual = fused_add_rmsnorm(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
        with record_function("layer.mlp"):
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class QwenBackbone(nn.Module):
    def __init__(
        self,
        config: QwenConfig,
        *,
        use_flash_attention: bool = False,
        use_paged_decode_attention: bool = False,
        use_flashinfer_attention: bool = False,
        paged_decode_max_splits: int = 64,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            QwenDecoderLayer(
                config,
                layer_idx,
                use_flash_attention=use_flash_attention,
                use_paged_decode_attention=use_paged_decode_attention,
                use_flashinfer_attention=use_flashinfer_attention,
                paged_decode_max_splits=paged_decode_max_splits,
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
        use_paged_decode_attention: bool = False,
        use_flashinfer_attention: bool = False,
        paged_decode_max_splits: int = 64,
        dtype: torch.dtype | None = None,
        w4a16: bool = False,
    ) -> None:
        super().__init__()
        if w4a16 and not torch.cuda.is_available():
            raise ValueError("W4A16 quantized weights require a CUDA device")
        self._w4a16 = w4a16
        geometry = cache.geometry
        if (
            geometry.num_layers != config.num_hidden_layers
            or geometry.num_kv_heads != config.num_key_value_heads
            or geometry.head_dim != config.head_dim
        ):
            raise ValueError("cache geometry must match Qwen configuration")
        if cache.K.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            if not use_flashinfer_attention:
                raise ValueError(
                    "FP8 KV cache requires the FlashInfer attention backend"
                )
            if dtype is None:
                raise ValueError(
                    "FP8 KV cache requires an explicit model dtype so the "
                    "attention query dtype differs from the storage dtype"
                )
        if use_flashinfer_attention and (
            use_flash_attention or use_paged_decode_attention
        ):
            raise ValueError(
                "FlashInfer attention cannot be combined with in-house attention flags"
            )
        if use_flash_attention and config.head_dim != 64:
            raise ValueError("FlashAttention v0 requires head_dim == 64")
        if use_paged_decode_attention:
            if not cache.K.is_cuda:
                raise ValueError("Paged Decode Attention requires CUDA cache storage")
            if config.head_dim % 32 != 0 or config.head_dim > 256:
                raise ValueError(
                    "Paged Decode Attention requires head_dim to be a multiple "
                    "of 32 and no greater than 256"
                )
            if paged_decode_max_splits <= 0:
                raise ValueError("paged_decode_max_splits must be positive")

        self.config = config
        self.cache = cache
        self.flashinfer: FlashInferPagedAttention | None = None
        self.decode_graph: DecodeCudaGraph | None = None
        if use_flashinfer_attention:
            if not cache.K.is_cuda:
                raise ValueError("FlashInfer attention requires CUDA cache storage")
            self.flashinfer = FlashInferPagedAttention.create(
                num_qo_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                page_size=geometry.block_len,
                device=cache.K.device,
                dtype=dtype if dtype is not None else cache.K.dtype,
                kv_dtype=cache.K.dtype,
                max_nnz=max(geometry.num_blocks, MAX_DECODE_GRAPH_BATCH * geometry.num_blocks),
            )
        # W4A16 builds the fp masters on CPU: quantization halves the
        # per-layer footprint before anything reaches the GPU, so a 7B model
        # never needs the ~22 GiB fp transient on the device.
        param_device = torch.device("cpu") if self._w4a16 else cache.K.device
        with torch.device(param_device):
            self.model = QwenBackbone(
                config,
                use_flash_attention=use_flash_attention,
                use_paged_decode_attention=use_paged_decode_attention,
                use_flashinfer_attention=use_flashinfer_attention,
                paged_decode_max_splits=paged_decode_max_splits,
            )
        if self.flashinfer is not None:
            for layer in self.model.layers:
                layer.self_attn.flashinfer = self.flashinfer
            if geometry.num_blocks >= 2:
                self.decode_graph = DecodeCudaGraph(
                    runner=self,
                    device=cache.K.device,
                    block_len=geometry.block_len,
                    num_blocks=geometry.num_blocks,
                    dummy_block=geometry.num_blocks - 1,
                )
        with torch.device(param_device):
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def load_checkpoint(self, path: str | Path) -> None:
        import json

        from safetensors.torch import load_file

        path = Path(path)
        if path.is_dir():
            single = path / "model.safetensors"
            index = path / "model.safetensors.index.json"
            if single.is_file():
                shard_files = [single]
            elif index.is_file():
                weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
                shard_files = sorted({path / name for name in weight_map.values()})
            else:
                raise FileNotFoundError(f"no model.safetensors or index under {path}")
        else:
            shard_files = [path]

        device_str = str(self.model.embed_tokens.weight.device)
        weights: dict[str, Tensor] = {}
        for shard_file in shard_files:
            weights.update(load_file(str(shard_file), device=device_str))
        if self.config.tie_word_embeddings and "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        self.load_state_dict(weights, strict=True)

        # W4A16 quantized on CPU masters; only now move the compact weights up.
        if self._w4a16:
            self.to(device=self.cache.K.device)

    def load_state_dict(self, state_dict, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(state_dict, dict):
            state_dict = PackedQKV.remap_hf_keys(state_dict)
        result = super().load_state_dict(state_dict, *args, **kwargs)
        for layer in self.model.layers:
            layer.mlp.pack_gate_up()
            layer.self_attn.qkv.pack()
        if self._w4a16:
            from einf.executors.torch.w4a16 import quantize_runner_w4a16

            quantize_runner_w4a16(self)
        return result

    def forward_compute(self, model_input: ModelInput) -> ModelOutput:
        with record_function("fwd.embed"):
            hidden_states = self.model.embed_tokens(model_input.input_token_ids)
        fused = use_flashinfer_fused(hidden_states)
        if fused:
            cos = None
            sin = None
        else:
            with record_function("fwd.rope_cache"):
                cos, sin = self.model.rotary_embedding(model_input.position)
        residual: Tensor | None = None
        for layer in self.model.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                cos,
                sin,
                model_input,
                self.cache,
            )
        with record_function("fwd.final_norm"):
            if residual is None:
                hidden_states = self.model.norm(hidden_states)
            else:
                hidden_states, residual = fused_add_rmsnorm(
                    hidden_states,
                    residual,
                    self.model.norm.weight,
                    self.model.norm.eps,
                )
        with record_function("fwd.lm_head"):
            return ModelOutput(logits=self.lm_head(hidden_states))

    def forward(self, model_input: ModelInput) -> ModelOutput:
        if self.flashinfer is not None:
            self.flashinfer.plan(model_input)
        return self.forward_compute(model_input)

    def try_decode_cuda_graph(self, model_input: ModelInput) -> ModelOutput | None:
        if self.decode_graph is None:
            return None
        return self.decode_graph.try_replay(model_input)

    def try_decode_cuda_graph_plan(self, plan: object) -> ModelOutput | None:
        if self.decode_graph is None:
            return None
        return self.decode_graph.try_replay_plan(plan)
