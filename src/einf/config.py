from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    eos_token_id: int
    rope_theta: float = 10000.0
