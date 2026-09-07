from __future__ import annotations

import pytest
import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch.flashinfer_attn import flashinfer_available
from einf.executors.torch.qwen import QwenConfig, QwenModelRunner


def make_storage(*, device, kv_dtype=None, use_custom_ops=False) -> TorchKVCacheStorage:
    geometry = KVCacheGeometry(
        num_layers=1,
        num_blocks=4,
        block_len=4,
        num_kv_heads=2,
        head_dim=8,
    )
    return TorchKVCacheStorage(
        geometry,
        dtype=torch.bfloat16,
        device=device,
        use_custom_ops=use_custom_ops,
        kv_dtype=kv_dtype,
    )


def test_fp8_storage_keeps_working_dtype_separate() -> None:
    storage = make_storage(device=torch.device("cpu"), kv_dtype=torch.float8_e4m3fn)

    assert storage.K.dtype is torch.float8_e4m3fn
    assert storage.V.dtype is torch.float8_e4m3fn
    assert storage.kv_dtype is torch.float8_e4m3fn


def test_fp8_fallback_write_quantizes_values() -> None:
    storage = make_storage(device=torch.device("cpu"), kv_dtype=torch.float8_e4m3fn)
    slots = torch.tensor([0, 3], dtype=torch.long)
    K = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    V = torch.randn(2, 2, 8, dtype=torch.bfloat16)

    storage.write_slots(0, slots, K, V)

    flat_K = storage.K[0].view(-1, 2, 8)
    flat_V = storage.V[0].view(-1, 2, 8)
    assert torch.equal(flat_K[0].float(), K[0].to(torch.float8_e4m3fn).float())
    assert torch.equal(flat_K[3].float(), K[1].to(torch.float8_e4m3fn).float())
    assert torch.equal(flat_V[0].float(), V[0].to(torch.float8_e4m3fn).float())
    assert torch.equal(flat_V[3].float(), V[1].to(torch.float8_e4m3fn).float())


def test_fp8_read_guard_points_at_flashinfer() -> None:
    storage = make_storage(device=torch.device("cpu"), kv_dtype=torch.float8_e4m3fn)

    with pytest.raises(NotImplementedError, match="FlashInfer"):
        storage.read_slots(0, torch.tensor([0], dtype=torch.long))


def test_fp8_cache_requires_flashinfer_runner_backend() -> None:
    storage = make_storage(device=torch.device("cpu"), kv_dtype=torch.float8_e4m3fn)
    config = QwenConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )

    with pytest.raises(ValueError, match="FlashInfer"):
        QwenModelRunner(config, cache=storage, dtype=torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not flashinfer_available(), reason="FlashInfer is required for the backend probe"
)
def test_write_slots_fp8_quantizes_and_saturates() -> None:
    storage = make_storage(
        device=torch.device("cuda"),
        kv_dtype=torch.float8_e4m3fn,
        use_custom_ops=True,
    )
    slots = torch.tensor([0, 1, 2], dtype=torch.long, device="cuda")
    K = torch.randn(3, 2, 8, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(3, 2, 8, dtype=torch.bfloat16, device="cuda")
    K[2] = 1.0e4

    storage.write_slots(0, slots, K, V)

    flat_K = storage.K[0].view(-1, 2, 8)
    flat_V = storage.V[0].view(-1, 2, 8)
    assert torch.equal(flat_K[:2].float(), K[:2].to(torch.float8_e4m3fn).float())
    assert torch.equal(flat_V[:2].float(), V[:2].to(torch.float8_e4m3fn).float())

    # E4M3-FN saturates at +/-448 instead of turning into NaN.
    assert torch.equal(
        flat_K[2].float(),
        torch.full((2, 8), 448.0, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    not flashinfer_available(), reason="FlashInfer is required for the backend probe"
)
def test_flashinfer_prefill_accepts_fp8_kv_cache() -> None:
    import flashinfer

    # head_dim must be a supported FP8-MMA tile size; 16 is not, 64 matches
    # the Qwen models this engine serves.
    device = torch.device("cuda")
    workspace = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    K = torch.randn(8, 4, 2, 64, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    V = torch.randn(8, 4, 2, 64, dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn
    )
    q = torch.randn(6, 4, 64, dtype=torch.bfloat16, device=device)
    wrapper.plan(
        torch.tensor([0, 4, 6], dtype=torch.int32, device=device),
        torch.tensor([0, 1, 3], dtype=torch.int32, device=device),
        torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        torch.tensor([4, 2], dtype=torch.int32, device=device),
        4,
        2,
        64,
        4,
        causal=True,
        sm_scale=0.125,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
    )

    output = wrapper.run(q, (K, V))
    torch.cuda.synchronize()

    assert output.dtype is torch.bfloat16
    assert torch.isfinite(output.float()).all()
