from __future__ import annotations

import pytest
import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch.decode_graph import select_decode_graph_bucket
from einf.executors.torch.flashinfer_attn import (
    build_paged_kv_csr,
    flashinfer_available,
)
from einf.executors.torch.input import ModelInput
from einf.executors.torch.qwen import QwenConfig, QwenModelRunner
from einf.scheduler import ScheduledBatch, ScheduledRequest, WorkType


def _assert_attention_logits_match(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """FlashInfer fused softmax and eager fp32 softmax differ in raw logits.

    Token ranking is the serving contract: compare softmax, then allow a
    coarser bf16 logit tolerance.
    """
    torch.testing.assert_close(
        torch.softmax(actual.float(), dim=-1),
        torch.softmax(expected.float(), dim=-1),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=2e-2,
        atol=2.5e-1,
    )


@pytest.mark.parametrize(
    ("batch", "bucket"),
    [
        (1, 1),
        (8, 8),
        (9, 10),
        (16, 16),
        (17, 20),
        (32, 32),
        (33, 40),
        (64, 64),
        (0, None),
        (65, None),
    ],
)
def test_select_decode_graph_bucket(batch: int, bucket: int | None) -> None:
    assert select_decode_graph_bucket(batch) == bucket


def test_build_paged_kv_csr_packs_padded_block_tables() -> None:
    block_tables = torch.tensor(
        [
            [4, 7, -1, -1],
            [1, 2, 3, -1],
        ],
        dtype=torch.long,
    )
    context_lens = torch.tensor([20, 48], dtype=torch.long)
    indptr, indices, last_page_len = build_paged_kv_csr(
        block_tables,
        context_lens,
        block_len=16,
    )
    assert indptr.tolist() == [0, 2, 5]
    assert indices.tolist() == [4, 7, 1, 2, 3]
    assert last_page_len.tolist() == [4, 16]


def test_build_paged_kv_csr_full_last_page() -> None:
    block_tables = torch.tensor([[9, -1], [8, 2]], dtype=torch.long)
    context_lens = torch.tensor([16, 17], dtype=torch.long)
    indptr, indices, last_page_len = build_paged_kv_csr(
        block_tables,
        context_lens,
        block_len=16,
    )
    assert indptr.tolist() == [0, 1, 3]
    assert indices.tolist() == [9, 8, 2]
    assert last_page_len.tolist() == [16, 1]


def test_build_paged_kv_csr_rejects_empty_context() -> None:
    with pytest.raises(ValueError, match="context_len >= 1"):
        build_paged_kv_csr(
            torch.tensor([[-1]], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            block_len=16,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_qwen_flashinfer_attention_matches_eager() -> None:
    torch.manual_seed(3)
    dtype = torch.bfloat16
    config = QwenConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    geometry = KVCacheGeometry(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_len=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    eager_runner = QwenModelRunner(
        config,
        cache=TorchKVCacheStorage(
            geometry,
            dtype=dtype,
            device="cuda",
            use_custom_ops=False,
        ),
    ).to(device="cuda", dtype=dtype).eval()
    flashinfer_runner = QwenModelRunner(
        config,
        cache=TorchKVCacheStorage(
            geometry,
            dtype=dtype,
            device="cuda",
            use_custom_ops=False,
        ),
        use_flashinfer_attention=True,
    ).to(device="cuda", dtype=dtype).eval()
    flashinfer_runner.load_state_dict(eager_runner.state_dict())

    block_table = tuple(range(8))
    prefill_ids = tuple((index % 29) + 3 for index in range(31))
    prefill_input = ModelInput.from_batch(
        ScheduledBatch(
            step_id=0,
            requests=(
                ScheduledRequest(
                    request_id="request",
                    input_token_ids=prefill_ids,
                    work_type=WorkType.PREFILL,
                    start_position=0,
                    block_table=block_table,
                    need_sample=True,
                ),
            ),
        ),
        block_len=4,
        device=torch.device("cuda"),
    )
    decode_input = ModelInput.from_batch(
        ScheduledBatch(
            step_id=1,
            requests=(
                ScheduledRequest(
                    request_id="request",
                    input_token_ids=(7,),
                    work_type=WorkType.DECODE,
                    start_position=31,
                    block_table=block_table,
                    need_sample=True,
                ),
            ),
        ),
        block_len=4,
        device=torch.device("cuda"),
    )
    with torch.inference_mode():
        eager_runner(prefill_input)
        flashinfer_runner(prefill_input)
        eager_logits = eager_runner(decode_input).logits
        flashinfer_logits = flashinfer_runner(decode_input).logits
    _assert_attention_logits_match(flashinfer_logits, eager_logits)


def _tiny_cuda_runners() -> tuple[QwenModelRunner, QwenModelRunner]:
    dtype = torch.bfloat16
    config = QwenConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=10000.0,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    geometry = KVCacheGeometry(
        num_layers=config.num_hidden_layers,
        num_blocks=8,
        block_len=4,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )

    def make_runner(*, flashinfer: bool) -> QwenModelRunner:
        return QwenModelRunner(
            config,
            cache=TorchKVCacheStorage(
                geometry,
                dtype=dtype,
                device="cuda",
                use_custom_ops=False,
            ),
            use_flashinfer_attention=flashinfer,
        ).to(device="cuda", dtype=dtype).eval()

    eager = make_runner(flashinfer=False)
    flashinfer = make_runner(flashinfer=True)
    flashinfer.load_state_dict(eager.state_dict())
    return eager, flashinfer


def _cuda_input(
    requests: tuple[ScheduledRequest, ...],
    *,
    step_id: int,
    block_len: int = 4,
) -> ModelInput:
    return ModelInput.from_batch(
        ScheduledBatch(step_id=step_id, requests=requests),
        block_len=block_len,
        device=torch.device("cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_qwen_flashinfer_mixed_prefill_decode_matches_eager() -> None:
    torch.manual_seed(4)
    eager_runner, flashinfer_runner = _tiny_cuda_runners()
    prefill_a = _cuda_input(
        (
            ScheduledRequest(
                request_id="a",
                input_token_ids=(3, 4, 5, 6, 7, 8),
                work_type=WorkType.PREFILL,
                start_position=0,
                block_table=(0, 1),
                need_sample=True,
            ),
        ),
        step_id=0,
    )
    mixed = _cuda_input(
        (
            ScheduledRequest(
                request_id="a",
                input_token_ids=(9,),
                work_type=WorkType.DECODE,
                start_position=6,
                block_table=(0, 1),
                need_sample=True,
            ),
            ScheduledRequest(
                request_id="b",
                input_token_ids=(10, 11, 12, 13),
                work_type=WorkType.PREFILL,
                start_position=0,
                block_table=(2, 3),
                need_sample=True,
            ),
        ),
        step_id=1,
    )
    with torch.inference_mode():
        eager_runner(prefill_a)
        flashinfer_runner(prefill_a)
        eager_logits = eager_runner(mixed).logits
        flashinfer_logits = flashinfer_runner(mixed).logits
    assert eager_logits.shape[0] == 5
    _assert_attention_logits_match(flashinfer_logits, eager_logits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_decode_cuda_graph_matches_flashinfer_forward() -> None:
    torch.manual_seed(6)
    _, flashinfer_runner = _tiny_cuda_runners()
    prefill = _cuda_input(
        (
            ScheduledRequest(
                request_id="a",
                input_token_ids=(3, 4, 5, 6),
                work_type=WorkType.PREFILL,
                start_position=0,
                block_table=(0, 1),
                need_sample=True,
            ),
        ),
        step_id=0,
    )
    decode = _cuda_input(
        (
            ScheduledRequest(
                request_id="a",
                input_token_ids=(7,),
                work_type=WorkType.DECODE,
                start_position=4,
                block_table=(0, 1),
                need_sample=True,
            ),
        ),
        step_id=1,
    )
    with torch.inference_mode():
        flashinfer_runner(prefill)
        reference = flashinfer_runner.forward(decode).logits.clone()
        graphed = flashinfer_runner.try_decode_cuda_graph(decode)
        plan = ScheduledBatch(
            step_id=1,
            requests=(
                ScheduledRequest(
                    request_id="a",
                    input_token_ids=(7,),
                    work_type=WorkType.DECODE,
                    start_position=4,
                    block_table=(0, 1),
                    need_sample=True,
                ),
            ),
        )
        graphed_plan = flashinfer_runner.try_decode_cuda_graph_plan(plan)
    assert graphed is not None, getattr(flashinfer_runner.decode_graph, "_errors", {})
    assert graphed_plan is not None, getattr(flashinfer_runner.decode_graph, "_errors", {})
    _assert_attention_logits_match(graphed.logits, reference)
    _assert_attention_logits_match(graphed_plan.logits, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not flashinfer_available(), reason="FlashInfer is required")
def test_decode_graph_pack_plan_matches_from_batch() -> None:
    torch.manual_seed(7)
    _, flashinfer_runner = _tiny_cuda_runners()
    requests = (
        ScheduledRequest(
            request_id="a",
            input_token_ids=(7,),
            work_type=WorkType.DECODE,
            start_position=4,
            block_table=(0, 1),
            need_sample=True,
        ),
        ScheduledRequest(
            request_id="b",
            input_token_ids=(9,),
            work_type=WorkType.DECODE,
            start_position=5,
            block_table=(2, 3),
            need_sample=True,
        ),
    )
    plan = ScheduledBatch(step_id=1, requests=requests)
    from_batch = _cuda_input(requests, step_id=1)
    graph = flashinfer_runner.decode_graph
    assert graph is not None
    packed = graph.pack_plan(plan)
    assert packed == (select_decode_graph_bucket(2), 2)
    static = graph._inputs[packed[0]]
    torch.testing.assert_close(static.input_token_ids[:2], from_batch.input_token_ids)
    torch.testing.assert_close(static.position[:2], from_batch.position)
    torch.testing.assert_close(static.slot_mapping[:2], from_batch.slot_mapping)
    torch.testing.assert_close(static.context_lens[:2], from_batch.context_lens)
    pages = from_batch.block_tables.size(1)
    torch.testing.assert_close(
        static.block_tables[:2, :pages],
        from_batch.block_tables,
    )
    page_indptr, page_indices, last_page_len = build_paged_kv_csr(
        from_batch.block_tables,
        from_batch.context_lens,
        block_len=4,
    )
    torch.testing.assert_close(
        graph._host_kv_indptr[:3].cpu(),
        page_indptr.cpu(),
    )
    torch.testing.assert_close(
        graph._host_kv_indices[: int(page_indptr[-1])].cpu(),
        page_indices.cpu(),
    )
    torch.testing.assert_close(
        graph._host_last_page[:2].cpu(),
        last_page_len.cpu(),
    )
