import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage
from einf.executors.torch.input import ModelInput
from einf.scheduler import ScheduledBatch, ScheduledRequest, WorkType


def test_model_input_writes_current_tokens_into_request_contexts() -> None:
    geometry = KVCacheGeometry(
        num_layers=1,
        num_blocks=5,
        block_len=2,
        num_kv_heads=1,
        head_dim=2,
    )
    storage = TorchKVCacheStorage(
        geometry,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    batch = ScheduledBatch(
        step_id=10,
        requests=(
            ScheduledRequest(
                request_id="a",
                input_token_ids=(10, 11),
                start_position=1,
                block_table=(3, 1),
                work_type=WorkType.PREFILL,
                need_sample=False,
            ),
            ScheduledRequest(
                request_id="b",
                input_token_ids=(20,),
                start_position=0,
                block_table=(4,),
                work_type=WorkType.DECODE,
                need_sample=True,
            ),
        ),
    )
    model_input = ModelInput.from_batch(
        batch,
        block_len=geometry.block_len,
        device=torch.device("cpu"),
    )
    current_K = torch.tensor(
        [[[10.0, 10.5]], [[11.0, 11.5]], [[20.0, 20.5]]],
    )
    current_V = current_K + 1000.0
    history_slot = torch.tensor([6], dtype=torch.long)
    history_K = torch.tensor([[[9.0, 9.5]]])
    history_V = history_K + 1000.0

    storage.write_slots(0, history_slot, history_K, history_V)
    storage.write_slots(
        0,
        model_input.slot_mapping,
        current_K,
        current_V,
    )

    request_a_K, request_a_V = storage.gather_context(
        0,
        block_tables=batch.requests[0].block_table,
        context_len=3,
    )
    request_b_K, request_b_V = storage.gather_context(
        0,
        block_tables=batch.requests[1].block_table,
        context_len=1,
    )

    torch.testing.assert_close(
        request_a_K,
        torch.cat((history_K, current_K[:2])),
    )
    torch.testing.assert_close(
        request_a_V,
        torch.cat((history_V, current_V[:2])),
    )
    torch.testing.assert_close(request_b_K, current_K[2:])
    torch.testing.assert_close(request_b_V, current_V[2:])
