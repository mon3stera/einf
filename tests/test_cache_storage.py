import torch

from einf.cache.storage import KVCacheGeometry, TorchKVCacheStorage


def make_storage() -> TorchKVCacheStorage:
    return TorchKVCacheStorage(
        KVCacheGeometry(
            num_layers=2,
            num_blocks=3,
            block_len=2,
            num_kv_heads=2,
            head_dim=3,
        ),
        dtype=torch.float32,
        device="cpu",
    )


def make_values(token_len: int, offset: float) -> torch.Tensor:
    return (
        torch.arange(token_len * 2 * 3, dtype=torch.float32)
        .view(token_len, 2, 3)
        .add(offset)
    )


def test_write_and_read_single_slot() -> None:
    storage = make_storage()
    slot_mapping = torch.tensor([3], dtype=torch.int64)
    K = make_values(token_len=1, offset=10)
    V = make_values(token_len=1, offset=100)

    storage.write_slots(layer_idx=0, slot_mapping=slot_mapping, K=K, V=V)
    actual_K, actual_V = storage.read_slots(
        layer_idx=0,
        slot_mapping=slot_mapping,
    )

    torch.testing.assert_close(actual_K, K)
    torch.testing.assert_close(actual_V, V)
    torch.testing.assert_close(storage.K[0, 1, 1], K[0])
    torch.testing.assert_close(storage.V[0, 1, 1], V[0])


def test_noncontiguous_slots_cross_blocks_and_preserve_read_order() -> None:
    storage = make_storage()
    slot_mapping = torch.tensor([0, 1, 3, 4], dtype=torch.int64)
    K = make_values(token_len=4, offset=10)
    V = make_values(token_len=4, offset=100)

    storage.write_slots(layer_idx=1, slot_mapping=slot_mapping, K=K, V=V)

    torch.testing.assert_close(storage.K[1, 0, 0], K[0])
    torch.testing.assert_close(storage.K[1, 0, 1], K[1])
    torch.testing.assert_close(storage.K[1, 1, 1], K[2])
    torch.testing.assert_close(storage.K[1, 2, 0], K[3])

    read_order = torch.tensor([4, 0, 3, 1], dtype=torch.int64)
    actual_K, actual_V = storage.read_slots(
        layer_idx=1,
        slot_mapping=read_order,
    )

    torch.testing.assert_close(actual_K, K[[3, 0, 2, 1]])
    torch.testing.assert_close(actual_V, V[[3, 0, 2, 1]])


def test_write_slots_preserves_other_layers_and_unselected_slots() -> None:
    storage = make_storage()
    storage.K.fill_(-1)
    storage.V.fill_(-2)

    slot_mapping = torch.tensor([1, 4], dtype=torch.int64)
    K = make_values(token_len=2, offset=10)
    V = make_values(token_len=2, offset=100)

    storage.write_slots(layer_idx=1, slot_mapping=slot_mapping, K=K, V=V)

    expected_K = torch.full_like(storage.K, -1)
    expected_V = torch.full_like(storage.V, -2)
    expected_K[1].flatten(0, 1).index_copy_(0, slot_mapping, K)
    expected_V[1].flatten(0, 1).index_copy_(0, slot_mapping, V)

    torch.testing.assert_close(storage.K, expected_K)
    torch.testing.assert_close(storage.V, expected_V)

def test_gather_context_reads_noncontiguous_blocks_in_logical_order() -> None:
    storage = make_storage()
    total_slot_len = (
        storage.geometry.num_blocks
        * storage.geometry.block_len
    )
    slot_mapping = torch.arange(total_slot_len, dtype=torch.long)
    K = make_values(total_slot_len, offset=0.0)
    V = make_values(total_slot_len, offset=1000.0)
    storage.write_slots(0, slot_mapping, K, V)

    actual_K, actual_V = storage.gather_context(
        0,
        block_tables=(2, 0),
        context_len=3,
    )

    expected_slots = torch.tensor([4, 5, 0], dtype=torch.long)
    torch.testing.assert_close(actual_K, K.index_select(0, expected_slots))
    torch.testing.assert_close(actual_V, V.index_select(0, expected_slots))


def test_gather_context_combines_incremental_writes() -> None:
    storage = make_storage()
    first_slots = torch.tensor([4, 5], dtype=torch.long)
    second_slots = torch.tensor([0], dtype=torch.long)
    first_K = make_values(2, offset=10.0)
    first_V = make_values(2, offset=110.0)
    second_K = make_values(1, offset=20.0)
    second_V = make_values(1, offset=120.0)

    storage.write_slots(1, first_slots, first_K, first_V)
    storage.write_slots(1, second_slots, second_K, second_V)

    actual_K, actual_V = storage.gather_context(
        1,
        block_tables=(2, 0),
        context_len=3,
    )

    torch.testing.assert_close(
        actual_K,
        torch.cat((first_K, second_K)),
    )
    torch.testing.assert_close(
        actual_V,
        torch.cat((first_V, second_V)),
    )
