from einf.cache.manager import KVCacheManager
from einf.cache.pool import BlockPool


def test_allocate_creates_block_table_and_translates_token_positions() -> None:
    pool = BlockPool(num_blocks=3)
    manager = KVCacheManager(pool, block_len=4)

    assert manager.allocate("req-1", required_cache_len=3)

    assert manager.block_table("req-1") == (0,)
    assert manager.translate("req-1", 0) == (0, 0)
    assert manager.translate("req-1", 3) == (0, 3)
    assert pool.free_len() == 2


def test_allocate_grows_capacity_without_reallocating_existing_blocks() -> None:
    pool = BlockPool(num_blocks=3)
    manager = KVCacheManager(pool, block_len=4)

    assert manager.allocate("req-1", required_cache_len=3)
    assert manager.allocate("req-1", required_cache_len=5)

    assert manager.block_table("req-1") == (0, 1)
    assert manager.translate("req-1", 4) == (1, 0)
    assert pool.free_len() == 1

    assert manager.allocate("req-1", required_cache_len=5)
    assert manager.block_table("req-1") == (0, 1)
    assert pool.free_len() == 1


def test_allocate_oom_keeps_existing_block_table_unchanged() -> None:
    pool = BlockPool(num_blocks=2)
    manager = KVCacheManager(pool, block_len=4)

    assert manager.allocate("req-1", required_cache_len=5)
    table_before = manager.block_table("req-1")

    assert not manager.allocate("req-1", required_cache_len=9)
    assert manager.block_table("req-1") == table_before
    assert pool.free_len() == 0


def test_release_returns_request_blocks_to_pool() -> None:
    pool = BlockPool(num_blocks=2)
    manager = KVCacheManager(pool, block_len=4)

    assert manager.allocate("req-1", required_cache_len=5)

    manager.release("req-1")

    assert manager.block_table("req-1") == ()
    assert not manager.has_block_table("req-1")
    assert pool.free_len() == 2
