from einf.cache.pool import BlockPool


def block_ids(blocks: object) -> list[int]:
    return [block.id for block in blocks]


def test_reserve_commit_and_free_restore_all_blocks() -> None:
    pool = BlockPool(num_blocks=4)

    reservation = pool.reserve(2)

    assert reservation is not None
    assert block_ids(reservation.blocks) == [0, 1]
    assert pool.free_len() == 2
    assert pool.owners == {}

    committed = reservation.commit("req-1")

    assert block_ids(committed) == [0, 1]
    assert block_ids(pool.owners["req-1"]) == [0, 1]
    assert pool.free_len() == 2

    pool.free("req-1")

    assert pool.free_len() == 4
    assert "req-1" not in pool.owners
    assert sorted(block_ids(pool.free_list)) == [0, 1, 2, 3]


def test_rollback_returns_reserved_blocks_to_pool() -> None:
    pool = BlockPool(num_blocks=4)
    reservation = pool.reserve(2)

    assert reservation is not None
    assert pool.free_len() == 2

    reservation.rollback()

    assert pool.free_len() == 4
    assert pool.owners == {}
    assert sorted(block_ids(pool.free_list)) == [0, 1, 2, 3]


def test_reserve_is_all_or_nothing_when_capacity_is_insufficient() -> None:
    pool = BlockPool(num_blocks=2)
    free_blocks_before = tuple(pool.free_list)

    reservation = pool.reserve(3)

    assert reservation is None
    assert tuple(pool.free_list) == free_blocks_before
    assert pool.free_len() == 2
    assert pool.owners == {}


def test_multiple_commits_append_blocks_for_same_request() -> None:
    pool = BlockPool(num_blocks=4)

    first = pool.reserve(1)
    assert first is not None
    first.commit("req-1")

    second = pool.reserve(2)
    assert second is not None
    second.commit("req-1")

    assert block_ids(pool.owners["req-1"]) == [0, 1, 2]
    assert pool.free_len() == 1

    pool.free("req-1")

    assert pool.free_len() == 4
    assert "req-1" not in pool.owners
    assert sorted(block_ids(pool.free_list)) == [0, 1, 2, 3]
