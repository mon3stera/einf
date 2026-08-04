"""Request-to-block-table mapping and logical cache growth."""

from einf.cache.pool import BlockPool

class KVCacheManager:
    def __init__(self, pool: BlockPool, block_len: int) -> None:
        self.pool = pool
        self.block_len = block_len

    @property
    def capacity(self) -> int:
        return self.pool.capacity_len * self.block_len

    def block_table(self, request_id: str) -> tuple[int, ...]:
        blocks = self.pool.owners.get(request_id, ())
        return tuple(block.id for block in blocks)

    def has_block_table(self, request_id: str) -> bool:
        return request_id in self.pool.owners

    def _required_blocks(self, required_cache_len: int) -> int:
        return (required_cache_len + self.block_len - 1) // self.block_len;

    def _ensure_capacity(self, request_id: str, required_cache_len: int) -> bool:
        required_blocks = self._required_blocks(required_cache_len)

        current_blocks = len(self.block_table(request_id))
        additional_blocks = required_blocks - current_blocks

        if additional_blocks <= 0:
            return True

        reservation = self.pool.reserve(additional_blocks)

        if reservation is None:
            return False

        reservation.commit(request_id)
        return True

    def translate(self, request_id: str, nth_token: int) -> tuple[int, int] | None:
        if not self.has_block_table(request_id):
            return None

        table = self.block_table(request_id)

        logical_index = nth_token // self.block_len

        if logical_index >= len(table):
            raise RuntimeError("KV cache capacity was not ensured before address translation")

        physical_index = table[logical_index]
        slot_offset = nth_token % self.block_len

        return (physical_index, slot_offset)

    def allocate(self, request_id: str, required_cache_len: int) -> bool:
        return self._ensure_capacity(request_id, required_cache_len)

    def release(self, request_id: str) -> None:
        self.pool.free(request_id)
