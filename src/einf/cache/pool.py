"""Fixed-capacity logical block pool."""

from collections import deque
from dataclasses import dataclass

@dataclass
class PhysicalTokenBlock:
    id: int


class BlockReservation:
    def __init__(self, blocks: tuple[PhysicalTokenBlock, ...], pool: "BlockPool"):
        self.blocks = blocks
        self.pool = pool

    def commit(self, request_id: str) -> tuple[PhysicalTokenBlock, ...]:
        existing = self.pool.owners.get(request_id, ())
        self.pool.owners[request_id] = existing + self.blocks
        return tuple(self.blocks)

    def rollback(self) -> None:
        for block in self.blocks:
            self.pool.free_list.append(block)


class BlockPool:
    def __init__(self, num_blocks: int) -> None:
        self.free_list = deque()
        self.owners = {}
        self.capacity_len = num_blocks

        for i in range(num_blocks):
            self.free_list.append(PhysicalTokenBlock(id=i))

    @property
    def capacity(self) -> int:
        return self.capacity_len

    def free_len(self) -> int:
        return len(self.free_list)

    def reserve(self, num: int) -> BlockReservation | None:
        if self.free_len() < num:
            return None

        blocks = []

        for _ in range(num):
            blocks.append(self.free_list.popleft())

        return BlockReservation(blocks=tuple(blocks), pool=self)

    def free(self, request_id: str) -> None:
        if request_id in self.owners:
            for block in self.owners[request_id]:
                self.free_list.append(block)
            del self.owners[request_id]
