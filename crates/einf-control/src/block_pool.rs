use std::collections::BTreeSet;
use std::fmt;

use crate::error::ControlError;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockId(pub u32);
impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "block-{}", self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockReservation {
    blocks: Vec<BlockId>,
}
impl BlockReservation {
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }
}

#[derive(Clone, Debug)]
pub struct BlockPool {
    capacity: usize,
    free: BTreeSet<BlockId>,
    allocated: BTreeSet<BlockId>,
}
impl BlockPool {
    pub fn new(num_blocks: usize) -> Self {
        let free = (0..num_blocks as u32).map(BlockId).collect();
        Self {
            capacity: num_blocks,
            free,
            allocated: BTreeSet::new(),
        }
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn available(&self) -> usize {
        self.free.len()
    }
    pub fn allocated(&self) -> usize {
        self.allocated.len()
    }
    pub fn can_reserve(&self, count: usize) -> bool {
        count <= self.available()
    }
    pub fn reserve(&mut self, count: usize) -> Result<BlockReservation, ControlError> {
        if !self.can_reserve(count) {
            return Err(ControlError::InsufficientBlocks {
                requested: count,
                available: self.available(),
            });
        }
        let blocks: Vec<_> = self.free.iter().take(count).copied().collect();
        for block in &blocks {
            self.free.remove(block);
            self.allocated.insert(*block);
        }
        Ok(BlockReservation { blocks })
    }
    pub fn release(&mut self, reservation: BlockReservation) -> Result<(), ControlError> {
        self.release_blocks(&reservation.blocks)
    }
    pub(crate) fn release_blocks(&mut self, blocks: &[BlockId]) -> Result<(), ControlError> {
        for block in blocks {
            if !self.allocated.remove(block) {
                return Err(ControlError::BlockNotInUse(*block));
            }
            self.free.insert(*block);
        }
        Ok(())
    }
}
