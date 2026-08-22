use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::error::ControlError;
use crate::RequestId;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "block-{}", self.0)
    }
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum BlockState {
    Writable { writer_rid: Option<RequestId> },
    Sealed { hash: [u8; 32] },
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PhysicalBlock {
    id: BlockId,
    state: BlockState,
    ref_cnt: usize,
}

impl PhysicalBlock {
    fn new(id: BlockId) -> Self {
        Self {
            id,
            state: BlockState::Writable { writer_rid: None },
            ref_cnt: 0,
        }
    }

    pub fn id(&self) -> BlockId {
        self.id
    }

    pub fn is_sealed(&self) -> bool {
        matches!(self.state, BlockState::Sealed { .. })
    }

    pub fn seal(&mut self, hash: [u8; 32]) {
        assert!(!self.is_sealed());
        self.state = BlockState::Sealed { hash };
    }

    pub fn reset(&mut self) {
        self.state = BlockState::Writable { writer_rid: None };
    }

    pub fn assert_set_writer(&mut self, writer: RequestId) {
        match &mut self.state {
            BlockState::Writable { writer_rid } => {
                *writer_rid = Some(writer);
            }
            _ => panic!("{} must be writable", self.id),
        }
    }

    pub fn assert_sealed(&self) -> [u8; 32] {
        match &self.state {
            BlockState::Sealed { hash } => hash.clone(),
            _ => panic!("{} must be sealed", self.id),
        }
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

#[derive(Debug, Clone, Copy)]
pub enum SealResult {
    Published(BlockId),
    Existing(BlockId),
}

#[derive(Clone, Debug)]
pub struct BlockPool {
    capacity: usize,
    blocks: Vec<PhysicalBlock>,
    free: BTreeSet<BlockId>,
    allocated: BTreeSet<BlockId>,
    sealed: BTreeMap<[u8; 32], BlockId>,
}

impl BlockPool {
    pub fn new(num_blocks: usize) -> Self {
        let blocks = (0..num_blocks as u32)
            .map(|e| PhysicalBlock::new(BlockId(e)))
            .collect();
        let free = (0..num_blocks as u32).map(BlockId).collect();

        Self {
            capacity: num_blocks,
            blocks,
            free,
            allocated: BTreeSet::new(),
            sealed: BTreeMap::new(),
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

    pub fn reserve(
        &mut self,
        count: usize,
        writer: RequestId,
    ) -> Result<BlockReservation, ControlError> {
        if !self.can_reserve(count) {
            return Err(ControlError::InsufficientBlocks {
                requested: count,
                available: self.available(),
            });
        }

        let reserved_ids: Vec<BlockId> = self.free.iter().take(count).copied().collect();

        for &bid in &reserved_ids {
            self.free.remove(&bid);
            self.allocated.insert(bid);
            let block = &mut self.blocks[bid.0 as usize];
            block.ref_cnt += 1;
            block.assert_set_writer(writer.clone());
        }

        Ok(BlockReservation {
            blocks: reserved_ids,
        })
    }

    pub fn seal_block(&mut self, id: BlockId, hash: [u8; 32]) -> SealResult {
        match self.sealed.get(&hash) {
            Some(existing) if *existing == id => return SealResult::Published(id),
            Some(existing) => return SealResult::Existing(*existing),
            None => {
                let block = self.block_mut(id).unwrap();
                block.seal(hash);
                self.sealed.insert(hash, id);
                return SealResult::Published(id);
            }
        }
    }

    pub fn sealed_block(&self, hash: &[u8; 32]) -> Option<&PhysicalBlock> {
        let id = self.sealed.get(hash)?;
        self.block(*id)
    }

    pub fn release(&mut self, reservation: BlockReservation) -> Result<(), ControlError> {
        self.release_blocks(&reservation.blocks)
    }

    pub(crate) fn release_blocks(&mut self, blocks: &[BlockId]) -> Result<(), ControlError> {
        for &bid in blocks {
            if !self.allocated.contains(&bid) {
                return Err(ControlError::BlockNotInUse(bid));
            }

            let block = self.blocks.get_mut(bid.0 as usize).unwrap();
            block.ref_cnt -= 1;

            if block.ref_cnt == 0 {
                if block.is_sealed() {
                    self.sealed.remove(&block.assert_sealed());
                }

                block.reset();
                self.free.insert(bid);
                self.allocated.remove(&bid);
            }
        }
        Ok(())
    }

    pub fn reuse_and_reserve(
        &mut self,
        reused: Vec<BlockId>,
        need: usize,
        writer: RequestId,
    ) -> Result<Vec<BlockId>, ControlError> {
        let reserved = self.reserve(need, writer)?;
        if let Err(e) = self.retain_shared(&reused) {
            self.release(reserved)?;
            return Err(e);
        };
        let mut delta = reused;
        delta.extend(reserved.blocks);
        Ok(delta)
    }

    pub fn retain_shared(&mut self, ids: &[BlockId]) -> Result<(), ControlError> {
        for &id in ids {
            let block = self.block_mut(id).ok_or(ControlError::UnknownBlock(id))?;

            if !block.is_sealed() {
                return Err(ControlError::UnsealedBlock(id));
            }
        }

        for &id in ids {
            let block = self.block_mut(id).unwrap();

            block.ref_cnt += 1;

            if self.free.contains(&id) {
                self.free.remove(&id);
                self.allocated.insert(id);
            }
        }

        Ok(())
    }

    pub fn block(&self, id: BlockId) -> Option<&PhysicalBlock> {
        self.blocks.get(id.0 as usize)
    }

    pub fn block_mut(&mut self, id: BlockId) -> Option<&mut PhysicalBlock> {
        self.blocks.get_mut(id.0 as usize)
    }
}
