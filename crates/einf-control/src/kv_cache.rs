use std::collections::BTreeMap;

use crate::block_pool::{BlockId, BlockPool};
use crate::error::ControlError;
use crate::request::RequestId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvAllocation {
    request_id: RequestId,
    blocks: Vec<BlockId>,
}
impl KvAllocation {
    pub fn request_id(&self) -> RequestId {
        self.request_id.clone()
    }
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }
}

#[derive(Clone, Debug)]
pub struct KvCacheManager {
    block_len: usize,
    pool: BlockPool,
    allocations: BTreeMap<RequestId, KvAllocation>,
}
impl KvCacheManager {
    pub fn new(num_blocks: usize, block_len: usize) -> Result<Self, ControlError> {
        if block_len == 0 {
            return Err(ControlError::InvalidConfig("block length must be positive"));
        }
        Ok(Self {
            block_len,
            pool: BlockPool::new(num_blocks),
            allocations: BTreeMap::new(),
        })
    }
    pub fn block_len(&self) -> usize {
        self.block_len
    }
    pub fn free_blocks(&self) -> usize {
        self.pool.available()
    }
    pub fn capacity_tokens(&self) -> usize {
        self.pool.capacity().saturating_mul(self.block_len)
    }
    pub fn allocation(&self, id: RequestId) -> Option<&KvAllocation> {
        self.allocations.get(&id)
    }
    pub fn block_table(&self, id: RequestId) -> Vec<BlockId> {
        self.allocations
            .get(&id)
            .map(|a| a.blocks.clone())
            .unwrap_or_default()
    }
    fn required_blocks(&self, tokens: usize) -> Result<usize, ControlError> {
        if tokens == 0 {
            return Ok(0);
        }
        tokens
            .checked_add(self.block_len - 1)
            .map(|v| v / self.block_len)
            .ok_or(ControlError::ArithmeticOverflow)
    }
    pub fn reserve_to(
        &mut self,
        id: RequestId,
        tokens: usize,
    ) -> Result<&KvAllocation, ControlError> {
        let required = self.required_blocks(tokens)?;
        let current = self
            .allocations
            .get(&id)
            .map(|a| a.blocks.len())
            .unwrap_or(0);
        if required > current {
            let reservation = self.pool.reserve(required - current)?;
            let mut blocks = self
                .allocations
                .remove(&id)
                .map(|a| a.blocks)
                .unwrap_or_default();
            blocks.extend(reservation.blocks().iter().copied());
            self.allocations.insert(
                id.clone(),
                KvAllocation {
                    request_id: id.clone(),
                    blocks,
                },
            );
        }
        Ok(self
            .allocations
            .get(&id)
            .expect("allocation inserted for required blocks"))
    }
    pub fn release(&mut self, id: RequestId) -> Result<(), ControlError> {
        if let Some(allocation) = self.allocations.remove(&id) {
            self.pool.release_blocks(&allocation.blocks)?;
        }
        Ok(())
    }
}
