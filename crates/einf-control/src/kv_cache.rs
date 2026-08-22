use sha2::{self, Digest, Sha256};
use std::collections::BTreeMap;

use crate::block_pool::{BlockId, BlockPool, SealResult};
use crate::error::ControlError;
use crate::request::RequestId;

const EMPTY_HASH_PREFIX: [u8; 32] = [
    0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KvAllocation {
    request_id: RequestId,
    blocks: Vec<BlockId>,
}

#[derive(Clone, Debug)]
pub struct ReusePlan {
    blocks: Vec<BlockId>,
    reused_len: usize,
}

impl ReusePlan {
    pub fn reused_len(&self) -> usize {
        self.reused_len
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn into_blocks(self) -> Vec<BlockId> {
        self.blocks
    }
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

    pub fn block_table_mut(&mut self, id: RequestId) -> Option<&mut KvAllocation> {
        self.allocations.get_mut(&id)
    }

    pub fn plan_reuse(&self, prompt: &[i64]) -> Option<ReusePlan> {
        debug_assert!(prompt.len() != 0);

        if prompt.len() < self.block_len {
            return None;
        }

        let mut blocks = Vec::new();
        let mut prefix = EMPTY_HASH_PREFIX;

        for chunk in prompt.chunks_exact(self.block_len) {
            let hash = hash_tokens(chunk, Some(&prefix));
            if let Some(block) = self.pool.sealed_block(&hash) {
                blocks.push(block.id());
            } else {
                break;
            }
            prefix = hash;
        }

        let max_reusable_blocks = prompt.len().saturating_sub(1) / self.block_len;

        blocks.truncate(max_reusable_blocks);

        if blocks.is_empty() {
            return None;
        }

        let reused_len = blocks.len() * self.block_len;
        Some(ReusePlan { blocks, reused_len })
    }

    pub fn seal_blocks(
        &mut self,
        id: RequestId,
        need_seal_tokens: &[i64],
        sealed_len: usize,
    ) -> Result<usize, ControlError> {
        let block_len = self.block_len;
        let table = self.allocations.get_mut(&id).unwrap();

        let last_sealed_idx = if sealed_len < block_len {
            None
        } else {
            Some((sealed_len / block_len) - 1)
        };

        let mut prefix = match last_sealed_idx {
            Some(idx) => {
                let blk = self.pool.block(table.blocks[idx]).unwrap();
                blk.assert_sealed()
            }
            None => EMPTY_HASH_PREFIX,
        };

        let mut next_idx = match last_sealed_idx {
            Some(idx) => idx + 1,
            None => 0,
        };

        let mut new_sealed_len = 0;

        for chunk in need_seal_tokens.chunks_exact(block_len) {
            let hash = hash_tokens(chunk, Some(&prefix));
            let bid = table.blocks[next_idx];

            match self.pool.seal_block(bid, hash) {
                SealResult::Published(_) => {}
                SealResult::Existing(existing) => {
                    self.pool.retain_shared(&vec![existing])?;
                    table.blocks[next_idx] = existing;
                    self.pool.release_blocks(&vec![bid])?;
                }
            }

            new_sealed_len += self.block_len;
            prefix = hash;
            next_idx += 1;
        }

        Ok(new_sealed_len)
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
        plan: Option<ReusePlan>,
        tokens: usize,
    ) -> Result<&KvAllocation, ControlError> {
        if plan.is_some() && self.allocations.contains_key(&id) {
            return Err(ControlError::InvalidReusePlan(
                "only a new running request can reuse cache".to_string(),
            ));
        }

        let tokens = match &plan {
            Some(plan) => tokens - plan.reused_len,
            None => tokens,
        };

        let required = self.required_blocks(tokens)?;

        let current = self
            .allocations
            .get(&id)
            .map(|a| a.blocks.len())
            .unwrap_or(0);

        if required > current {
            let reused = match plan {
                Some(plan) => plan.blocks,
                None => vec![],
            };

            let delta = self
                .pool
                .reuse_and_reserve(reused, required - current, id.clone())?;

            let mut existing = self
                .allocations
                .remove(&id)
                .map(|a| a.blocks)
                .unwrap_or_default();

            existing.extend(delta);

            self.allocations.insert(
                id.clone(),
                KvAllocation {
                    request_id: id.clone(),
                    blocks: existing,
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

fn hash_tokens(tokens: &[i64], prefix: Option<&[u8]>) -> [u8; 32] {
    let mut hasher = Sha256::new();

    if let Some(prefix) = prefix {
        hasher.update(prefix);
    }

    for token in tokens {
        hasher.update(token.to_le_bytes());
    }

    hasher.finalize().into()
}
