use std::fmt;

use crate::block_pool::BlockId;
use crate::execution::BatchId;
use crate::request::{RequestId, RequestState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ControlError {
    InvalidConfig(&'static str),
    DuplicateRequest(RequestId),
    UnknownRequest(RequestId),
    UnknownBlock(BlockId),
    InvalidState {
        request_id: RequestId,
        expected: &'static str,
        actual: RequestState,
    },
    RequestInvariant {
        request_id: RequestId,
        message: String,
    },
    InsufficientBlocks {
        requested: usize,
        available: usize,
    },
    BlockNotInUse(BlockId),
    BlockAlreadyInUse(BlockId),
    ArithmeticOverflow,
    NoOutstandingBatch,
    StaleBatch {
        expected: BatchId,
        actual: BatchId,
    },
    InvalidExecutionResult(String),
    RequestInFlight(RequestId),
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::DuplicateRequest(id) => write!(f, "request already exists: {id}"),
            Self::UnknownRequest(id) => write!(f, "unknown request: {id}"),
            Self::UnknownBlock(id) => write!(f, "unknown block: {id}"),
            Self::InvalidState {
                request_id,
                expected,
                actual,
            } => write!(f, "request {request_id} is {actual:?}, expected {expected}"),
            Self::RequestInvariant {
                request_id,
                message,
            } => write!(f, "request {request_id} invariant violation: {message}"),
            Self::InsufficientBlocks {
                requested,
                available,
            } => write!(
                f,
                "insufficient blocks: requested {requested}, available {available}"
            ),
            Self::BlockNotInUse(id) => write!(f, "block is not allocated: {id}"),
            Self::BlockAlreadyInUse(id) => write!(f, "block is already allocated: {id}"),
            Self::ArithmeticOverflow => write!(f, "arithmetic overflow"),
            Self::NoOutstandingBatch => write!(f, "there is no outstanding batch"),
            Self::StaleBatch { expected, actual } => {
                write!(f, "stale batch: expected {:?}, got {:?}", expected, actual)
            }
            Self::InvalidExecutionResult(message) => {
                write!(f, "invalid execution result: {message}")
            }
            Self::RequestInFlight(id) => write!(f, "request is in flight: {id}"),
        }
    }
}

impl std::error::Error for ControlError {}
