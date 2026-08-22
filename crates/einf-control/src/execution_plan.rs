use crate::block_pool::BlockId;
use crate::execution::BatchId;
use crate::request::RequestId;
use crate::sample::SamplingPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkType {
    Prefill,
    Decode,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledRequest {
    pub request_id: RequestId,
    pub input_token_ids: Vec<i64>,
    pub work_type: WorkType,
    pub start_position: usize,
    pub block_table: Vec<BlockId>,
    pub need_sample: bool,
    pub sampling_plan: Option<SamplingPlan>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchPlan {
    pub step_id: BatchId,
    pub requests: Vec<ScheduledRequest>,
}

pub type ScheduledBatch = BatchPlan;
