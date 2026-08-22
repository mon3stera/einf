use std::cmp::Ordering;

use crate::execution_plan::WorkType;
use crate::request::{RequestId, RequestState};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PolicyRequest {
    pub id: RequestId,
    pub arrival_index: u64,
    pub state: RequestState,
    pub work_type: WorkType,
}

pub trait SchedulingPolicy {
    fn compare(&self, left: &PolicyRequest, right: &PolicyRequest) -> Ordering;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Fcfs;
impl SchedulingPolicy for Fcfs {
    fn compare(&self, left: &PolicyRequest, right: &PolicyRequest) -> Ordering {
        left.arrival_index
            .cmp(&right.arrival_index)
            .then(left.id.cmp(&right.id))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DecodeFirst;
impl SchedulingPolicy for DecodeFirst {
    fn compare(&self, left: &PolicyRequest, right: &PolicyRequest) -> Ordering {
        let work = |kind: WorkType| if kind == WorkType::Decode { 0u8 } else { 1u8 };
        work(left.work_type)
            .cmp(&work(right.work_type))
            .then(left.arrival_index.cmp(&right.arrival_index))
            .then(left.id.cmp(&right.id))
    }
}
