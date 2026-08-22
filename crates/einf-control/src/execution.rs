use crate::request::RequestId;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BatchId(pub u64);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestExecutionResult {
    pub request_id: RequestId,
    pub generated_token_ids: Vec<i64>,
    pub cached_len_delta: usize,
    pub is_eos: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionResult {
    pub step_id: BatchId,
    pub request_results: Vec<RequestExecutionResult>,
}
