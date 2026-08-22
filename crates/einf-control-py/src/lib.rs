use einf_control as core;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn control_error(error: core::ControlError) -> PyErr {
    match error {
        core::ControlError::InvalidConfig(_)
        | core::ControlError::DuplicateRequest(_)
        | core::ControlError::UnknownRequest(_)
        | core::ControlError::InvalidExecutionResult(_)
        | core::ControlError::InvalidReusePlan(_) => PyValueError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

#[pyclass(frozen, eq, eq_int, from_py_object, module = "einf._control")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorkType {
    Prefill = 1,
    Decode = 2,
}

impl From<core::WorkType> for WorkType {
    fn from(value: core::WorkType) -> Self {
        match value {
            core::WorkType::Prefill => Self::Prefill,
            core::WorkType::Decode => Self::Decode,
        }
    }
}

impl From<WorkType> for core::WorkType {
    fn from(value: WorkType) -> Self {
        match value {
            WorkType::Prefill => Self::Prefill,
            WorkType::Decode => Self::Decode,
        }
    }
}

#[pyclass(frozen, eq, eq_int, from_py_object, module = "einf._control")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RequestState {
    Waiting = 1,
    Running = 2,
    Finished = 3,
    Cancelled = 4,
    Failed = 5,
}

impl From<core::RequestState> for RequestState {
    fn from(value: core::RequestState) -> Self {
        match value {
            core::RequestState::Waiting => Self::Waiting,
            core::RequestState::Running => Self::Running,
            core::RequestState::Finished => Self::Finished,
            core::RequestState::Cancelled => Self::Cancelled,
            core::RequestState::Failed => Self::Failed,
        }
    }
}

#[pyclass(frozen, eq, eq_int, from_py_object, module = "einf._control")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompletionReason {
    Eos = 1,
    Length = 2,
}

impl From<core::CompletionReason> for CompletionReason {
    fn from(value: core::CompletionReason) -> Self {
        match value {
            core::CompletionReason::Eos => Self::Eos,
            core::CompletionReason::Length => Self::Length,
        }
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    inner: core::SamplingParams,
}

impl From<core::SamplingParams> for SamplingParams {
    fn from(inner: core::SamplingParams) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (
        temperature=0.0,
        *,
        top_k=None,
        top_p=1.0,
        min_p=0.0,
        seed=0,
        stop_token_ids=Vec::new(),
        num_logprobs=0,
    ))]
    fn new(
        temperature: f32,
        top_k: Option<u32>,
        top_p: f32,
        min_p: f32,
        seed: u64,
        stop_token_ids: Vec<i64>,
        num_logprobs: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: core::SamplingParams::new(
                temperature,
                top_k,
                top_p,
                min_p,
                seed,
                stop_token_ids,
                num_logprobs,
            )
            .map_err(control_error)?,
        })
    }

    #[getter]
    fn temperature(&self) -> f32 {
        self.inner.temperature()
    }

    #[getter]
    fn is_greedy(&self) -> bool {
        self.inner.is_greedy()
    }

    #[getter]
    fn top_k(&self) -> Option<u32> {
        self.inner.top_k
    }

    #[getter]
    fn top_p(&self) -> f32 {
        self.inner.top_p
    }

    #[getter]
    fn min_p(&self) -> f32 {
        self.inner.min_p
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    #[getter]
    fn stop_token_ids(&self) -> Vec<i64> {
        self.inner.stop_token_ids.clone()
    }

    #[getter]
    fn num_logprobs(&self) -> usize {
        self.inner.num_logprobs
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingPlan {
    params: SamplingParams,
    sample_index: u64,
}

impl From<core::SamplingPlan> for SamplingPlan {
    fn from(value: core::SamplingPlan) -> Self {
        Self {
            params: value.params.into(),
            sample_index: value.sample_index,
        }
    }
}

impl From<&SamplingPlan> for core::SamplingPlan {
    fn from(value: &SamplingPlan) -> Self {
        Self {
            params: value.params.inner.clone(),
            sample_index: value.sample_index,
        }
    }
}

#[pymethods]
impl SamplingPlan {
    #[new]
    fn new(params: PyRef<'_, SamplingParams>, sample_index: u64) -> Self {
        Self {
            params: SamplingParams {
                inner: params.inner.clone(),
            },
            sample_index,
        }
    }

    #[getter]
    fn params(&self) -> SamplingParams {
        self.params.clone()
    }

    #[getter]
    fn sample_index(&self) -> u64 {
        self.sample_index
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct RequestSpec {
    request_id: String,
    prompt_token_ids: Vec<i64>,
    max_new_len: usize,
    sampling_params: SamplingParams,
}

#[pymethods]
impl RequestSpec {
    #[new]
    #[pyo3(signature = (request_id, prompt_token_ids, max_new_len, sampling_params=None))]
    fn new(
        request_id: String,
        prompt_token_ids: Vec<i64>,
        max_new_len: usize,
        sampling_params: Option<PyRef<'_, SamplingParams>>,
    ) -> Self {
        Self {
            request_id,
            prompt_token_ids,
            max_new_len,
            sampling_params: SamplingParams {
                inner: sampling_params
                    .as_deref()
                    .map(|params| params.inner.clone())
                    .unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn request_id(&self) -> &str {
        &self.request_id
    }

    #[getter]
    fn prompt_token_ids(&self) -> Vec<i64> {
        self.prompt_token_ids.clone()
    }

    #[getter]
    fn max_new_len(&self) -> usize {
        self.max_new_len
    }

    #[getter]
    fn sampling_params(&self) -> SamplingParams {
        self.sampling_params.clone()
    }
}

impl From<&RequestSpec> for core::RequestSpec {
    fn from(value: &RequestSpec) -> Self {
        Self {
            request_id: core::RequestId::new(value.request_id.clone()),
            prompt_token_ids: value.prompt_token_ids.clone(),
            max_new_len: value.max_new_len,
            sampling_params: value.sampling_params.inner.clone(),
        }
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct ScheduledRequest {
    request_id: String,
    input_token_ids: Vec<i64>,
    work_type: WorkType,
    start_position: usize,
    block_table: Vec<u32>,
    need_sample: bool,
    sampling_plan: Option<SamplingPlan>,
}

#[pymethods]
impl ScheduledRequest {
    #[new]
    #[pyo3(signature = (
        request_id,
        input_token_ids,
        work_type,
        start_position,
        block_table,
        need_sample,
        sampling_plan=None,
    ))]
    fn new(
        request_id: String,
        input_token_ids: Vec<i64>,
        work_type: WorkType,
        start_position: usize,
        block_table: Vec<u32>,
        need_sample: bool,
        sampling_plan: Option<PyRef<'_, SamplingPlan>>,
    ) -> PyResult<Self> {
        let sampling_plan = sampling_plan.as_deref().cloned();
        if !need_sample && sampling_plan.is_some() {
            return Err(PyValueError::new_err(
                "sampling_plan cannot be present when need_sample is false",
            ));
        }
        Ok(Self {
            request_id,
            input_token_ids,
            work_type,
            start_position,
            block_table,
            need_sample,
            sampling_plan,
        })
    }

    #[getter]
    fn request_id(&self) -> &str {
        &self.request_id
    }

    #[getter]
    fn input_token_ids(&self) -> Vec<i64> {
        self.input_token_ids.clone()
    }

    #[getter]
    fn work_type(&self) -> WorkType {
        self.work_type
    }

    #[getter]
    fn start_position(&self) -> usize {
        self.start_position
    }

    #[getter]
    fn block_table(&self) -> Vec<u32> {
        self.block_table.clone()
    }

    #[getter]
    fn need_sample(&self) -> bool {
        self.need_sample
    }

    #[getter]
    fn sampling_plan(&self) -> Option<SamplingPlan> {
        self.sampling_plan.clone()
    }
}

impl From<core::ScheduledRequest> for ScheduledRequest {
    fn from(value: core::ScheduledRequest) -> Self {
        Self {
            request_id: value.request_id.0,
            input_token_ids: value.input_token_ids,
            work_type: value.work_type.into(),
            start_position: value.start_position,
            block_table: value.block_table.into_iter().map(|id| id.0).collect(),
            need_sample: value.need_sample,
            sampling_plan: value.sampling_plan.map(Into::into),
        }
    }
}

impl From<&ScheduledRequest> for core::ScheduledRequest {
    fn from(value: &ScheduledRequest) -> Self {
        Self {
            request_id: core::RequestId::new(value.request_id.clone()),
            input_token_ids: value.input_token_ids.clone(),
            work_type: value.work_type.into(),
            start_position: value.start_position,
            block_table: value
                .block_table
                .iter()
                .copied()
                .map(core::BlockId)
                .collect(),
            need_sample: value.need_sample,
            sampling_plan: value.sampling_plan.as_ref().map(Into::into),
        }
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct BatchPlan {
    step_id: u64,
    requests: Vec<ScheduledRequest>,
}

#[pymethods]
impl BatchPlan {
    #[new]
    fn new(step_id: u64, requests: Vec<ScheduledRequest>) -> Self {
        Self { step_id, requests }
    }

    #[getter]
    fn step_id(&self) -> u64 {
        self.step_id
    }

    #[getter]
    fn requests(&self) -> Vec<ScheduledRequest> {
        self.requests.clone()
    }
}

impl From<core::BatchPlan> for BatchPlan {
    fn from(value: core::BatchPlan) -> Self {
        Self {
            step_id: value.step_id.0,
            requests: value.requests.into_iter().map(Into::into).collect(),
        }
    }
}

impl From<&BatchPlan> for core::BatchPlan {
    fn from(value: &BatchPlan) -> Self {
        Self {
            step_id: core::BatchId(value.step_id),
            requests: value.requests.iter().map(Into::into).collect(),
        }
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct RequestExecutionResult {
    request_id: String,
    generated_token_ids: Vec<i64>,
    cached_len_delta: usize,
    is_eos: bool,
}

#[pymethods]
impl RequestExecutionResult {
    #[new]
    fn new(
        request_id: String,
        generated_token_ids: Vec<i64>,
        cached_len_delta: usize,
        is_eos: bool,
    ) -> Self {
        Self {
            request_id,
            generated_token_ids,
            cached_len_delta,
            is_eos,
        }
    }

    #[getter]
    fn request_id(&self) -> &str {
        &self.request_id
    }

    #[getter]
    fn generated_token_ids(&self) -> Vec<i64> {
        self.generated_token_ids.clone()
    }

    #[getter]
    fn cached_len_delta(&self) -> usize {
        self.cached_len_delta
    }

    #[getter]
    fn is_eos(&self) -> bool {
        self.is_eos
    }
}

impl From<&RequestExecutionResult> for core::RequestExecutionResult {
    fn from(value: &RequestExecutionResult) -> Self {
        Self {
            request_id: core::RequestId::new(value.request_id.clone()),
            generated_token_ids: value.generated_token_ids.clone(),
            cached_len_delta: value.cached_len_delta,
            is_eos: value.is_eos,
        }
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionResult {
    step_id: u64,
    request_results: Vec<RequestExecutionResult>,
}

#[pymethods]
impl ExecutionResult {
    #[new]
    fn new(step_id: u64, request_results: Vec<RequestExecutionResult>) -> Self {
        Self {
            step_id,
            request_results,
        }
    }

    #[getter]
    fn step_id(&self) -> u64 {
        self.step_id
    }

    #[getter]
    fn request_results(&self) -> Vec<RequestExecutionResult> {
        self.request_results.clone()
    }
}

impl From<&ExecutionResult> for core::ExecutionResult {
    fn from(value: &ExecutionResult) -> Self {
        Self {
            step_id: core::BatchId(value.step_id),
            request_results: value.request_results.iter().map(Into::into).collect(),
        }
    }
}

#[pyclass(frozen, eq, from_py_object, module = "einf._control")]
#[derive(Clone, Debug, PartialEq)]
pub struct RequestView {
    request_id: String,
    prompt_token_ids: Vec<i64>,
    arrival_index: u64,
    max_new_len: usize,
    sampling_params: SamplingParams,
    sample_index: u64,
    generated_token_ids: Vec<i64>,
    state: RequestState,
    cached_len: usize,
    completion_reason: Option<CompletionReason>,
    error: Option<String>,
}

impl From<&core::Request> for RequestView {
    fn from(value: &core::Request) -> Self {
        Self {
            request_id: value.request_id().0,
            prompt_token_ids: value.prompt_token_ids().to_vec(),
            arrival_index: value.arrival_index(),
            max_new_len: value.max_new_len(),
            sampling_params: value.sampling_params().clone().into(),
            sample_index: value.sample_index(),
            generated_token_ids: value.generated_token_ids().to_vec(),
            state: value.state().into(),
            cached_len: value.cached_len(),
            completion_reason: value.completion_reason().map(Into::into),
            error: value.error().map(str::to_owned),
        }
    }
}

#[pymethods]
impl RequestView {
    #[getter]
    fn request_id(&self) -> &str {
        &self.request_id
    }
    #[getter]
    fn prompt_token_ids(&self) -> Vec<i64> {
        self.prompt_token_ids.clone()
    }
    #[getter]
    fn arrival_index(&self) -> u64 {
        self.arrival_index
    }
    #[getter]
    fn max_new_len(&self) -> usize {
        self.max_new_len
    }
    #[getter]
    fn sampling_params(&self) -> SamplingParams {
        self.sampling_params.clone()
    }
    #[getter]
    fn sample_index(&self) -> u64 {
        self.sample_index
    }
    #[getter]
    fn generated_token_ids(&self) -> Vec<i64> {
        self.generated_token_ids.clone()
    }
    #[getter]
    fn state(&self) -> RequestState {
        self.state
    }
    #[getter]
    fn cached_len(&self) -> usize {
        self.cached_len
    }
    #[getter]
    fn completion_reason(&self) -> Option<CompletionReason> {
        self.completion_reason
    }
    #[getter]
    fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
}

enum SchedulerInner {
    Fcfs(core::Scheduler<core::Fcfs>),
    DecodeFirst(core::Scheduler<core::DecodeFirst>),
}

#[pyclass(module = "einf._control", unsendable)]
pub struct Scheduler {
    inner: SchedulerInner,
}

macro_rules! with_scheduler {
    ($self:expr, |$scheduler:ident| $body:expr) => {
        match &$self.inner {
            SchedulerInner::Fcfs($scheduler) => $body,
            SchedulerInner::DecodeFirst($scheduler) => $body,
        }
    };
}

macro_rules! with_scheduler_mut {
    ($self:expr, |$scheduler:ident| $body:expr) => {
        match &mut $self.inner {
            SchedulerInner::Fcfs($scheduler) => $body,
            SchedulerInner::DecodeFirst($scheduler) => $body,
        }
    };
}

#[pymethods]
impl Scheduler {
    #[new]
    #[pyo3(signature = (*, policy="fcfs", num_blocks, block_len, max_batch_len, max_prefill_chunk_len))]
    fn new(
        policy: &str,
        num_blocks: usize,
        block_len: usize,
        max_batch_len: usize,
        max_prefill_chunk_len: usize,
    ) -> PyResult<Self> {
        let config = core::SchedulerConfig {
            max_batch_len,
            max_prefill_chunk_len,
        };
        let cache = core::KvCacheManager::new(num_blocks, block_len).map_err(control_error)?;
        let inner = match policy {
            "fcfs" => SchedulerInner::Fcfs(
                core::Scheduler::new(config, core::Fcfs, cache).map_err(control_error)?,
            ),
            "decode_first" => SchedulerInner::DecodeFirst(
                core::Scheduler::new(config, core::DecodeFirst, cache).map_err(control_error)?,
            ),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown scheduling policy {other:?}; expected 'fcfs' or 'decode_first'"
                )))
            }
        };
        Ok(Self { inner })
    }

    fn submit(&mut self, spec: PyRef<'_, RequestSpec>) -> PyResult<String> {
        let request_id = spec.request_id.clone();
        with_scheduler_mut!(self, |scheduler| scheduler.submit((&*spec).into()))
            .map_err(control_error)?;
        Ok(request_id)
    }

    fn schedule(&mut self) -> PyResult<Option<BatchPlan>> {
        with_scheduler_mut!(self, |scheduler| scheduler.schedule())
            .map(|plan| plan.map(Into::into))
            .map_err(control_error)
    }

    fn apply_result(&mut self, result: PyRef<'_, ExecutionResult>) -> PyResult<()> {
        with_scheduler_mut!(self, |scheduler| scheduler.apply_result((&*result).into()))
            .map_err(control_error)
    }

    fn fail_batch(&mut self, batch: PyRef<'_, BatchPlan>, message: &str) -> PyResult<()> {
        let batch: core::BatchPlan = (&*batch).into();
        with_scheduler_mut!(self, |scheduler| scheduler.fail_batch(&batch, message))
            .map_err(control_error)
    }

    fn cancel(&mut self, request_id: &str) -> PyResult<()> {
        let request_id = core::RequestId::new(request_id);
        with_scheduler_mut!(self, |scheduler| scheduler.cancel(&request_id)).map_err(control_error)
    }

    fn request(&self, request_id: &str) -> PyResult<RequestView> {
        let request_id = core::RequestId::new(request_id);
        with_scheduler!(self, |scheduler| scheduler
            .request(&request_id)
            .map(Into::into))
        .ok_or_else(|| PyValueError::new_err(format!("unknown request: {request_id}")))
    }

    fn block_table(&self, request_id: &str) -> Vec<u32> {
        let request_id = core::RequestId::new(request_id);
        with_scheduler!(self, |scheduler| scheduler.cache().block_table(request_id))
            .into_iter()
            .map(|id| id.0)
            .collect()
    }

    fn free_blocks(&self) -> usize {
        with_scheduler!(self, |scheduler| scheduler.cache().free_blocks())
    }

    fn waiting_ids(&self) -> Vec<String> {
        with_scheduler!(self, |scheduler| scheduler
            .waiting_ids()
            .map(|id| id.0.clone())
            .collect())
    }

    fn running_ids(&self) -> Vec<String> {
        with_scheduler!(self, |scheduler| scheduler
            .running_ids()
            .map(|id| id.0.clone())
            .collect())
    }
}

#[pymodule]
fn _control(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<WorkType>()?;
    module.add_class::<RequestState>()?;
    module.add_class::<CompletionReason>()?;
    module.add_class::<SamplingParams>()?;
    module.add_class::<SamplingPlan>()?;
    module.add_class::<RequestSpec>()?;
    module.add_class::<ScheduledRequest>()?;
    module.add_class::<BatchPlan>()?;
    module.add("ScheduledBatch", module.getattr("BatchPlan")?)?;
    module.add_class::<RequestExecutionResult>()?;
    module.add_class::<ExecutionResult>()?;
    module.add_class::<RequestView>()?;
    module.add_class::<Scheduler>()?;

    // Keep the all-caps spelling used by the former Python enums.
    let work_type = module.getattr("WorkType")?;
    work_type.setattr("PREFILL", work_type.getattr("Prefill")?)?;
    work_type.setattr("DECODE", work_type.getattr("Decode")?)?;
    let request_state = module.getattr("RequestState")?;
    for (upper, rust) in [
        ("WAITING", "Waiting"),
        ("RUNNING", "Running"),
        ("FINISHED", "Finished"),
        ("CANCELLED", "Cancelled"),
        ("FAILED", "Failed"),
    ] {
        request_state.setattr(upper, request_state.getattr(rust)?)?;
    }
    let completion_reason = module.getattr("CompletionReason")?;
    completion_reason.setattr("EOS", completion_reason.getattr("Eos")?)?;
    completion_reason.setattr("LENGTH", completion_reason.getattr("Length")?)?;
    Ok(())
}
