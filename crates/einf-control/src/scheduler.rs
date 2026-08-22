use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::error::ControlError;
use crate::execution::{BatchId, ExecutionResult};
use crate::execution_plan::{BatchPlan, ScheduledRequest, WorkType};
use crate::kv_cache::{KvCacheManager, ReusePlan};
use crate::policy::{PolicyRequest, SchedulingPolicy};
use crate::request::{
    AdvanceResult, CompletionReason, Request, RequestId, RequestSpec, RequestState,
};
use crate::sample::SamplingPlan;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    pub max_batch_len: usize,
    pub max_prefill_chunk_len: usize,
}

pub struct Scheduler<P> {
    config: SchedulerConfig,
    policy: P,
    cache: KvCacheManager,
    requests: BTreeMap<RequestId, Request>,
    waiting: VecDeque<RequestId>,
    running: Vec<RequestId>,
    next_arrival: u64,
    next_step: u64,
    outstanding: Option<BatchPlan>,
}

impl<P: SchedulingPolicy> Scheduler<P> {
    pub fn new(
        config: SchedulerConfig,
        policy: P,
        cache: KvCacheManager,
    ) -> Result<Self, ControlError> {
        if config.max_batch_len == 0 || config.max_prefill_chunk_len == 0 {
            return Err(ControlError::InvalidConfig(
                "batch and prefill limits must be positive",
            ));
        }

        Ok(Self {
            config,
            policy,
            cache,
            requests: BTreeMap::new(),
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_arrival: 0,
            next_step: 0,
            outstanding: None,
        })
    }

    pub fn submit(&mut self, spec: RequestSpec) -> Result<(), ControlError> {
        if self.requests.contains_key(&spec.request_id) {
            return Err(ControlError::DuplicateRequest(spec.request_id));
        }

        let id = spec.request_id.clone();
        let request = Request::create(spec, self.next_arrival)?;

        self.next_arrival = self
            .next_arrival
            .checked_add(1)
            .ok_or(ControlError::ArithmeticOverflow)?;
        self.requests.insert(id.clone(), request);
        self.waiting.push_back(id);
        Ok(())
    }

    pub fn request(&self, id: &RequestId) -> Option<&Request> {
        self.requests.get(id)
    }

    pub fn cache(&self) -> &KvCacheManager {
        &self.cache
    }

    pub fn waiting_ids(&self) -> impl Iterator<Item = &RequestId> {
        self.waiting.iter()
    }

    pub fn running_ids(&self) -> impl Iterator<Item = &RequestId> {
        self.running.iter()
    }

    fn batch_contains(&self, id: &RequestId) -> bool {
        self.outstanding.as_ref().map_or(false, |batch| {
            batch.requests.iter().any(|item| &item.request_id == id)
        })
    }

    pub fn cancel(&mut self, id: &RequestId) -> Result<(), ControlError> {
        if !self.requests.contains_key(id) {
            return Err(ControlError::UnknownRequest(id.clone()));
        }
        if self.batch_contains(id) {
            return Err(ControlError::RequestInFlight(id.clone()));
        }
        let request = self
            .requests
            .get_mut(id)
            .expect("request existence checked above");
        request.cancel()?;
        self.running.retain(|candidate| candidate != id);
        self.waiting.retain(|candidate| candidate != id);
        self.cache.release(id.clone())
    }

    fn work_type(request: &Request) -> WorkType {
        if request.work_is_decode() {
            WorkType::Decode
        } else {
            WorkType::Prefill
        }
    }

    fn policy_request(&self, request: &Request) -> PolicyRequest {
        PolicyRequest {
            id: request.request_id(),
            arrival_index: request.arrival_index(),
            state: request.state(),
            work_type: Self::work_type(request),
        }
    }

    fn ordered_running_ids(&self) -> Vec<RequestId> {
        let mut ids = self
            .running
            .iter()
            .filter(|id| {
                self.requests
                    .get(*id)
                    .map_or(false, |request| request.state() == RequestState::Running)
            })
            .cloned()
            .collect::<Vec<_>>();

        ids.sort_by(|left, right| {
            self.policy.compare(
                &self.policy_request(self.requests.get(left).expect("running request")),
                &self.policy_request(self.requests.get(right).expect("running request")),
            )
        });

        ids
    }

    fn victim_id(&self, protected: &BTreeSet<RequestId>) -> Option<RequestId> {
        self.ordered_running_ids()
            .into_iter()
            .filter(|id| !protected.contains(id))
            .last()
    }

    fn preempt_id(&mut self, id: &RequestId) -> Result<(), ControlError> {
        let request = self
            .requests
            .get_mut(id)
            .ok_or_else(|| ControlError::UnknownRequest(id.clone()))?;
        request.preempt()?;
        self.running.retain(|candidate| candidate != id);
        self.waiting.push_front(id.clone());
        self.cache.release(id.clone())
    }

    fn fail_id(&mut self, id: &RequestId, message: String) -> Result<(), ControlError> {
        let request = self
            .requests
            .get_mut(id)
            .ok_or_else(|| ControlError::UnknownRequest(id.clone()))?;
        request.fail(message)?;
        self.running.retain(|candidate| candidate != id);
        self.cache.release(id.clone())
    }

    fn prepare_schedule_context(
        &self,
        request: &Request,
        plan: &Option<ReusePlan>,
    ) -> (usize, Vec<i64>, WorkType) {
        match plan {
            Some(plan) => {
                let start = plan.reused_len();
                let context = request.context_token_ids();
                let work_type = WorkType::Prefill;
                (start, context, work_type)
            }
            None => (
                request.cached_len(),
                request.context_token_ids(),
                Self::work_type(request),
            ),
        }
    }

    fn schedule_one(
        &mut self,
        id: &RequestId,
        plan: Option<ReusePlan>,
        remaining: usize,
    ) -> Result<Option<(ScheduledRequest, usize)>, ControlError> {
        let request = self
            .requests
            .get(id)
            .ok_or_else(|| ControlError::UnknownRequest(id.clone()))?;

        let (start, context, work_type) = self.prepare_schedule_context(request, &plan);

        let pending = if work_type == WorkType::Prefill {
            context.len().saturating_sub(start)
        } else {
            1
        };

        let scheduled_len = if work_type == WorkType::Prefill {
            self.config
                .max_prefill_chunk_len
                .min(pending)
                .min(remaining)
        } else {
            1.min(remaining)
        };

        if scheduled_len == 0 {
            return Ok(None);
        }

        if start
            .checked_add(scheduled_len)
            .map_or(true, |end| end > context.len())
        {
            return Err(ControlError::InvalidExecutionResult(
                "decode request has no input token".into(),
            ));
        }

        let required = start
            .checked_add(scheduled_len)
            .ok_or(ControlError::ArithmeticOverflow)?;

        let table = self
            .cache
            .reserve_to(id.clone(), plan, required)?
            .blocks()
            .to_vec();

        let input = context[start..start + scheduled_len].to_vec();
        let need_sample = scheduled_len == pending;
        let sampling_plan = need_sample.then(|| SamplingPlan {
            params: request.sampling_params().clone(),
            sample_index: request.sample_index(),
        });

        Ok(Some((
            ScheduledRequest {
                request_id: id.clone(),
                input_token_ids: input,
                work_type,
                start_position: start,
                block_table: table,
                need_sample,
                sampling_plan,
            },
            scheduled_len,
        )))
    }

    fn try_schedule_running(
        &mut self,
        id: &RequestId,
        remaining: usize,
        protected: &BTreeSet<RequestId>,
        selected: &mut Vec<ScheduledRequest>,
    ) -> Result<usize, ControlError> {
        loop {
            match self.schedule_one(id, None, remaining) {
                Ok(Some((item, used))) => {
                    selected.push(item);
                    return Ok(used);
                }
                Ok(None) => return Ok(0),
                Err(ControlError::InsufficientBlocks { .. }) => {
                    let Some(victim) = self.victim_id(protected) else {
                        self.fail_id(id, format!("Insufficient memory to fulfill request {}", id))?;
                        return Ok(0);
                    };
                    if victim == *id {
                        self.fail_id(id, format!("Insufficient memory to fulfill request {}", id))?;
                        return Ok(0);
                    }
                    self.preempt_id(&victim)?;
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub fn schedule(&mut self) -> Result<Option<BatchPlan>, ControlError> {
        if self.outstanding.is_some() {
            return Err(ControlError::InvalidExecutionResult(
                "previous batch is still outstanding".into(),
            ));
        }

        let mut remaining = self.config.max_batch_len;
        let mut selected = Vec::new();
        let mut protected = BTreeSet::new();

        for id in self.ordered_running_ids() {
            if remaining == 0 {
                break;
            }
            if self
                .requests
                .get(&id)
                .map_or(true, |request| request.state() != RequestState::Running)
            {
                continue;
            }
            let used = self.try_schedule_running(&id, remaining, &protected, &mut selected)?;
            if used != 0 {
                remaining -= used;
                protected.insert(id);
            }
        }

        while remaining > 0 {
            let Some(id) = self.waiting.front().cloned() else {
                break;
            };

            if self
                .requests
                .get(&id)
                .map_or(true, |request| request.state() != RequestState::Waiting)
            {
                self.waiting.pop_front();
                continue;
            }

            let request = self
                .requests
                .get(&id)
                .ok_or_else(|| ControlError::UnknownRequest(id.clone()))?;

            let plan = self.cache.plan_reuse(&request.prompt_token_ids());
            let reused_len = match &plan {
                Some(plan) => plan.reused_len(),
                None => 0,
            };

            match self.schedule_one(&id, plan, remaining) {
                Ok(Some((item, used))) => {
                    self.waiting.pop_front();

                    let request = self.requests.get_mut(&id).expect("waiting request");

                    request.admit()?;

                    if reused_len != 0 {
                        request.reuse(reused_len)?;
                    }

                    self.running.push(id.clone());
                    selected.push(item);
                    remaining -= used;
                    protected.insert(id);
                }
                Ok(None) => break,
                Err(ControlError::InsufficientBlocks { .. }) => break,
                Err(error) => return Err(error),
            }
        }

        if selected.is_empty() {
            return Ok(None);
        }
        let plan = BatchPlan {
            step_id: BatchId(self.next_step),
            requests: selected,
        };
        self.next_step = self
            .next_step
            .checked_add(1)
            .ok_or(ControlError::ArithmeticOverflow)?;
        self.outstanding = Some(plan.clone());
        Ok(Some(plan))
    }

    fn reject_result<T>(
        &mut self,
        plan: &BatchPlan,
        error: ControlError,
    ) -> Result<T, ControlError> {
        self.outstanding = Some(plan.clone());
        Err(error)
    }

    fn seal_blocks(
        cache: &mut KvCacheManager,
        request: &Request,
        old_cached_len: usize,
    ) -> Result<(), ControlError> {
        let sealed_len = (old_cached_len / cache.block_len()) * cache.block_len();
        let context = request.context_token_ids();
        let seal_end = (request.cached_len() / cache.block_len()) * cache.block_len();
        let need_seal_tokens = &context[sealed_len..seal_end];
        cache.seal_blocks(request.request_id(), need_seal_tokens, sealed_len)?;
        Ok(())
    }

    pub fn apply_result(&mut self, result: ExecutionResult) -> Result<(), ControlError> {
        let plan = self
            .outstanding
            .take()
            .ok_or(ControlError::NoOutstandingBatch)?;

        if result.step_id != plan.step_id {
            return self.reject_result(
                &plan,
                ControlError::StaleBatch {
                    expected: plan.step_id,
                    actual: result.step_id,
                },
            );
        }

        if result.request_results.len() != plan.requests.len() {
            return self.reject_result(
                &plan,
                ControlError::InvalidExecutionResult("result count does not match batch".into()),
            );
        }

        let mut seen = BTreeSet::new();

        for item in &result.request_results {
            if !seen.insert(item.request_id.clone()) {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidExecutionResult("duplicate request result".into()),
                );
            }

            let Some(scheduled) = plan
                .requests
                .iter()
                .find(|scheduled| scheduled.request_id == item.request_id)
            else {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidExecutionResult("unknown request result".into()),
                );
            };

            if item.cached_len_delta != scheduled.input_token_ids.len() {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidExecutionResult("cached length delta mismatch".into()),
                );
            }

            let Some(request) = self.requests.get(&item.request_id) else {
                return self
                    .reject_result(&plan, ControlError::UnknownRequest(item.request_id.clone()));
            };

            if request.state() != RequestState::Running {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidState {
                        request_id: item.request_id.clone(),
                        expected: "Running",
                        actual: request.state(),
                    },
                );
            }

            let Some(generated_len) = request
                .generated_token_ids()
                .len()
                .checked_add(item.generated_token_ids.len())
            else {
                return self.reject_result(&plan, ControlError::ArithmeticOverflow);
            };

            if generated_len > request.max_new_len() {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidExecutionResult(
                        "generated token count exceeds request limit".into(),
                    ),
                );
            }

            if item.generated_token_ids.len() > 1 {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidExecutionResult(
                        "execution result generated more than one token".into(),
                    ),
                );
            }

            if scheduled.need_sample != scheduled.sampling_plan.is_some() {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidExecutionResult(
                        "sampling plan does not match need_sample".into(),
                    ),
                );
            }
            if item.generated_token_ids.is_empty() != !scheduled.need_sample {
                return self.reject_result(
                    &plan,
                    ControlError::InvalidExecutionResult(
                        "generated token presence does not match sampling plan".into(),
                    ),
                );
            }
            if let Some(sampling_plan) = &scheduled.sampling_plan {
                if sampling_plan.sample_index != request.sample_index() {
                    return self.reject_result(
                        &plan,
                        ControlError::InvalidExecutionResult(
                            "sampling plan index does not match request".into(),
                        ),
                    );
                }
            }
        }

        // Validate the complete result before mutating any request or cache state.
        let mut staged_requests = self.requests.clone();
        let mut staged_cache = self.cache.clone();
        let mut staged_running = self.running.clone();

        for item in result.request_results {
            let request = staged_requests
                .get_mut(&item.request_id)
                .expect("validated request result");

            let reason = if item.is_eos {
                Some(CompletionReason::Eos)
            } else if request.generated_token_ids().len() + item.generated_token_ids.len()
                == request.max_new_len()
            {
                Some(CompletionReason::Length)
            } else {
                None
            };

            let old_cached_len = request.cached_len();

            if let Err(error) = request.advance(AdvanceResult {
                generated_token_ids: item.generated_token_ids,
                cached_len_delta: item.cached_len_delta,
                completion_reason: reason,
            }) {
                return self.reject_result(&plan, error);
            }

            if let Err(error) = Self::seal_blocks(&mut staged_cache, request, old_cached_len) {
                return self.reject_result(&plan, error);
            }

            if request.state().is_terminal() {
                staged_running.retain(|candidate| candidate != &item.request_id);
                if let Err(error) = staged_cache.release(item.request_id.clone()) {
                    return self.reject_result(&plan, error);
                }
            }
        }

        self.requests = staged_requests;
        self.cache = staged_cache;
        self.running = staged_running;
        Ok(())
    }

    pub fn fail_batch(
        &mut self,
        batch: &BatchPlan,
        message: impl Into<String>,
    ) -> Result<(), ControlError> {
        let plan = self
            .outstanding
            .take()
            .ok_or(ControlError::NoOutstandingBatch)?;
        if plan != *batch {
            self.outstanding = Some(plan.clone());
            return Err(ControlError::StaleBatch {
                expected: plan.step_id,
                actual: batch.step_id,
            });
        }
        let mut staged_requests = self.requests.clone();
        let mut staged_cache = self.cache.clone();
        let mut staged_running = self.running.clone();
        let message = message.into();
        for item in &plan.requests {
            let request = staged_requests
                .get_mut(&item.request_id)
                .ok_or_else(|| ControlError::UnknownRequest(item.request_id.clone()))?;
            request.fail(message.clone())?;
            staged_running.retain(|candidate| candidate != &item.request_id);
            staged_cache.release(item.request_id.clone())?;
        }
        self.requests = staged_requests;
        self.cache = staged_cache;
        self.running = staged_running;
        Ok(())
    }
}
