use std::fmt;

use crate::error::ControlError;
use crate::sample::SamplingParams;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RequestId(pub String);

impl RequestId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl From<&str> for RequestId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for RequestId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestState {
    Waiting,
    Running,
    Finished,
    Cancelled,
    Failed,
}
impl RequestState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionReason {
    Eos,
    Length,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestSpec {
    pub request_id: RequestId,
    pub prompt_token_ids: Vec<i64>,
    pub max_new_len: usize,
    pub sampling_params: SamplingParams,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdvanceResult {
    pub generated_token_ids: Vec<i64>,
    pub cached_len_delta: usize,
    pub completion_reason: Option<CompletionReason>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    request_id: RequestId,
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

impl Request {
    pub fn create(spec: RequestSpec, arrival_index: u64) -> Result<Self, ControlError> {
        if spec.prompt_token_ids.is_empty() {
            return Err(ControlError::InvalidConfig(
                "prompt must contain at least one token",
            ));
        }
        if spec.max_new_len == 0 {
            return Err(ControlError::InvalidConfig("max_new_len must be positive"));
        }

        let request = Self {
            request_id: spec.request_id,
            prompt_token_ids: spec.prompt_token_ids,
            arrival_index,
            max_new_len: spec.max_new_len,
            sampling_params: spec.sampling_params,
            sample_index: 0,
            generated_token_ids: Vec::new(),
            state: RequestState::Waiting,
            cached_len: 0,
            completion_reason: None,
            error: None,
        };
        request.assert_invariants()?;
        Ok(request)
    }

    pub fn request_id(&self) -> RequestId {
        self.request_id.clone()
    }

    pub fn prompt_token_ids(&self) -> &[i64] {
        &self.prompt_token_ids
    }

    pub fn arrival_index(&self) -> u64 {
        self.arrival_index
    }

    pub fn max_new_len(&self) -> usize {
        self.max_new_len
    }

    pub fn sampling_params(&self) -> &SamplingParams {
        &self.sampling_params
    }

    pub fn sample_index(&self) -> u64 {
        self.sample_index
    }

    pub fn generated_token_ids(&self) -> &[i64] {
        &self.generated_token_ids
    }

    pub fn state(&self) -> RequestState {
        self.state
    }

    pub fn cached_len(&self) -> usize {
        self.cached_len
    }

    pub fn completion_reason(&self) -> Option<CompletionReason> {
        self.completion_reason
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn context_token_ids(&self) -> Vec<i64> {
        self.prompt_token_ids
            .iter()
            .chain(&self.generated_token_ids)
            .copied()
            .collect()
    }

    pub fn pending_len(&self) -> usize {
        if self.cached_len >= self.prompt_token_ids.len() {
            1
        } else {
            self.prompt_token_ids.len() - self.cached_len
        }
    }

    pub fn work_is_decode(&self) -> bool {
        self.cached_len >= self.prompt_token_ids.len()
    }

    pub fn admit(&mut self) -> Result<(), ControlError> {
        self.assert_invariants()?;
        self.assert_state(RequestState::Waiting)?;
        self.state = RequestState::Running;
        self.assert_invariants()
    }

    pub fn reuse(&mut self, reused_len: usize) -> Result<(), ControlError> {
        self.assert_invariants()?;
        self.assert_state(RequestState::Running)?;
        self.cached_len += reused_len;
        assert!(self.cached_len < self.prompt_token_ids.len());
        self.assert_invariants()
    }

    pub fn preempt(&mut self) -> Result<(), ControlError> {
        self.assert_invariants()?;
        self.assert_state(RequestState::Running)?;
        self.state = RequestState::Waiting;
        self.cached_len = 0;
        self.assert_invariants()
    }

    pub fn advance(&mut self, result: AdvanceResult) -> Result<(), ControlError> {
        self.assert_invariants()?;
        self.assert_state(RequestState::Running)?;

        let generated_len = self
            .generated_token_ids
            .len()
            .checked_add(result.generated_token_ids.len())
            .ok_or(ControlError::ArithmeticOverflow)?;

        if generated_len > self.max_new_len {
            return Err(self.invariant("advance exceeds max_new_len"));
        }

        let sampled_tokens = u64::try_from(result.generated_token_ids.len())
            .map_err(|_| ControlError::ArithmeticOverflow)?;
        let sample_index = self
            .sample_index
            .checked_add(sampled_tokens)
            .ok_or(ControlError::ArithmeticOverflow)?;

        let cached_len = self
            .cached_len
            .checked_add(result.cached_len_delta)
            .ok_or(ControlError::ArithmeticOverflow)?;

        if let Some(reason) = result.completion_reason {
            self.state = RequestState::Finished;
            self.completion_reason = Some(reason);
        }

        self.generated_token_ids.extend(result.generated_token_ids);
        self.sample_index = sample_index;
        self.cached_len = cached_len;
        self.assert_invariants()
    }

    pub fn finish(&mut self, reason: CompletionReason) -> Result<(), ControlError> {
        self.assert_invariants()?;
        self.assert_state(RequestState::Running)?;
        self.state = RequestState::Finished;
        self.completion_reason = Some(reason);
        self.assert_invariants()
    }

    pub fn fail(&mut self, error: impl Into<String>) -> Result<(), ControlError> {
        self.assert_invariants()?;
        self.assert_state(RequestState::Running)?;
        self.state = RequestState::Failed;
        self.error = Some(error.into());
        self.assert_invariants()
    }

    pub fn cancel(&mut self) -> Result<(), ControlError> {
        self.assert_invariants()?;

        match self.state {
            RequestState::Waiting | RequestState::Running => {
                self.state = RequestState::Cancelled;
                self.assert_invariants()
            }
            RequestState::Cancelled => Ok(()),
            state => Err(ControlError::InvalidState {
                request_id: self.request_id.clone(),
                expected: "Waiting or Running",
                actual: state,
            }),
        }
    }

    fn invariant(&self, message: &'static str) -> ControlError {
        ControlError::RequestInvariant {
            request_id: self.request_id.clone(),
            message: message.to_owned(),
        }
    }

    fn assert_state(&self, expected: RequestState) -> Result<(), ControlError> {
        if self.state == expected {
            Ok(())
        } else {
            Err(ControlError::InvalidState {
                request_id: self.request_id.clone(),
                expected: match expected {
                    RequestState::Waiting => "Waiting",
                    RequestState::Running => "Running",
                    _ => "requested state",
                },
                actual: self.state,
            })
        }
    }

    pub fn assert_invariants(&self) -> Result<(), ControlError> {
        if self.generated_token_ids.len() > self.max_new_len {
            return Err(self.invariant("generated token count exceeds max_new_len"));
        }
        if usize::try_from(self.sample_index).ok() != Some(self.generated_token_ids.len()) {
            return Err(self.invariant("sample index does not match generated token count"));
        }

        match self.state {
            RequestState::Waiting | RequestState::Running => {
                if self.completion_reason.is_some() || self.error.is_some() {
                    return Err(self.invariant("active request has terminal metadata"));
                }
            }
            RequestState::Finished => {
                if self.completion_reason.is_none() || self.error.is_some() {
                    return Err(self.invariant("finished request has invalid terminal metadata"));
                }
            }
            RequestState::Cancelled => {
                if self.completion_reason.is_some() || self.error.is_some() {
                    return Err(self.invariant("cancelled request has terminal metadata"));
                }
            }
            RequestState::Failed => {
                if self.completion_reason.is_some() || self.error.is_none() {
                    return Err(self.invariant("failed request has invalid terminal metadata"));
                }
            }
        }
        Ok(())
    }
}
