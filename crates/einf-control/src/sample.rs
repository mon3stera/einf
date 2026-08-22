use crate::error::ControlError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SamplingMode {
    Greedy,
    Random { temperature: f32 },
}

impl SamplingMode {
    pub fn temperature(self) -> f32 {
        match self {
            Self::Greedy => 0.0,
            Self::Random { temperature } => temperature,
        }
    }

    pub fn is_greedy(self) -> bool {
        matches!(self, Self::Greedy)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub mode: SamplingMode,
    pub top_k: Option<u32>,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub stop_token_ids: Vec<i64>,
    pub num_logprobs: usize,
}

impl SamplingParams {
    pub fn new(
        temperature: f32,
        top_k: Option<u32>,
        top_p: f32,
        min_p: f32,
        seed: u64,
        stop_token_ids: Vec<i64>,
        num_logprobs: usize,
    ) -> Result<Self, ControlError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(ControlError::InvalidConfig(
                "temperature must be finite and non-negative",
            ));
        }
        if matches!(top_k, Some(0)) {
            return Err(ControlError::InvalidConfig(
                "top_k must be positive when specified",
            ));
        }
        if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
            return Err(ControlError::InvalidConfig(
                "top_p must be finite and in (0, 1]",
            ));
        }
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err(ControlError::InvalidConfig(
                "min_p must be finite and in [0, 1]",
            ));
        }
        if seed > i64::MAX as u64 {
            return Err(ControlError::InvalidConfig("seed must not exceed i64::MAX"));
        }

        let mode = if temperature == 0.0 {
            if top_k.is_some() || top_p != 1.0 || min_p != 0.0 {
                return Err(ControlError::InvalidConfig(
                    "top_k, top_p, and min_p require positive temperature",
                ));
            }
            SamplingMode::Greedy
        } else {
            SamplingMode::Random { temperature }
        };

        Ok(Self {
            mode,
            top_k,
            top_p,
            min_p,
            seed,
            stop_token_ids,
            num_logprobs,
        })
    }

    pub fn greedy(stop_token_ids: Vec<i64>, num_logprobs: usize) -> Self {
        Self {
            mode: SamplingMode::Greedy,
            top_k: None,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            stop_token_ids,
            num_logprobs,
        }
    }

    pub fn temperature(&self) -> f32 {
        self.mode.temperature()
    }

    pub fn is_greedy(&self) -> bool {
        self.mode.is_greedy()
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::greedy(Vec::new(), 0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingPlan {
    pub params: SamplingParams,
    pub sample_index: u64,
}
