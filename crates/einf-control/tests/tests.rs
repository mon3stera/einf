//! Stage-0 parity tests for the Rust control plane.
//!
//! These tests exercise the public Rust API and preserve the Python baseline
//! scenarios. Prefix-cache behavior is covered separately in `prefix_cache.rs`.

use einf_control::{
    AdvanceResult, BatchId, CompletionReason, DecodeFirst, ExecutionResult, Fcfs, KvCacheManager,
    RequestExecutionResult, RequestId, RequestSpec, RequestState, Scheduler, SchedulerConfig,
    WorkType,
};

fn scheduler<P>(
    policy: P,
    num_blocks: usize,
    block_len: usize,
    max_batch_len: usize,
    max_prefill_chunk_len: usize,
) -> Scheduler<P>
where
    P: einf_control::SchedulingPolicy,
{
    Scheduler::new(
        SchedulerConfig {
            max_batch_len,
            max_prefill_chunk_len,
        },
        policy,
        KvCacheManager::new(num_blocks, block_len).unwrap(),
    )
    .unwrap()
}

fn spec(id: &str, prompt: &[i64], max_new_len: usize) -> RequestSpec {
    RequestSpec {
        request_id: RequestId::from(id),
        prompt_token_ids: prompt.to_vec(),
        max_new_len,
        sampling_params: einf_control::SamplingParams::default(),
    }
}

fn batch_ids(batch: &einf_control::BatchPlan) -> Vec<RequestId> {
    batch
        .requests
        .iter()
        .map(|item| item.request_id.clone())
        .collect()
}

/// Deterministic Rust equivalent of the Python FakeExecutor.
fn fake_result(batch: &einf_control::BatchPlan) -> ExecutionResult {
    ExecutionResult {
        step_id: batch.step_id,
        request_results: batch
            .requests
            .iter()
            .map(|item| RequestExecutionResult {
                request_id: item.request_id.clone(),
                generated_token_ids: if item.need_sample {
                    vec![item.input_token_ids.last().copied().unwrap_or(0) + 1]
                } else {
                    Vec::new()
                },
                cached_len_delta: item.input_token_ids.len(),
                is_eos: false,
            })
            .collect(),
    }
}

fn run_until_idle<P>(control: &mut Scheduler<P>)
where
    P: einf_control::SchedulingPolicy,
{
    while let Some(batch) = control.schedule().unwrap() {
        control.apply_result(fake_result(&batch)).unwrap();
    }
}

#[test]
fn request_rejects_empty_prompt_and_zero_output_budget() {
    assert!(matches!(
        einf_control::Request::create(spec("empty", &[], 1), 0),
        Err(einf_control::ControlError::InvalidConfig(_))
    ));
    assert!(matches!(
        einf_control::Request::create(spec("zero-output", &[1], 0), 0),
        Err(einf_control::ControlError::InvalidConfig(_))
    ));
}

#[test]
fn sampling_params_validate_and_default_to_greedy() {
    let default = einf_control::SamplingParams::default();
    assert!(default.is_greedy());
    assert_eq!(default.temperature(), 0.0);
    assert!(einf_control::SamplingParams::new(0.0, Some(1), 1.0, 0.0, 0, vec![], 0).is_err());
    assert!(einf_control::SamplingParams::new(f32::NAN, None, 1.0, 0.0, 0, vec![], 0).is_err());
    assert!(einf_control::SamplingParams::new(1.0, Some(0), 1.0, 0.0, 0, vec![], 0).is_err());
    assert!(einf_control::SamplingParams::new(1.0, None, 0.0, 0.0, 0, vec![], 0).is_err());

    let random =
        einf_control::SamplingParams::new(0.75, Some(50), 0.95, 0.05, 123, vec![2, 3], 4).unwrap();
    assert!(!random.is_greedy());
    assert_eq!(random.temperature(), 0.75);

    assert!(
        einf_control::SamplingParams::new(1.0, None, 1.0, 0.0, i64::MAX as u64, vec![], 0,).is_ok()
    );
    assert!(
        einf_control::SamplingParams::new(1.0, None, 1.0, 0.0, i64::MAX as u64 + 1, vec![], 0,)
            .is_err()
    );
}

#[test]
fn request_lifecycle_matches_python_baseline() {
    let mut request = einf_control::Request::create(spec("req-1", &[10, 20, 30], 2), 7).unwrap();
    assert_eq!(request.state(), RequestState::Waiting);
    assert!(request.fail("waiting failure").is_err());
    request.admit().unwrap();
    request
        .advance(AdvanceResult {
            generated_token_ids: vec![40],
            cached_len_delta: 3,
            completion_reason: None,
        })
        .unwrap();
    assert_eq!(request.generated_token_ids(), &[40]);
    assert_eq!(request.cached_len(), 3);

    let before = request.clone();
    assert!(request
        .advance(AdvanceResult {
            generated_token_ids: vec![41, 42],
            cached_len_delta: 1,
            completion_reason: Some(CompletionReason::Length),
        })
        .is_err());
    assert_eq!(request, before);

    request
        .advance(AdvanceResult {
            generated_token_ids: vec![41],
            cached_len_delta: 1,
            completion_reason: Some(CompletionReason::Length),
        })
        .unwrap();
    assert_eq!(request.state(), RequestState::Finished);
    assert_eq!(request.completion_reason(), Some(CompletionReason::Length));
    assert!(request
        .advance(AdvanceResult {
            generated_token_ids: vec![42],
            cached_len_delta: 1,
            completion_reason: None,
        })
        .is_err());
}

#[test]
fn block_pool_and_kv_cache_match_python_baseline() {
    let mut pool = einf_control::BlockPool::new(4);
    let reservation = pool.reserve(2, RequestId::from("pool-test")).unwrap();
    assert_eq!(
        reservation.blocks(),
        &[einf_control::BlockId(0), einf_control::BlockId(1)]
    );
    pool.release(reservation).unwrap();
    assert_eq!(pool.available(), 4);
    assert!(pool.reserve(5, RequestId::from("pool-test")).is_err());
    assert_eq!(pool.available(), 4);

    let mut cache = KvCacheManager::new(3, 4).unwrap();
    let id = RequestId::from("req-1");
    cache.reserve_to(id.clone(), None, 3).unwrap();
    assert_eq!(
        cache.block_table(id.clone()),
        vec![einf_control::BlockId(0)]
    );
    cache.reserve_to(id.clone(), None, 5).unwrap();
    assert_eq!(
        cache.block_table(id.clone()),
        vec![einf_control::BlockId(0), einf_control::BlockId(1)]
    );
    let table_before = cache.block_table(id.clone());
    assert!(cache.reserve_to(id.clone(), None, 13).is_err());
    assert_eq!(cache.block_table(id.clone()), table_before);
    cache.release(id).unwrap();
    assert_eq!(cache.free_blocks(), 3);
}

#[test]
fn batch_plan_matches_python_execution_contract() {
    let mut control = scheduler(Fcfs, 4, 4, 8, 8);
    control.submit(spec("req-1", &[10, 20, 30], 2)).unwrap();
    let prefill = control.schedule().unwrap().unwrap();
    assert_eq!(prefill.step_id, BatchId(0));
    let item = &prefill.requests[0];
    assert_eq!(item.input_token_ids, vec![10, 20, 30]);
    assert_eq!(item.work_type, WorkType::Prefill);
    assert_eq!(item.start_position, 0);
    assert_eq!(item.block_table, vec![einf_control::BlockId(0)]);
    assert!(item.need_sample);
    assert_eq!(item.sampling_plan.as_ref().unwrap().sample_index, 0);
    assert!(item.sampling_plan.as_ref().unwrap().params.is_greedy());
    control.apply_result(fake_result(&prefill)).unwrap();
    assert_eq!(
        control
            .request(&RequestId::from("req-1"))
            .unwrap()
            .sample_index(),
        1
    );

    let decode = control.schedule().unwrap().unwrap();
    assert_eq!(decode.step_id, BatchId(1));
    assert_eq!(decode.requests[0].input_token_ids, vec![31]);
    assert_eq!(decode.requests[0].work_type, WorkType::Decode);
    assert_eq!(decode.requests[0].start_position, 3);
    assert_eq!(
        decode.requests[0]
            .sampling_plan
            .as_ref()
            .unwrap()
            .sample_index,
        1
    );
}

#[test]
fn dynamic_batch_obeys_token_budget() {
    let mut control = scheduler(Fcfs, 4, 2, 2, 8);
    control.submit(spec("req-1", &[10, 20], 2)).unwrap();
    control.submit(spec("req-2", &[30], 2)).unwrap();
    let first = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&first), vec![RequestId::from("req-1")]);
    control.apply_result(fake_result(&first)).unwrap();
    let second = control.schedule().unwrap().unwrap();
    assert_eq!(
        batch_ids(&second),
        vec![RequestId::from("req-1"), RequestId::from("req-2")]
    );
    assert_eq!(
        second
            .requests
            .iter()
            .map(|item| item.input_token_ids.len())
            .sum::<usize>(),
        2
    );
}

#[test]
fn decode_grows_block_table_at_boundary() {
    let mut control = scheduler(Fcfs, 2, 4, 4, 8);
    control.submit(spec("req-1", &[10, 20, 30, 40], 2)).unwrap();
    let prefill = control.schedule().unwrap().unwrap();
    assert_eq!(
        prefill.requests[0].block_table,
        vec![einf_control::BlockId(0)]
    );
    control.apply_result(fake_result(&prefill)).unwrap();
    let decode = control.schedule().unwrap().unwrap();
    assert_eq!(decode.requests[0].input_token_ids, vec![41]);
    assert_eq!(decode.requests[0].start_position, 4);
    assert_eq!(
        decode.requests[0].block_table,
        vec![einf_control::BlockId(0), einf_control::BlockId(1)]
    );
}

#[test]
fn server_finishes_requests_and_releases_cache() {
    let mut control = scheduler(Fcfs, 3, 2, 3, 8);
    let first_id = RequestId::from("req-1");
    let second_id = RequestId::from("req-2");
    control.submit(spec("req-1", &[10, 20], 2)).unwrap();
    control.submit(spec("req-2", &[30], 2)).unwrap();
    run_until_idle(&mut control);

    let first = control.request(&first_id).unwrap();
    assert_eq!(first.state(), RequestState::Finished);
    assert_eq!(first.generated_token_ids(), &[21, 22]);
    assert_eq!(first.cached_len(), 3);
    let second = control.request(&second_id).unwrap();
    assert_eq!(second.state(), RequestState::Finished);
    assert_eq!(second.generated_token_ids(), &[31, 32]);
    assert_eq!(second.cached_len(), 2);
    assert_eq!(control.cache().free_blocks(), 3);
    assert!(control.running_ids().next().is_none());
}

#[test]
fn eos_wins_when_eos_and_length_arrive_together() {
    let mut control = scheduler(Fcfs, 1, 4, 8, 8);
    let id = RequestId::from("req-1");
    control.submit(spec("req-1", &[10, 20, 30], 1)).unwrap();
    let batch = control.schedule().unwrap().unwrap();
    control
        .apply_result(ExecutionResult {
            step_id: batch.step_id,
            request_results: vec![RequestExecutionResult {
                request_id: id.clone(),
                generated_token_ids: vec![31],
                cached_len_delta: 3,
                is_eos: true,
            }],
        })
        .unwrap();
    assert_eq!(
        control.request(&id).unwrap().completion_reason(),
        Some(CompletionReason::Eos)
    );
    assert_eq!(control.cache().free_blocks(), 1);
}

#[test]
fn decode_first_prioritizes_decode_over_prefill() {
    let mut control = scheduler(DecodeFirst, 5, 2, 3, 2);
    let prefill_id = RequestId::from("prefill");
    let decode_id = RequestId::from("decode");
    control
        .submit(spec("prefill", &[10, 20, 30, 40, 50], 1))
        .unwrap();
    control.submit(spec("decode", &[100], 3)).unwrap();

    let first = control.schedule().unwrap().unwrap();
    assert_eq!(
        batch_ids(&first),
        vec![prefill_id.clone(), decode_id.clone()]
    );
    control.apply_result(fake_result(&first)).unwrap();
    let second = control.schedule().unwrap().unwrap();
    assert_eq!(
        batch_ids(&second),
        vec![decode_id.clone(), prefill_id.clone()]
    );
    assert_eq!(second.requests[0].input_token_ids, vec![101]);
    assert_eq!(second.requests[1].input_token_ids, vec![30, 40]);
    control.apply_result(fake_result(&second)).unwrap();
    let third = control.schedule().unwrap().unwrap();
    assert_eq!(
        batch_ids(&third),
        vec![decode_id.clone(), prefill_id.clone()]
    );
    assert_eq!(third.requests[0].input_token_ids, vec![102]);
    assert_eq!(third.requests[1].input_token_ids, vec![50]);
    assert!(third.requests.iter().all(|item| item.need_sample));
    control.apply_result(fake_result(&third)).unwrap();
    assert_eq!(
        control.request(&decode_id).unwrap().generated_token_ids(),
        &[101, 102, 103]
    );
    assert_eq!(
        control.request(&prefill_id).unwrap().generated_token_ids(),
        &[51]
    );
    assert_eq!(control.cache().free_blocks(), 5);
}

#[test]
fn waiting_prefill_joins_existing_decode() {
    let mut control = scheduler(DecodeFirst, 4, 2, 3, 2);
    let decode_id = RequestId::from("decode");
    let prefill_id = RequestId::from("late-prefill");
    control.submit(spec("decode", &[100], 3)).unwrap();
    let first = control.schedule().unwrap().unwrap();
    control.apply_result(fake_result(&first)).unwrap();
    control
        .submit(spec("late-prefill", &[10, 20, 30, 40], 1))
        .unwrap();

    let mixed = control.schedule().unwrap().unwrap();
    assert_eq!(
        batch_ids(&mixed),
        vec![decode_id.clone(), prefill_id.clone()]
    );
    assert_eq!(mixed.requests[0].input_token_ids, vec![101]);
    assert_eq!(mixed.requests[1].input_token_ids, vec![10, 20]);
    assert!(!mixed.requests[1].need_sample);
    control.apply_result(fake_result(&mixed)).unwrap();
    let final_batch = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&final_batch), vec![decode_id, prefill_id]);
    assert_eq!(final_batch.requests[0].input_token_ids, vec![102]);
    assert_eq!(final_batch.requests[1].input_token_ids, vec![30, 40]);
    assert!(final_batch.requests[1].need_sample);
}

#[test]
fn chunked_prefill_samples_only_on_final_chunk() {
    let mut control = scheduler(Fcfs, 3, 2, 2, 2);
    let id = RequestId::from("req-1");
    control
        .submit(spec("req-1", &[10, 20, 30, 40, 50], 1))
        .unwrap();
    for (tokens, start, need_sample) in [
        (vec![10, 20], 0, false),
        (vec![30, 40], 2, false),
        (vec![50], 4, true),
    ] {
        let batch = control.schedule().unwrap().unwrap();
        assert_eq!(batch.requests[0].input_token_ids, tokens);
        assert_eq!(batch.requests[0].start_position, start);
        assert_eq!(batch.requests[0].need_sample, need_sample);
        assert_eq!(batch.requests[0].sampling_plan.is_some(), need_sample);
        control.apply_result(fake_result(&batch)).unwrap();
    }
    let request = control.request(&id).unwrap();
    assert_eq!(request.state(), RequestState::Finished);
    assert_eq!(request.generated_token_ids(), &[51]);
    assert_eq!(request.cached_len(), 5);
    assert_eq!(control.cache().free_blocks(), 3);
}

#[test]
fn waiting_and_running_cancel_release_cache() {
    let mut control = scheduler(Fcfs, 1, 2, 2, 2);
    let cancelled_id = RequestId::from("cancelled");
    let live_id = RequestId::from("live");
    control.submit(spec("cancelled", &[10, 20], 1)).unwrap();
    control.submit(spec("live", &[30, 40], 2)).unwrap();
    control.cancel(&cancelled_id).unwrap();
    let batch = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&batch), vec![live_id.clone()]);
    control.apply_result(fake_result(&batch)).unwrap();
    assert_eq!(control.cache().free_blocks(), 0);
    control.cancel(&live_id).unwrap();
    assert_eq!(
        control.request(&live_id).unwrap().state(),
        RequestState::Cancelled
    );
    assert_eq!(control.cache().free_blocks(), 1);
    assert!(control.schedule().unwrap().is_none());
}

#[test]
fn executor_failure_fails_batch_releases_cache_and_continues() {
    let mut control = scheduler(Fcfs, 2, 2, 4, 8);
    let request_ids = (0..3)
        .map(|index| RequestId::from(format!("req-{index}")))
        .collect::<Vec<_>>();
    for (index, request_id) in request_ids.iter().enumerate() {
        control
            .submit(RequestSpec {
                request_id: request_id.clone(),
                prompt_token_ids: vec![index as i64 * 10 + 1, index as i64 * 10 + 2],
                max_new_len: 2,
                sampling_params: einf_control::SamplingParams::default(),
            })
            .unwrap();
    }

    let first = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&first), request_ids[..2].to_vec());
    control
        .fail_batch(&first, "RuntimeError: injected execution failure")
        .unwrap();
    for id in &request_ids[..2] {
        assert_eq!(control.request(id).unwrap().state(), RequestState::Failed);
        assert_eq!(
            control.request(id).unwrap().error(),
            Some("RuntimeError: injected execution failure")
        );
    }
    assert_eq!(
        control.request(&request_ids[2]).unwrap().state(),
        RequestState::Waiting
    );
    assert_eq!(control.cache().free_blocks(), 2);

    let second = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&second), vec![request_ids[2].clone()]);
    control
        .fail_batch(&second, "RuntimeError: injected execution failure")
        .unwrap();
    assert_eq!(
        control.request(&request_ids[2]).unwrap().state(),
        RequestState::Failed
    );
    assert_eq!(control.cache().free_blocks(), 2);
    assert!(control.schedule().unwrap().is_none());
}

#[test]
fn single_request_oom_fails_without_infinite_reschedule() {
    let mut control = scheduler(Fcfs, 1, 2, 2, 2);
    let id = RequestId::from("too-large");
    control.submit(spec("too-large", &[10, 20], 2)).unwrap();
    let first = control.schedule().unwrap().unwrap();
    control.apply_result(fake_result(&first)).unwrap();
    assert_eq!(control.request(&id).unwrap().cached_len(), 2);
    assert_eq!(control.request(&id).unwrap().generated_token_ids(), &[21]);

    // Python baseline: one running request that cannot grow must fail rather
    // than preempt itself and emit a redundant prefill recomputation batch.
    let second = control.schedule().unwrap();
    assert!(second.is_none(), "OOM retry must not emit recompute work");
    let request = control.request(&id).unwrap();
    assert_eq!(request.state(), RequestState::Failed);
    assert_eq!(
        request.error(),
        Some("Insufficient memory to fulfill request too-large")
    );
    assert_eq!(control.cache().free_blocks(), 1);
}

#[test]
fn fcfs_preempts_newer_request_and_recomputes_it() {
    let mut control = scheduler(Fcfs, 2, 2, 4, 2);
    let older_id = RequestId::from("older");
    let newer_id = RequestId::from("newer");
    control.submit(spec("older", &[10, 20], 2)).unwrap();
    control.submit(spec("newer", &[30, 40], 2)).unwrap();
    let first = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&first), vec![older_id.clone(), newer_id.clone()]);
    control.apply_result(fake_result(&first)).unwrap();

    let second = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&second), vec![older_id.clone()]);
    assert_eq!(
        control.request(&newer_id).unwrap().state(),
        RequestState::Waiting
    );
    assert_eq!(control.request(&newer_id).unwrap().cached_len(), 0);
    control.apply_result(fake_result(&second)).unwrap();
    assert_eq!(
        control.request(&older_id).unwrap().state(),
        RequestState::Finished
    );

    let recompute = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&recompute), vec![newer_id.clone()]);
    assert_eq!(recompute.requests[0].input_token_ids, vec![30, 40]);
    assert_eq!(recompute.requests[0].start_position, 0);
    assert!(!recompute.requests[0].need_sample);
    control.apply_result(fake_result(&recompute)).unwrap();
    let decode = control.schedule().unwrap().unwrap();
    assert_eq!(decode.requests[0].input_token_ids, vec![41]);
    control.apply_result(fake_result(&decode)).unwrap();
    assert_eq!(
        control.request(&newer_id).unwrap().state(),
        RequestState::Finished
    );
    assert_eq!(control.cache().free_blocks(), 2);
}

#[test]
fn decode_first_preempts_prefill_before_decode() {
    let mut control = scheduler(DecodeFirst, 2, 2, 4, 2);
    let prefill_id = RequestId::from("prefill");
    let decode_id = RequestId::from("decode");
    control
        .submit(spec("prefill", &[30, 40, 50, 60], 1))
        .unwrap();
    control.submit(spec("decode", &[10, 20], 2)).unwrap();
    let first = control.schedule().unwrap().unwrap();
    control.apply_result(fake_result(&first)).unwrap();

    let decode = control.schedule().unwrap().unwrap();
    assert_eq!(batch_ids(&decode), vec![decode_id.clone()]);
    assert_eq!(
        control.request(&prefill_id).unwrap().state(),
        RequestState::Waiting
    );
    assert_eq!(control.request(&prefill_id).unwrap().cached_len(), 0);
    control.apply_result(fake_result(&decode)).unwrap();
    assert_eq!(
        control.request(&decode_id).unwrap().state(),
        RequestState::Finished
    );

    let first_recompute = control.schedule().unwrap().unwrap();
    assert_eq!(first_recompute.requests[0].input_token_ids, vec![30, 40]);
    control.apply_result(fake_result(&first_recompute)).unwrap();
    let final_prefill = control.schedule().unwrap().unwrap();
    assert_eq!(final_prefill.requests[0].input_token_ids, vec![50, 60]);
    assert!(final_prefill.requests[0].need_sample);
    control.apply_result(fake_result(&final_prefill)).unwrap();
    assert_eq!(
        control.request(&prefill_id).unwrap().state(),
        RequestState::Finished
    );
    assert_eq!(control.cache().free_blocks(), 2);
}

#[test]
fn malformed_result_is_transactional_and_inflight_cancel_is_rejected() {
    let mut control = scheduler(Fcfs, 2, 2, 2, 2);
    let id = RequestId::from("req-1");
    control.submit(spec("req-1", &[10, 20], 1)).unwrap();
    let batch = control.schedule().unwrap().unwrap();
    assert!(control.cancel(&id).is_err());
    let malformed = ExecutionResult {
        step_id: batch.step_id,
        request_results: vec![RequestExecutionResult {
            request_id: id.clone(),
            generated_token_ids: Vec::new(),
            cached_len_delta: 0,
            is_eos: false,
        }],
    };
    assert!(control.apply_result(malformed).is_err());
    assert_eq!(control.request(&id).unwrap().state(), RequestState::Running);
    assert_eq!(control.request(&id).unwrap().cached_len(), 0);
    control.apply_result(fake_result(&batch)).unwrap();
}

#[test]
fn duplicate_request_and_invalid_config_are_rejected() {
    assert!(Scheduler::new(
        SchedulerConfig {
            max_batch_len: 0,
            max_prefill_chunk_len: 1,
        },
        Fcfs,
        KvCacheManager::new(1, 2).unwrap(),
    )
    .is_err());
    assert!(KvCacheManager::new(1, 0).is_err());
    let mut control = scheduler(Fcfs, 1, 2, 2, 2);
    control.submit(spec("duplicate", &[1], 1)).unwrap();
    assert!(control.submit(spec("duplicate", &[2], 1)).is_err());
}
