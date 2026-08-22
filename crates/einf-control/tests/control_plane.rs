use einf_control::{
    AdvanceResult, BatchId, CompletionReason, DecodeFirst, ExecutionResult, Fcfs, KvCacheManager,
    RequestExecutionResult, RequestId, RequestSpec, RequestState, Scheduler, SchedulerConfig,
    WorkType,
};

fn scheduler<P>(
    policy: P,
    blocks: usize,
    block_len: usize,
    batch: usize,
    chunk: usize,
) -> Scheduler<P>
where
    P: einf_control::SchedulingPolicy,
{
    Scheduler::new(
        SchedulerConfig {
            max_batch_len: batch,
            max_prefill_chunk_len: chunk,
        },
        policy,
        KvCacheManager::new(blocks, block_len).unwrap(),
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
                    vec![]
                },
                cached_len_delta: item.input_token_ids.len(),
                is_eos: false,
            })
            .collect(),
    }
}

#[test]
fn request_lifecycle_matches_python_baseline() {
    let mut request = einf_control::Request::create(spec("req-1", &[10, 20, 30], 2), 7).unwrap();
    assert_eq!(request.state(), RequestState::Waiting);
    request.admit().unwrap();
    request
        .advance(AdvanceResult {
            generated_token_ids: vec![40],
            cached_len_delta: 3,
            completion_reason: None,
        })
        .unwrap();
    request
        .advance(AdvanceResult {
            generated_token_ids: vec![41],
            cached_len_delta: 1,
            completion_reason: Some(CompletionReason::Length),
        })
        .unwrap();
    assert_eq!(request.state(), RequestState::Finished);
    assert_eq!(request.generated_token_ids(), &[40, 41]);
}

#[test]
fn block_pool_and_kv_allocation_are_all_or_nothing() {
    let mut cache = KvCacheManager::new(2, 4).unwrap();
    cache.reserve_to(RequestId::from("req-1"), None, 5).unwrap();
    assert_eq!(cache.block_table(RequestId::from("req-1")).len(), 2);
    assert!(cache.reserve_to(RequestId::from("req-1"), None, 9).is_err());
    assert_eq!(cache.block_table(RequestId::from("req-1")).len(), 2);
    cache.release(RequestId::from("req-1")).unwrap();
    assert_eq!(cache.free_blocks(), 2);
}

#[test]
fn scheduler_builds_and_reconciles_batch_plan() {
    let mut scheduler = scheduler(Fcfs, 3, 2, 2, 2);
    scheduler.submit(spec("req-1", &[10, 20], 2)).unwrap();
    scheduler.submit(spec("req-2", &[30], 2)).unwrap();

    let first = scheduler.schedule().unwrap().unwrap();
    assert_eq!(first.requests.len(), 1);
    assert_eq!(first.requests[0].request_id, RequestId::from("req-1"));
    assert_eq!(first.requests[0].work_type, WorkType::Prefill);
    scheduler.apply_result(fake_result(&first)).unwrap();

    let second = scheduler.schedule().unwrap().unwrap();
    assert_eq!(
        second
            .requests
            .iter()
            .map(|item| item.input_token_ids.len())
            .sum::<usize>(),
        2
    );
    scheduler.apply_result(fake_result(&second)).unwrap();
}

#[test]
fn decode_first_prioritizes_decode_after_prefill() {
    let mut scheduler = scheduler(DecodeFirst, 5, 2, 3, 2);
    scheduler
        .submit(spec("prefill", &[10, 20, 30, 40, 50], 1))
        .unwrap();
    scheduler.submit(spec("decode", &[100], 3)).unwrap();
    let first = scheduler.schedule().unwrap().unwrap();
    scheduler.apply_result(fake_result(&first)).unwrap();
    let second = scheduler.schedule().unwrap().unwrap();
    assert_eq!(second.requests[0].request_id, RequestId::from("decode"));
    assert_eq!(second.requests[0].work_type, WorkType::Decode);
}

#[test]
fn stale_result_does_not_consume_outstanding_batch() {
    let mut scheduler = scheduler(Fcfs, 2, 2, 2, 2);
    scheduler.submit(spec("req-1", &[10, 20], 1)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    let stale = ExecutionResult {
        step_id: BatchId(batch.step_id.0 + 1),
        request_results: vec![],
    };
    assert!(scheduler.apply_result(stale).is_err());
    scheduler.apply_result(fake_result(&batch)).unwrap();
}

#[test]
fn cancellation_releases_waiting_request_without_scheduling_it() {
    let mut scheduler = scheduler(Fcfs, 1, 2, 2, 2);
    let id = RequestId::from("cancelled");
    scheduler.submit(spec("cancelled", &[10, 20], 1)).unwrap();
    scheduler.cancel(&id).unwrap();
    assert_eq!(
        scheduler.request(&id).unwrap().state(),
        RequestState::Cancelled
    );
    assert!(scheduler.schedule().unwrap().is_none());
}

#[test]
fn malformed_result_is_transactional_and_in_flight_cancel_is_rejected() {
    let mut scheduler = scheduler(Fcfs, 2, 2, 2, 2);
    let id = RequestId::from("req-1");
    scheduler.submit(spec("req-1", &[10, 20], 1)).unwrap();
    let batch = scheduler.schedule().unwrap().unwrap();
    assert!(scheduler.cancel(&id).is_err());
    let malformed = ExecutionResult {
        step_id: batch.step_id,
        request_results: vec![RequestExecutionResult {
            request_id: id.clone(),
            generated_token_ids: vec![],
            cached_len_delta: 0,
            is_eos: false,
        }],
    };
    assert!(scheduler.apply_result(malformed).is_err());
    assert_eq!(
        scheduler.request(&id).unwrap().state(),
        RequestState::Running
    );
    scheduler.apply_result(fake_result(&batch)).unwrap();
}
