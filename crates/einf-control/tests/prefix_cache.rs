use einf_control::{
    BatchPlan, ExecutionResult, Fcfs, KvCacheManager, RequestExecutionResult, RequestId,
    RequestSpec, Scheduler, SchedulerConfig, WorkType,
};

fn scheduler(
    blocks: usize,
    block_len: usize,
    max_batch_len: usize,
    max_prefill_chunk_len: usize,
) -> Scheduler<Fcfs> {
    Scheduler::new(
        SchedulerConfig {
            max_batch_len,
            max_prefill_chunk_len,
        },
        Fcfs,
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

fn fake_result(batch: &BatchPlan) -> ExecutionResult {
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

#[test]
fn cache_growth_oom_preserves_existing_allocation() {
    let mut cache = KvCacheManager::new(1, 4).unwrap();
    let id = RequestId::from("request");

    cache.reserve_to(id.clone(), None, 4).unwrap();
    let before = cache.block_table(id.clone());

    assert!(cache.reserve_to(id.clone(), None, 8).is_err());
    assert_eq!(cache.block_table(id), before);
    assert_eq!(cache.free_blocks(), 0);
}

#[test]
fn cache_reuses_longest_contiguous_full_block_prefix() {
    let mut cache = KvCacheManager::new(6, 4).unwrap();
    let owner = RequestId::from("owner");
    let follower = RequestId::from("follower");
    let owner_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let follower_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 99];

    cache
        .reserve_to(owner.clone(), None, owner_prompt.len())
        .unwrap();
    assert_eq!(cache.seal_blocks(owner, &owner_prompt, 0).unwrap(), 8);

    let reuse = cache.plan_reuse(&follower_prompt).unwrap();
    assert_eq!(reuse.reused_len(), 8);
    let reused_blocks = reuse.blocks().to_vec();

    let allocation = cache
        .reserve_to(follower.clone(), Some(reuse), follower_prompt.len())
        .unwrap();
    assert_eq!(&allocation.blocks()[..2], reused_blocks);
    assert_eq!(allocation.blocks().len(), 3);
}

#[test]
fn exact_aligned_prompt_recomputes_then_canonicalizes_last_block() {
    let mut cache = KvCacheManager::new(4, 4).unwrap();
    let owner = RequestId::from("owner");
    let follower = RequestId::from("follower");
    let prompt = [1, 2, 3, 4, 5, 6, 7, 8];

    cache.reserve_to(owner.clone(), None, prompt.len()).unwrap();
    cache.seal_blocks(owner.clone(), &prompt, 0).unwrap();
    let canonical = cache.block_table(owner.clone());

    let reuse = cache.plan_reuse(&prompt).unwrap();
    assert_eq!(reuse.reused_len(), 4);
    assert_eq!(reuse.blocks(), &canonical[..1]);

    cache
        .reserve_to(follower.clone(), Some(reuse), prompt.len())
        .unwrap();
    let during_recompute = cache.block_table(follower.clone());
    assert_eq!(during_recompute[0], canonical[0]);
    assert_ne!(during_recompute[1], canonical[1]);

    cache
        .seal_blocks(follower.clone(), &prompt[4..], 4)
        .unwrap();
    assert_eq!(cache.block_table(follower.clone()), canonical);
    assert_eq!(cache.free_blocks(), 2);

    cache.release(owner).unwrap();
    assert_eq!(cache.free_blocks(), 2);
    assert!(cache.plan_reuse(&prompt).is_some());

    cache.release(follower).unwrap();
    assert_eq!(cache.free_blocks(), 4);
    assert!(cache.plan_reuse(&prompt).is_none());
}

#[test]
fn scheduler_reuses_prefix_from_an_active_request() {
    let mut control = scheduler(8, 4, 16, 16);
    let owner = RequestId::from("owner");
    let follower = RequestId::from("follower");
    let owner_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let follower_prompt = [1, 2, 3, 4, 5, 6, 7, 8, 99];

    control.submit(spec("owner", &owner_prompt, 3)).unwrap();
    let owner_prefill = control.schedule().unwrap().unwrap();
    assert_eq!(owner_prefill.requests[0].start_position, 0);
    control.apply_result(fake_result(&owner_prefill)).unwrap();
    let owner_table = control.cache().block_table(owner.clone());

    control
        .submit(spec("follower", &follower_prompt, 1))
        .unwrap();
    let mixed = control.schedule().unwrap().unwrap();
    let follower_item = mixed
        .requests
        .iter()
        .find(|item| item.request_id == follower)
        .unwrap();

    assert_eq!(follower_item.work_type, WorkType::Prefill);
    assert_eq!(follower_item.start_position, 8);
    assert_eq!(follower_item.input_token_ids, vec![99]);
    assert!(follower_item.need_sample);
    assert_eq!(&follower_item.block_table[..2], &owner_table[..2]);
}
