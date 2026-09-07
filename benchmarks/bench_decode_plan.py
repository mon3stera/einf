"""Micro-benchmark: where does the draft decode plan's host time go?

Times, against the production FlashInferPagedAttention wrapper for the
0.5B draft geometry (14 q heads / 2 kv heads / head_dim 64 / page 32):
  1. plan_decode_bucket with the engine's host-pinned CSR (current path)
  2. plan with DEVICE indptr (the pre-C pattern, expect a sync drain)
  3. wrapper.run alone (the replayed attention)
  4. the Python prelude plan runs per call (to("cpu") + get_seq_lens)
"""

from __future__ import annotations

import time

import torch

from einf.executors.torch.flashinfer_attn import FlashInferPagedAttention

DEVICE = torch.device("cuda")
ITERS = 200


def bench(fn, iters: int = ITERS) -> float:
    fn()  # warmup / capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us per call


def main() -> None:
    attn = FlashInferPagedAttention.create(
        num_qo_heads=14,
        num_kv_heads=2,
        head_dim=64,
        page_size=32,
        device=DEVICE,
        dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        max_nnz=8192,
    )

    page_size = 32
    ctx = 300
    pages = (ctx + page_size - 1) // page_size

    pin = {"pin_memory": True}
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, **pin)
    kv_indptr = torch.tensor([0, pages], dtype=torch.int32, **pin)
    last_page_len = torch.tensor([ctx - (pages - 1) * page_size], dtype=torch.int32, **pin)
    kv_indices_dev = torch.arange(8192, dtype=torch.int32, device=DEVICE)

    def do_plan_direct() -> None:
        attn._active = attn.ensure_decode_wrapper(1)
        attn.ensure_decode_wrapper(1).plan(
            kv_indptr,
            kv_indices_dev[:pages],
            last_page_len,
            14,
            2,
            64,
            page_size,
            pos_encoding_mode="NONE",
            sm_scale=attn.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    t_plan_host = bench(do_plan_direct)
    print(f"plan (pinned-host indptr): {t_plan_host:8.1f} us/call")

    kv_indptr_dev = kv_indptr.to(DEVICE, non_blocking=True)
    lpl_dev = last_page_len.to(DEVICE, non_blocking=True)
    torch.cuda.synchronize()

    def do_plan_device() -> None:
        attn.ensure_decode_wrapper(1).plan(
            kv_indptr_dev,
            kv_indices_dev[:pages],
            lpl_dev,
            14,
            2,
            64,
            page_size,
            pos_encoding_mode="NONE",
            sm_scale=attn.sm_scale,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )

    t_plan_dev = bench(do_plan_device)
    print(f"plan (device indptr, syncs): {t_plan_dev:8.1f} us/call")

    def do_prelude() -> None:
        indptr_host = kv_indptr_dev.to("cpu")
        lpl_host = lpl_dev.to("cpu")
        lens = (indptr_host[1:] - indptr_host[:-1] - 1) * page_size + lpl_host
        return lens

    t_prelude = bench(do_prelude)
    print(f"device->host indptr fetch:   {t_prelude:8.1f} us/call")

    q = torch.randn(1, 14, 1, 64, dtype=torch.bfloat16, device=DEVICE)
    k_cache = torch.randn(256, page_size, 2, 64, dtype=torch.bfloat16, device=DEVICE)
    v_cache = torch.randn_like(k_cache)
    do_plan_direct()
    t_run = bench(lambda: attn.run(q, k_cache, v_cache), 500)
    print(f"run (decode attention):      {t_run:8.1f} us/call")


if __name__ == "__main__":
    main()
