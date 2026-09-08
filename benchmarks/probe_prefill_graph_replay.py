"""Feasibility probe: CUDA-graph replay of a prefill-shaped masked forward
across growing kv lengths.

A draft level forward has a fixed qo (static level width) but its kv grows
every step. Graphing it requires planning ONCE at a bucket's maximum kv and
replaying at smaller actual kv: the captured tile schedule covers kv tiles
beyond the actual context, which must be neutralized by the custom mask
(False bits) plus zero-filled cache rows.

This probe captures a qo=4 masked paged prefill at kv=64 and replays it at
smaller kv values (updating indptr/last_page_len/mask tokens), comparing
against a fresh plan+run at each actual kv. Match => graph buckets for the
tree level forwards are feasible with the existing FlashInfer kernels.
"""

from __future__ import annotations

import sys

import torch

import flashinfer

sys.path.insert(0, "src")

from einf.executors.torch.spec_tree import pack_mask_flashinfer

DEVICE = torch.device("cuda")


def build_mask(qo: int, kv: int) -> torch.Tensor:
    """Row r sees all kv columns up to a causal-ish boundary (r+1)."""
    mask = torch.zeros(qo, kv, dtype=torch.bool)

    for r in range(qo):
        mask[r, : kv - qo + r + 1] = True

    return mask


def main() -> None:
    torch.manual_seed(11)

    qo, kv_max, page = 4, 64, 32
    heads, kvh, dim = 14, 2, 64

    q_static = torch.randn(qo, heads, dim, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = torch.randn(1, 2, page * 4, kvh, dim, dtype=torch.bfloat16, device=DEVICE)

    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    pages_max = (kv_max + page - 1) // page
    mask_buf = torch.zeros(
        qo * kv_max // 8 + 1, dtype=torch.uint8, device=DEVICE
    )

    # FlashInfer's CUDA-graph mode: persistent buffers the captured kernels
    # read at replay time, so plan() can stay outside the graph and replays
    # need no re-planning.
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        qo_indptr_buf=torch.zeros(2, dtype=torch.int32, device=DEVICE),
        paged_kv_indptr_buf=torch.zeros(2, dtype=torch.int32, device=DEVICE),
        paged_kv_indices_buf=torch.zeros(8, dtype=torch.int32, device=DEVICE),
        paged_kv_last_page_len_buf=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        custom_mask_buf=mask_buf,
        mask_indptr_buf=torch.zeros(2, dtype=torch.int32, device=DEVICE),
    )

    def plan_and_fill(kv: int) -> None:
        """Fill the static mask buffer and plan at the given kv."""
        mask = build_mask(qo, kv)
        packed = pack_mask_flashinfer(mask).to(DEVICE)
        mask_buf.zero_()
        mask_buf[: packed.numel()] = packed
        wrapper.plan(
            qo_indptr=torch.tensor([0, qo], dtype=torch.int32, device=DEVICE),
            paged_kv_indptr=torch.tensor(
                [0, (kv + page - 1) // page], dtype=torch.int32, device=DEVICE
            ),
            paged_kv_indices=torch.arange(
                (kv + page - 1) // page, dtype=torch.int32, device=DEVICE
            ),
            paged_kv_last_page_len=torch.tensor(
                [kv - ((kv + page - 1) // page - 1) * page],
                dtype=torch.int32,
                device=DEVICE,
            ),
            num_qo_heads=heads,
            num_kv_heads=kvh,
            head_dim_qk=dim,
            head_dim_vo=dim,
            page_size=page,
            causal=False,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            packed_custom_mask=mask_buf,
        )

    # ---- eager reference at several kv values ----
    references: dict[int, torch.Tensor] = {}

    for kv in (64, 40, 30):
        plan_and_fill(kv)
        references[kv] = wrapper.run(
            q_static, kv_cache[:, :, : (kv + page - 1) // page * page]
        ).clone()

    # ---- capture at kv_max ----
    plan_and_fill(kv_max)

    static_out = torch.empty(qo, heads, dim, dtype=torch.bfloat16, device=DEVICE)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        for _ in range(2):
            static_out.copy_(wrapper.run(q_static, kv_cache))

    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out.copy_(wrapper.run(q_static, kv_cache))

    print("captured at kv=64")

    # ---- replay at smaller kv: refill ONLY the mask (False beyond the
    # actual kv); the captured plan stays at bucket-max kv and its extra
    # tiles are neutralized by the mask plus zero cache rows ----
    for kv in (40, 30):
        mask = build_mask(qo, kv)
        packed = pack_mask_flashinfer(mask).to(DEVICE)
        mask_buf.zero_()
        mask_buf[: packed.numel()] = packed
        graph.replay()
        torch.cuda.synchronize()

        ref = references[kv]
        diff = (static_out.float() - ref.float()).abs()
        per_row = diff.amax(dim=(1, 2))
        argmax_match = (
            static_out.float().argmax(-1) == ref.float().argmax(-1)
        ).all().item()
        print(
            f"replay at kv={kv}: max diff {diff.max().item():.4f} "
            f"per-row {[f'{v:.3f}' for v in per_row.tolist()]} "
            f"argmax match {argmax_match}"
        )

    # sanity: replay at kv_max itself must be exact
    plan_and_fill(kv_max)
    graph.replay()
    torch.cuda.synchronize()
    diff = (static_out.float() - references[kv_max].float()).abs().max()
    print(f"replay at kv=64 (capture kv): max diff {diff.item():.4f}")

    # three-way localization: is the WRAPPER state intact after capture?
    plan_and_fill(kv_max)
    eager_after = wrapper.run(
        q_static, kv_cache[:, :, :pages_max * page]
    ).clone()
    torch.cuda.synchronize()
    d_ref = (eager_after.float() - references[kv_max].float()).abs().max()
    d_replay = (eager_after.float() - static_out.float()).abs().max()
    print(
        f"eager-after-capture vs reference: {d_ref.item():.4f}; "
        f"vs replay: {d_replay.item():.4f}"
    )


if __name__ == "__main__":
    main()
