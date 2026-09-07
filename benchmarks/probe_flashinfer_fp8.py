"""Probe: FlashInfer 0.6.4 on SM89 with BF16 Q and FP8-E4M3 paged KV cache.

Validates that plan()/run() accept kv_data_type != q_data_type for both the
prefill wrapper and the tensor-core decode wrapper (including CUDA-graph
capture), and quantifies the numeric gap against a BF16 cache.
"""

import torch
import flashinfer

torch.manual_seed(0)
device = torch.device("cuda")
print("gpu:", torch.cuda.get_device_name(0))

NUM_QO = 14
NUM_KV = 2
HEAD_DIM = 64
PAGE = 16
NUM_PAGES = 64
SCALE = HEAD_DIM ** -0.5

workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
kv_cache_dtype = torch.float8_e4m3fn


def make_cache(dtype):
    k = torch.randn(NUM_PAGES, PAGE, NUM_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(NUM_PAGES, PAGE, NUM_KV, HEAD_DIM, dtype=torch.bfloat16, device=device)
    if dtype is torch.bfloat16:
        return k.contiguous(), v.contiguous()
    return k.to(dtype).contiguous(), v.to(dtype).contiguous()


def build_csr(context_lens):
    lens = torch.tensor(context_lens, dtype=torch.int32)
    pages_per_req = (lens + PAGE - 1) // PAGE
    indptr = torch.zeros(len(context_lens) + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(pages_per_req, 0)
    indices = torch.arange(int(indptr[-1]), dtype=torch.int32)
    last = lens % PAGE
    last = torch.where(last == 0, torch.full_like(last, PAGE), last)
    return indptr.to(device), indices.to(device), last.to(device), lens


def prefill_case(kv_dtype):
    batch = 3
    q_lens = [5, 1, 8]
    context_lens = [35, 1, 20]
    q = torch.randn(sum(q_lens), NUM_QO, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_cache, v_cache = make_cache(kv_dtype)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    indptr, indices, last, _ = build_csr(context_lens)
    qo_indptr = torch.tensor([0, 5, 6, 14], dtype=torch.int32, device=device)
    wrapper.plan(
        qo_indptr, indptr, indices, last,
        NUM_QO, NUM_KV, HEAD_DIM, PAGE,
        causal=True, sm_scale=SCALE,
        q_data_type=torch.bfloat16, kv_data_type=kv_dtype,
    )
    out = wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()
    return out


def decode_case(kv_dtype, use_graph):
    batch = 4
    context_lens = [17, 33, 1, 64]
    q = torch.randn(batch, NUM_QO, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k_cache, v_cache = make_cache(kv_dtype)
    indptr, indices, last, _ = build_csr(context_lens)
    if use_graph:
        indptr_buf = torch.zeros(batch + 1, dtype=torch.int32, device=device)
        indices_buf = torch.zeros(int(indptr[-1]), dtype=torch.int32, device=device)
        last_buf = torch.zeros(batch, dtype=torch.int32, device=device)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_cuda_graph=True, use_tensor_cores=True,
            paged_kv_indptr_buffer=indptr_buf,
            paged_kv_indices_buffer=indices_buf,
            paged_kv_last_page_len_buffer=last_buf,
        )
        wrapper.plan(
            indptr, indices, last, NUM_QO, NUM_KV, HEAD_DIM, PAGE,
            pos_encoding_mode="NONE", sm_scale=SCALE,
            q_data_type=torch.bfloat16, kv_data_type=kv_dtype,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = wrapper.run(q, (k_cache, v_cache))
        graph.replay()
        torch.cuda.synchronize()
        return out
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=True)
    wrapper.plan(
        indptr, indices, last, NUM_QO, NUM_KV, HEAD_DIM, PAGE,
        pos_encoding_mode="NONE", sm_scale=SCALE,
        q_data_type=torch.bfloat16, kv_data_type=kv_dtype,
    )
    out = wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()
    return out


for name, fn, args in [
    ("prefill/bf16", prefill_case, (torch.bfloat16,)),
    ("prefill/fp8", prefill_case, (kv_cache_dtype,)),
    ("decode/bf16", decode_case, (torch.bfloat16, False)),
    ("decode/fp8", decode_case, (kv_cache_dtype, False)),
    ("decode-graph/fp8", decode_case, (kv_cache_dtype, True)),
]:
    try:
        out = fn(*args)
        ok = torch.isfinite(out.float()).all().item()
        print(f"PROBE {name}: shape={tuple(out.shape)} dtype={out.dtype} finite={ok}")
    except Exception as error:
        print(f"PROBE {name}: FAILED: {type(error).__name__}: {error}")

# Numeric gap: identical inputs, only cache storage differs.
ref = prefill_case(torch.bfloat16)
got = prefill_case(kv_cache_dtype)
diff = (ref.float() - got.float()).abs()
denom = ref.float().abs().clamp_min(1e-3)
print(f"NUMERIC prefill fp8-vs-bf16: max_abs={diff.max().item():.4e} mean_abs={diff.mean().item():.4e} max_rel={(diff / denom).max().item():.4e}")

ref = decode_case(torch.bfloat16, False)
got = decode_case(kv_cache_dtype, False)
diff = (ref.float() - got.float()).abs()
print(f"NUMERIC decode fp8-vs-bf16: max_abs={diff.max().item():.4e} mean_abs={diff.mean().item():.4e}")
