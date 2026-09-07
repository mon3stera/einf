"""Final validation: paged prefill causal / chain-mask / tree-mask vs correct per-head reference."""
import numpy as np
import torch
import flashinfer

torch.manual_seed(0)
dev = "cuda"
H, D, PREFIX, K, PAGE = 4, 64, 2, 3, 1
n_pages = PREFIX + K
q = torch.randn(K, H, D, device=dev, dtype=torch.float16)
k_cache = torch.randn(n_pages, PAGE, H, D, device=dev, dtype=torch.float16)
v_cache = torch.randn(n_pages, PAGE, H, D, device=dev, dtype=torch.float16)
k, v = k_cache[:, 0], v_cache[:, 0]
S = D ** -0.5

qo_indptr = torch.tensor([0, K], dtype=torch.int32, device=dev)
paged_kv_indptr = torch.tensor([0, n_pages], dtype=torch.int32, device=dev)
paged_kv_indices = torch.arange(n_pages, dtype=torch.int32, device=dev)
last_page_len = torch.tensor([1], dtype=torch.int32, device=dev)
workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=dev)


def ref(mask):
    """Correct per-head reference: mask [K, KV] broadcast over heads."""
    sc = torch.einsum("ihd,jhd->hij", q.float(), k.float()) * S
    sc = sc.masked_fill(~mask.unsqueeze(0), float("-inf"))
    p = torch.softmax(sc, dim=-1)
    return torch.einsum("hij,jhd->ihd", p, v.float())


def run_mask(mask):
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    bits = np.packbits(mask.flatten().cpu().numpy(), bitorder="little")
    packed = torch.from_numpy(bits).to(dev)
    wrapper.plan(qo_indptr, paged_kv_indptr, paged_kv_indices, last_page_len,
                 H, H, D, PAGE, packed_custom_mask=packed, q_data_type=torch.float16)
    out = wrapper.run(q, (k_cache, v_cache))
    torch.cuda.synchronize()
    return out.float()


chain = torch.zeros(K, n_pages, dtype=torch.bool, device=dev)
chain[:, :PREFIX] = True
chain[:, PREFIX:] = torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev))

tree = torch.zeros(K, n_pages, dtype=torch.bool, device=dev)
tree[:, :PREFIX] = True
tree[:, PREFIX + 0] = True
tree[1, PREFIX + 1] = True
tree[2, PREFIX + 2] = True
tree[2, PREFIX + 0] = True  # draft2 parent = draft0; draft2 must NOT see draft1

w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
w.plan(qo_indptr, paged_kv_indptr, paged_kv_indices, last_page_len,
       H, H, D, PAGE, causal=True, q_data_type=torch.float16)
out_causal = w.run(q, (k_cache, v_cache)).float()
torch.cuda.synchronize()

print(f"causal      vs per-head ref: {(out_causal - ref(chain)).abs().max().item():.4e}")
print(f"chain mask  vs per-head ref: {(run_mask(chain) - ref(chain)).abs().max().item():.4e}")
print(f"tree  mask  vs per-head ref: {(run_mask(tree) - ref(tree)).abs().max().item():.4e}")
print(f"causal vs chain-mask outputs: {(out_causal - run_mask(chain)).abs().max().item():.4e}")
