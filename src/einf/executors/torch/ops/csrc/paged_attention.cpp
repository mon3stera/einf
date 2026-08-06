#include "paged_attention.h"

#include <torch/library.h>

#include <cmath>

namespace einf::ops {

void check_paged_decode_attention_inputs(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    double scale) {
  TORCH_CHECK(Q.dim() == 2, "Q must have shape [num_attention_heads, head_dim]");
  TORCH_CHECK(K_cache.dim() == 4, "K_cache must have shape [num_blocks, block_len, num_kv_heads, head_dim]");
  TORCH_CHECK(V_cache.sizes() == K_cache.sizes(), "V_cache must match K_cache shape");
  TORCH_CHECK(K_cache.size(0) > 0, "num_blocks must be positive");
  TORCH_CHECK(K_cache.size(1) > 0, "block_len must be positive");
  TORCH_CHECK(K_cache.size(2) > 0, "num_kv_heads must be positive");
  TORCH_CHECK(K_cache.size(3) > 0, "head_dim must be positive");
  TORCH_CHECK(Q.size(0) > 0, "num_attention_heads must be positive");
  TORCH_CHECK(Q.size(1) == K_cache.size(3), "Q head_dim must match K_cache");
  TORCH_CHECK(Q.size(0) % K_cache.size(2) == 0, "num_attention_heads must be divisible by num_kv_heads");

  TORCH_CHECK(Q.scalar_type() == K_cache.scalar_type(), "Q dtype must match K_cache");
  TORCH_CHECK(V_cache.scalar_type() == K_cache.scalar_type(), "V_cache dtype must match K_cache");
  TORCH_CHECK(Q.device() == K_cache.device(), "Q device must match K_cache");
  TORCH_CHECK(V_cache.device() == K_cache.device(), "V_cache device must match K_cache");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K_cache.is_contiguous(), "K_cache must be contiguous");
  TORCH_CHECK(V_cache.is_contiguous(), "V_cache must be contiguous");

  TORCH_CHECK(block_table.dim() == 1, "block_table must have shape [num_logical_blocks]");
  TORCH_CHECK(block_table.scalar_type() == at::kLong, "block_table must have dtype torch.int64");
  TORCH_CHECK(block_table.device() == K_cache.device(), "block_table device must match K_cache");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");

  TORCH_CHECK(context_len > 0, "Paged Decode Attention requires a positive context_len");
  const int64_t block_len = K_cache.size(1);
  const int64_t required_blocks = (context_len + block_len - 1) / block_len;
  TORCH_CHECK(required_blocks <= block_table.size(0), "block_table is too short for context_len");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be finite and positive");
}

at::Tensor paged_decode_attention_cpu(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    double scale) {
  check_paged_decode_attention_inputs(
      Q, K_cache, V_cache, block_table, context_len, scale);
  TORCH_CHECK(!Q.is_cuda(), "CPU kernel received CUDA tensors");
  TORCH_CHECK(
      false,
      "einf::paged_decode_attention CPU kernel is not implemented; Gate 5 focuses on CUDA");
  return {};
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def(
      "paged_decode_attention(Tensor Q, Tensor K_cache, Tensor V_cache, "
      "Tensor block_table, int context_len, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl(
      "paged_decode_attention",
      TORCH_FN(einf::ops::paged_decode_attention_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl(
      "paged_decode_attention",
      TORCH_FN(einf::ops::paged_decode_attention_cuda));
}
