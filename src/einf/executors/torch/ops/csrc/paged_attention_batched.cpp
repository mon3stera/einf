#include "paged_attention_batched.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>

namespace einf::ops {

namespace {

void check_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA Tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_paged_decode_attention_batched_inputs(
    const torch::Tensor& query,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& single_query_request_indices,
    double scale) {
  check_cuda_contiguous(query, "query");
  check_cuda_contiguous(K_cache, "K_cache");
  check_cuda_contiguous(V_cache, "V_cache");
  check_cuda_contiguous(block_tables, "block_tables");
  check_cuda_contiguous(context_lens, "context_lens");
  check_cuda_contiguous(query_start_loc, "query_start_loc");
  check_cuda_contiguous(
      single_query_request_indices,
      "single_query_request_indices");

  TORCH_CHECK(query.dim() == 3, "query must have shape [T,Hq,D]");
  TORCH_CHECK(
      K_cache.dim() == 4,
      "K_cache must have shape [num_blocks,block_len,Hkv,D]");
  TORCH_CHECK(
      V_cache.sizes() == K_cache.sizes(),
      "V_cache must have the same shape as K_cache");
  TORCH_CHECK(
      block_tables.dim() == 2,
      "block_tables must have shape [num_requests,max_num_blocks]");
  TORCH_CHECK(
      context_lens.dim() == 1,
      "context_lens must have shape [num_requests]");
  TORCH_CHECK(
      query_start_loc.dim() == 1,
      "query_start_loc must have shape [num_requests+1]");
  TORCH_CHECK(
      single_query_request_indices.dim() == 1,
      "single_query_request_indices must have shape [B_decode]");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(
      dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16,
      "query must use FP32, FP16, or BF16 storage");
  TORCH_CHECK(
      K_cache.scalar_type() == dtype && V_cache.scalar_type() == dtype,
      "query, K_cache, and V_cache must have the same dtype");
  TORCH_CHECK(
      block_tables.scalar_type() == at::kLong,
      "block_tables must use torch.int64");
  TORCH_CHECK(
      context_lens.scalar_type() == at::kLong,
      "context_lens must use torch.int64");
  TORCH_CHECK(
      query_start_loc.scalar_type() == at::kLong,
      "query_start_loc must use torch.int64");
  TORCH_CHECK(
      single_query_request_indices.scalar_type() == at::kLong,
      "single_query_request_indices must use torch.int64");

  const auto device = query.device();
  TORCH_CHECK(
      K_cache.device() == device && V_cache.device() == device &&
          block_tables.device() == device && context_lens.device() == device &&
          query_start_loc.device() == device &&
          single_query_request_indices.device() == device,
      "all inputs must be on the same CUDA device");

  const int64_t num_requests = context_lens.size(0);
  TORCH_CHECK(
      block_tables.size(0) == num_requests,
      "block_tables and context_lens must describe the same requests");
  TORCH_CHECK(
      query_start_loc.numel() == num_requests + 1,
      "query_start_loc must contain num_requests+1 entries");
  TORCH_CHECK(
      query.size(2) == K_cache.size(3),
      "query and cache head dimensions must match");
  TORCH_CHECK(
      query.size(1) % K_cache.size(2) == 0,
      "num_attention_heads must be divisible by num_kv_heads");
  TORCH_CHECK(
      K_cache.size(0) > 0 && K_cache.size(1) > 0 && K_cache.size(2) > 0,
      "cache dimensions must be positive");
  TORCH_CHECK(
      block_tables.size(1) > 0 || single_query_request_indices.numel() == 0,
      "block_tables must contain at least one block column");
  TORCH_CHECK(
      std::isfinite(scale) && scale > 0.0,
      "scale must be finite and positive");
}

torch::Tensor paged_decode_attention_batched_cpu(
    const torch::Tensor& query,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& single_query_request_indices,
    double scale) {
  TORCH_CHECK(false, "einf::paged_decode_attention_batched is CUDA-only");
}

torch::Tensor paged_decode_attention_batched(
    const torch::Tensor& query,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& single_query_request_indices,
    double scale) {
  check_paged_decode_attention_batched_inputs(
      query,
      K_cache,
      V_cache,
      block_tables,
      context_lens,
      query_start_loc,
      single_query_request_indices,
      scale);
  return paged_decode_attention_batched_cuda(
      query,
      K_cache,
      V_cache,
      block_tables,
      context_lens,
      query_start_loc,
      single_query_request_indices,
      scale);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def(
      "paged_decode_attention_batched(Tensor query, Tensor K_cache, "
      "Tensor V_cache, Tensor block_tables, Tensor context_lens, "
      "Tensor query_start_loc, Tensor single_query_request_indices, "
      "float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl(
      "paged_decode_attention_batched",
      &paged_decode_attention_batched_cpu);
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl(
      "paged_decode_attention_batched",
      &paged_decode_attention_batched);
}

}  // namespace einf::ops
