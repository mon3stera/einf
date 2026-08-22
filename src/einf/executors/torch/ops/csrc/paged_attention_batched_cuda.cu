#include "paged_attention_batched.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace einf::ops {

namespace {

constexpr int kNumWarps = 4;
constexpr int kWarpSize = 32;

template <typename scalar_t, int HEAD_DIM, int NUM_WARPS>
__global__ void paged_decode_attention_batched_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ K_cache,
    const scalar_t* __restrict__ V_cache,
    const int64_t* __restrict__ block_tables,
    const int64_t* __restrict__ context_lens,
    const int64_t* __restrict__ query_start_loc,
    const int64_t* __restrict__ single_query_request_indices,
    scalar_t* __restrict__ output,
    int64_t num_blocks,
    int64_t block_len,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t max_num_blocks_per_request,
    float scale) {
  // Learning scaffold:
  //   blockIdx.x selects one Query head.
  //   blockIdx.y selects one compact single-Query batch slot.
  //
  // Map the compact slot back to the packed request and Query row:
  //   request_idx = single_query_request_indices[blockIdx.y]
  //   query_token_idx = query_start_loc[request_idx]
  //
  // Then reuse the existing single-request Vec2 Paged Decode body with:
  //   query[query_token_idx,Hq,D]
  //   block_tables[request_idx,max_num_blocks_per_request]
  //   context_lens[request_idx]
  //
  // Write a compact result to output[blockIdx.y,blockIdx.x,D].
  // The caller will scatter these B_decode rows back into packed token order.
}

template <typename scalar_t, int HEAD_DIM>
void launch_paged_decode_attention_batched(
    const torch::Tensor& query,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& single_query_request_indices,
    torch::Tensor& output,
    float scale,
    cudaStream_t stream) {
  const dim3 threads(kNumWarps * kWarpSize);
  const dim3 blocks(
      static_cast<unsigned int>(query.size(1)),
      static_cast<unsigned int>(single_query_request_indices.numel()));

  paged_decode_attention_batched_kernel<
      scalar_t,
      HEAD_DIM,
      kNumWarps><<<blocks, threads, 0, stream>>>(
      query.data_ptr<scalar_t>(),
      K_cache.data_ptr<scalar_t>(),
      V_cache.data_ptr<scalar_t>(),
      block_tables.data_ptr<int64_t>(),
      context_lens.data_ptr<int64_t>(),
      query_start_loc.data_ptr<int64_t>(),
      single_query_request_indices.data_ptr<int64_t>(),
      output.data_ptr<scalar_t>(),
      K_cache.size(0),
      K_cache.size(1),
      query.size(1),
      K_cache.size(2),
      block_tables.size(1),
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void dispatch_head_dim(
    const torch::Tensor& query,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& single_query_request_indices,
    torch::Tensor& output,
    float scale,
    cudaStream_t stream) {
#define EINF_DISPATCH_HEAD_DIM(HEAD_DIM_VALUE)                              \
  case HEAD_DIM_VALUE:                                                     \
    launch_paged_decode_attention_batched<scalar_t, HEAD_DIM_VALUE>(       \
        query,                                                             \
        K_cache,                                                           \
        V_cache,                                                           \
        block_tables,                                                      \
        context_lens,                                                      \
        query_start_loc,                                                   \
        single_query_request_indices,                                      \
        output,                                                            \
        scale,                                                             \
        stream);                                                           \
    break

  switch (query.size(2)) {
    EINF_DISPATCH_HEAD_DIM(32);
    EINF_DISPATCH_HEAD_DIM(64);
    EINF_DISPATCH_HEAD_DIM(96);
    EINF_DISPATCH_HEAD_DIM(128);
    EINF_DISPATCH_HEAD_DIM(160);
    EINF_DISPATCH_HEAD_DIM(192);
    EINF_DISPATCH_HEAD_DIM(224);
    EINF_DISPATCH_HEAD_DIM(256);
    default:
      TORCH_CHECK(
          false,
          "head_dim must be one of 32,64,96,128,160,192,224,256");
  }

#undef EINF_DISPATCH_HEAD_DIM
}

}  // namespace

torch::Tensor paged_decode_attention_batched_cuda(
    const torch::Tensor& query,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& single_query_request_indices,
    double scale) {
  c10::cuda::CUDAGuard device_guard(query.device());

  auto output = torch::empty(
      {single_query_request_indices.numel(), query.size(1), query.size(2)},
      query.options());
  if (output.numel() == 0) {
    return output;
  }

  TORCH_CHECK(
      false,
      "einf::paged_decode_attention_batched learning scaffold: "
      "implement the packed request/query mapping and reuse the single-request "
      "Vec2 Paged Decode body before enabling execution");

  const auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  if (query.scalar_type() == at::kFloat) {
    dispatch_head_dim<float>(
        query,
        K_cache,
        V_cache,
        block_tables,
        context_lens,
        query_start_loc,
        single_query_request_indices,
        output,
        static_cast<float>(scale),
        stream);
  } else if (query.scalar_type() == at::kHalf) {
    dispatch_head_dim<at::Half>(
        query,
        K_cache,
        V_cache,
        block_tables,
        context_lens,
        query_start_loc,
        single_query_request_indices,
        output,
        static_cast<float>(scale),
        stream);
  } else {
    dispatch_head_dim<at::BFloat16>(
        query,
        K_cache,
        V_cache,
        block_tables,
        context_lens,
        query_start_loc,
        single_query_request_indices,
        output,
        static_cast<float>(scale),
        stream);
  }
  return output;
}

}  // namespace einf::ops
