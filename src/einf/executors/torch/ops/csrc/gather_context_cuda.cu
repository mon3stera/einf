#include "gather_context.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace einf::ops {

template <typename scalar_t>
__global__ void gather_context_kernel(
    const scalar_t* __restrict__ K_cache,
    const scalar_t* __restrict__ V_cache,
    const int64_t* __restrict__ block_table,
    scalar_t* __restrict__ K,
    scalar_t* __restrict__ V,
    int64_t context_len,
    int64_t block_len,
    int64_t num_kv_heads,
    int64_t head_dim) {
  // K, V cache -> (num_blocks, block_len, num_kv_heads, head_dim)
  // K, V -> (context_len, num_kv_heads, head_dim)

  const int64_t elements_per_token = num_kv_heads * head_dim;
  const int64_t total = context_len * num_kv_heads * head_dim;

  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < total) {
    const int64_t token_idx = tid / elements_per_token;

    const int64_t logical_block_idx = token_idx / block_len;
    const int64_t slot_offset = token_idx % block_len;

    const int64_t physical_block_idx = block_table[logical_block_idx];

    const int64_t flatten_slot_idx =
        physical_block_idx * block_len + slot_offset;
    const int64_t src_idx =
        flatten_slot_idx * elements_per_token + tid % elements_per_token;

    K[tid] = K_cache[src_idx];
    V[tid] = V_cache[src_idx];
  }
}

std::tuple<at::Tensor, at::Tensor> gather_context_cuda(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len) {
  check_gather_context_inputs(K_cache, V_cache, block_table, context_len);
  TORCH_CHECK(K_cache.is_cuda(), "K_cache must be a CUDA tensor");
  TORCH_CHECK(V_cache.is_cuda(), "V_cache must be a CUDA tensor");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(K_cache.device());

  const int64_t num_kv_heads = K_cache.size(2);
  const int64_t head_dim = K_cache.size(3);
  auto K = at::empty({context_len, num_kv_heads, head_dim}, K_cache.options());
  auto V = at::empty({context_len, num_kv_heads, head_dim}, V_cache.options());

  if (context_len == 0) {
    return {K, V};
  }

  const int64_t block_len = K_cache.size(1);
  const int64_t total = context_len * num_kv_heads * head_dim;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      K_cache.scalar_type(),
      "gather_context_cuda",
      [&] {
        gather_context_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(),
                block_table.data_ptr<int64_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                context_len,
                block_len,
                num_kv_heads,
                head_dim);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {K, V};
}

}  // namespace einf::ops
