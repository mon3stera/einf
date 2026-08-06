#include "kv_cache.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace einf::ops {

template<typename scalar_t>
__global__ void write_slots_kernel(
    scalar_t* __restrict__ K_cache,
    scalar_t* __restrict__ V_cache,
    const int64_t* __restrict__ slot_mapping,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    // K, V cache -> (num_blocks, block_len, num_kv_heads, head_dim)
    // K, V -> (num_tokens, num_kv_heads, head_dim)

    const int64_t elements_per_token = num_kv_heads * head_dim;
    const int64_t total = num_tokens * elements_per_token;

    const int64_t tid =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (tid < total) {
        const int64_t token_idx = tid / elements_per_token;

        const int64_t slot_idx = slot_mapping[token_idx];

        if (slot_idx < 0) {
            return;
        }

        const int64_t tgt_idx =
            slot_idx * elements_per_token + tid % elements_per_token;

        K_cache[tgt_idx] = K[tid];
        V_cache[tgt_idx] = V[tid];
    }
}

void write_slots_cuda(
    at::Tensor& K_cache,
    at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V) {
  check_write_slots_inputs(K_cache, V_cache, slot_mapping, K, V);
  TORCH_CHECK(K_cache.is_cuda(), "K_cache must be a CUDA tensor");
  TORCH_CHECK(V_cache.is_cuda(), "V_cache must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
  TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(K_cache.device());

  const int64_t num_tokens = K.size(0);
  const int64_t num_kv_heads = K.size(1);
  const int64_t head_dim = K.size(2);
  const int64_t total = num_tokens * num_kv_heads * head_dim;

  if (total == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      K.scalar_type(),
      "write_slots_cuda",
      [&] {
        write_slots_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int64_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                num_tokens,
                num_kv_heads,
                head_dim);
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace einf::ops
