#include "kv_cache.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace einf::ops {

// The torch.float8_e4m3fn byte layout is exactly CUDA's __NV_E4M3, so the FP8
// instantiation stores through an __nv_fp8_e4m3 view and lets the satfinite
// constructor clamp out-of-range values to +/-448.
template <typename cache_t, typename input_t>
__global__ void write_slots_kernel(
    cache_t* __restrict__ K_cache,
    cache_t* __restrict__ V_cache,
    const int64_t* __restrict__ slot_mapping,
    const input_t* __restrict__ K,
    const input_t* __restrict__ V,
    int64_t num_tokens,
    int64_t num_kv_heads,
    int64_t head_dim,
    float k_scale,
    float v_scale) {
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

        const float k_value = static_cast<float>(K[tid]) * k_scale;
        const float v_value = static_cast<float>(V[tid]) * v_scale;

        if constexpr (std::is_same_v<cache_t, __nv_fp8_e4m3>) {
            K_cache[tgt_idx] = __nv_fp8_e4m3(k_value);
            V_cache[tgt_idx] = __nv_fp8_e4m3(v_value);
        } else {
            K_cache[tgt_idx] = static_cast<cache_t>(k_value);
            V_cache[tgt_idx] = static_cast<cache_t>(v_value);
        }
    }
}

template <typename cache_t, typename input_t>
void launch_write_slots(
    at::Tensor& K_cache,
    at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t total,
    float k_scale,
    float v_scale) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    write_slots_kernel<cache_t, input_t>
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<cache_t*>(K_cache.data_ptr()),
            reinterpret_cast<cache_t*>(V_cache.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            reinterpret_cast<const input_t*>(K.data_ptr()),
            reinterpret_cast<const input_t*>(V.data_ptr()),
            K.size(0),
            K.size(1),
            K.size(2),
            k_scale,
            v_scale);
}

void write_slots_cuda(
    at::Tensor& K_cache,
    at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V,
    double k_scale,
    double v_scale) {
  check_write_slots_inputs(K_cache, V_cache, slot_mapping, K, V);
  TORCH_CHECK(K_cache.is_cuda(), "K_cache must be a CUDA tensor");
  TORCH_CHECK(V_cache.is_cuda(), "V_cache must be a CUDA tensor");
  TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be a CUDA tensor");
  TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
  TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(K_cache.device());

  const int64_t total = K.numel();
  const float k_scale_f = static_cast<float>(k_scale);
  const float v_scale_f = static_cast<float>(v_scale);

  if (total == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      K.scalar_type(),
      "write_slots_input",
      [&] {
        using input_t = scalar_t;
        if (K_cache.scalar_type() == at::ScalarType::Float8_e4m3fn) {
          launch_write_slots<__nv_fp8_e4m3, input_t>(
              K_cache, V_cache, slot_mapping, K, V, total, k_scale_f, v_scale_f);
          return;
        }
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            K_cache.scalar_type(),
            "write_slots_cache",
            [&] {
              launch_write_slots<scalar_t, input_t>(
                  K_cache, V_cache, slot_mapping, K, V, total, k_scale_f, v_scale_f);
            });
      });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace einf::ops
