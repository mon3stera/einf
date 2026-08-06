#include "paged_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename scalar_t>
__device__ __forceinline__ float2 load_vec2_as_float(const scalar_t* ptr) {
  return make_float2(static_cast<float>(ptr[0]), static_cast<float>(ptr[1]));
}

template<>
__device__ __forceinline__ float2 load_vec2_as_float(const float* ptr) {
  return *reinterpret_cast<const float2*>(ptr);
}

template<>
__device__ __forceinline__ float2 load_vec2_as_float<c10::Half>(const c10::Half* ptr) {
  const __half2 value = *reinterpret_cast<const __half2*>(ptr);
  return __half22float2(value);
}

template<>
__device__ __forceinline__ float2 load_vec2_as_float<c10::BFloat16>(const c10::BFloat16* ptr) {
  const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(ptr);
  return __bfloat1622float2(value);
}

namespace einf::ops {

template <typename scalar_t, int HEAD_DIM, int NUM_WARPS>
__global__ void paged_decode_attention_kernel(
    const scalar_t* Q,
    const scalar_t* K_cache,
    const scalar_t* V_cache,
    const int64_t* block_table,
    scalar_t* output,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t block_len,
    int64_t context_len,
    float scale) {
  // Q -> [Hq, D] (q_len == 1)
  // K/V cache -> [num_blocks, block_len, Hkv, D]
  // block_table -> [num_logical_blocks]
  // output -> [Hq, D]

  // grid.x = num_attention_heads. Multiple warps split one head's Context.
  static_assert(HEAD_DIM <= 256);
  static_assert(HEAD_DIM % 32 == 0);

  __shared__ float partial_m[NUM_WARPS];
  __shared__ float partial_l[NUM_WARPS];
  __shared__ float2 partial_acc[NUM_WARPS][HEAD_DIM / 2];

  constexpr int kScalarSteps = HEAD_DIM / 32;
  constexpr int kVecSteps = (kScalarSteps + 1) / 2;
  constexpr int kNumVecs = HEAD_DIM / 2;

  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  int q_head = blockIdx.x;

  if (q_head >= num_attention_heads) {
    return;
  }

  int head_kv = q_head / (num_attention_heads / num_kv_heads);
  int q_base = q_head * HEAD_DIM;

  float m_g = -INFINITY;
  float l_g = 0.0f;

  float2 acc[kVecSteps] = {};
  float2 q[kVecSteps] = {};

  #pragma unroll
  for (int step = 0; step < kVecSteps; step++) {
    const int vec_idx = lane + step * 32;

    if (vec_idx < kNumVecs) {
      const int dim = vec_idx * 2;
      q[step] = load_vec2_as_float(Q + q_base + dim);
    }
  }

  const int num_logical_blocks = (context_len + block_len - 1) / block_len;
  const int blocks_per_warp = (num_logical_blocks + NUM_WARPS - 1) / NUM_WARPS;
  const int logical_start = warp_id * blocks_per_warp;
  const int logical_end = min(num_logical_blocks, logical_start + blocks_per_warp);

  for (int logical_block = logical_start; logical_block < logical_end; logical_block++) {
    const int64_t physical_block = block_table[logical_block];
    const int valid_tokens = min(block_len, context_len - logical_block * block_len);

    for (int slot_offset = 0; slot_offset < valid_tokens; slot_offset++) {
      float score = 0.0f;

      int64_t kv_base = physical_block * block_len * num_kv_heads * HEAD_DIM
        + slot_offset * num_kv_heads * HEAD_DIM
        + head_kv * HEAD_DIM;

      #pragma unroll
      for (int step = 0; step < kVecSteps; step++) {
        int vec_idx = lane + step * 32;

        if (vec_idx < kNumVecs) {
          const int dim = vec_idx * 2;
          const float2 k = load_vec2_as_float(K_cache + kv_base + dim);
          score += q[step].x * k.x + q[step].y * k.y;
        }
      }

      for (int offset = 16; offset >= 1; offset >>= 1) {
        score += __shfl_down_sync(0xffffffff, score, offset);
      }

      score = score * scale;

      float alpha = 0.0f, beta = 0.0f;
      if (lane == 0) {
        float m_new = fmaxf(score, m_g);
        alpha = expf(m_g - m_new);
        beta = expf(score - m_new);
        l_g = l_g * alpha + beta;
        m_g = m_new;
      }

      alpha = __shfl_sync(0xffffffff, alpha, 0);
      beta = __shfl_sync(0xffffffff, beta, 0);

      #pragma unroll
      for (int step = 0; step < kVecSteps; step++) {
        int vec_idx = lane + step * 32;

        if (vec_idx < kNumVecs) {
          const int dim = vec_idx * 2;
          const float2 v = load_vec2_as_float(V_cache + kv_base + dim);
          acc[step].x = alpha * acc[step].x + beta * v.x;
          acc[step].y = alpha * acc[step].y + beta * v.y;
        }
      }
    }
  }

  if (lane == 0) {
    partial_m[warp_id] = m_g;
    partial_l[warp_id] = l_g;
  }

  #pragma unroll
  for (int step = 0; step < kVecSteps; step++) {
    int vec_idx = lane + step * 32;
    if (vec_idx < kNumVecs) {
      partial_acc[warp_id][vec_idx] = acc[step];
    }
  }

  __syncthreads();

  if (warp_id == 0) {
    float m = -INFINITY;
    float l = 0.0f;
    float2 merged_acc[kVecSteps] = {};

    for (int w = 0; w < NUM_WARPS; w++) {
      float w_l = partial_l[w];

      // warp without context tokens
      if (w_l == 0) {
        continue;
      }

      float alpha = 0.0f;
      float beta = 0.0f;
      float w_m = partial_m[w];

      if (lane == 0) {
        float m_new = fmaxf(m, w_m);
        alpha = expf(m - m_new);
        beta = expf(w_m - m_new);
        l = l * alpha + w_l * beta;
        m = m_new;
      }

      alpha = __shfl_sync(0xffffffff, alpha, 0);
      beta = __shfl_sync(0xffffffff, beta, 0);

      #pragma unroll
      for (int step = 0; step < kVecSteps; step++) {
        int vec_idx = lane + step * 32;
        if (vec_idx < kNumVecs) {
          merged_acc[step].x = alpha * merged_acc[step].x + beta * partial_acc[w][vec_idx].x;
          merged_acc[step].y = alpha * merged_acc[step].y + beta * partial_acc[w][vec_idx].y;
        }
      }
    }

    l = __shfl_sync(0xffffffff, l, 0);

    #pragma unroll
    for (int step = 0; step < kVecSteps; step++) {
      int vec_idx = lane + step * 32;
      if (vec_idx < kNumVecs) {
        const int dim = vec_idx * 2;
        output[q_head * HEAD_DIM + dim] = static_cast<scalar_t>(merged_acc[step].x / l);
        output[q_head * HEAD_DIM + dim + 1] = static_cast<scalar_t>(merged_acc[step].y / l);
      }
    }
  }
}

template <typename scalar_t, int HEAD_DIM, int NUM_WARPS>
void launch_paged_decode_attention_kernel(
    const scalar_t* Q,
    const scalar_t* K_cache,
    const scalar_t* V_cache,
    const int64_t* block_table,
    scalar_t* output,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t block_len,
    int64_t context_len,
    float scale,
    cudaStream_t stream) {
  constexpr int threads = 32 * NUM_WARPS;
  const dim3 blocks(static_cast<unsigned int>(num_attention_heads));
  paged_decode_attention_kernel<scalar_t, HEAD_DIM, NUM_WARPS>
      <<<blocks, threads, 0, stream>>>(
          Q,
          K_cache,
          V_cache,
          block_table,
          output,
          num_attention_heads,
          num_kv_heads,
          block_len,
          context_len,
          scale);
}

at::Tensor paged_decode_attention_cuda(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    double scale) {
  check_paged_decode_attention_inputs(
      Q, K_cache, V_cache, block_table, context_len, scale);
  TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
  TORCH_CHECK(K_cache.is_cuda(), "K_cache must be a CUDA tensor");
  TORCH_CHECK(V_cache.is_cuda(), "V_cache must be a CUDA tensor");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(Q.device());

  auto output = at::empty_like(Q);
  const int64_t num_attention_heads = Q.size(0);
  const int64_t num_kv_heads = K_cache.size(2);
  const int64_t head_dim = Q.size(1);
  const int64_t block_len = K_cache.size(1);

  TORCH_CHECK(
      head_dim <= 256,
      "paged_decode_attention requires head_dim <= 256");
  TORCH_CHECK(
      head_dim % 32 == 0,
      "paged_decode_attention requires head_dim divisible by 32");

  constexpr int num_warps = 4;
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      Q.scalar_type(),
      "paged_decode_attention_cuda",
      [&] {
        switch (head_dim) {
          case 32:
            launch_paged_decode_attention_kernel<scalar_t, 32, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          case 64:
            launch_paged_decode_attention_kernel<scalar_t, 64, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          case 96:
            launch_paged_decode_attention_kernel<scalar_t, 96, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          case 128:
            launch_paged_decode_attention_kernel<scalar_t, 128, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          case 160:
            launch_paged_decode_attention_kernel<scalar_t, 160, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          case 192:
            launch_paged_decode_attention_kernel<scalar_t, 192, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          case 224:
            launch_paged_decode_attention_kernel<scalar_t, 224, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          case 256:
            launch_paged_decode_attention_kernel<scalar_t, 256, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(), num_attention_heads,
                num_kv_heads, block_len, context_len,
                static_cast<float>(scale), stream);
            break;
          default:
            TORCH_CHECK(false, "unsupported head_dim: ", head_dim);
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace einf::ops
