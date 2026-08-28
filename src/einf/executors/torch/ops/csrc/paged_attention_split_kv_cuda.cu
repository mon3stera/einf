#include "paged_attention_split_kv.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
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

// Stage 1: one CTA owns one (query_head, Context split).
//
// The learner-authored body should:
//   1. Convert split_idx into a contiguous logical-block interval.
//   2. Let NUM_WARPS partition only that interval.
//   3. Reuse the existing paged QK, online-softmax, and weighted-V logic.
//   4. Merge the NUM_WARPS states inside the CTA.
//   5. Write one FP32 (m, l, acc[HEAD_DIM]) state to Global Memory.
//
// Workspace layout:
//   state_idx = q_head * num_splits + split_idx
//   partial_m[state_idx]
//   partial_l[state_idx]
//   partial_acc[state_idx * HEAD_DIM + dim]
template <typename scalar_t, int HEAD_DIM, int NUM_WARPS>
__global__ void paged_decode_attention_split_kv_partial_kernel(
    const scalar_t* Q,
    const scalar_t* K_cache,
    const scalar_t* V_cache,
    const int64_t* block_table,
    float* partial_m,
    float* partial_l,
    float* partial_acc,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t block_len,
    int64_t context_len,
    int64_t num_splits,
    float scale) {
  // Prefer a quotient/remainder partition so num_splits <= num_logical_blocks gives
  // every split at least one block:
  //   base = num_logical_blocks / num_splits
  //   extra = num_logical_blocks % num_splits
  //   split_count = base + (split_idx < extra)
  //   split_start = split_idx * base + min(split_idx, extra)
  // A ceil-sized partition can leave final splits empty even when S <= N.
  // partial_m / partial_l -> [Hq, num_splits]
  // partial_acc -> [Hq, num_splits, head_dim]
    static_assert(HEAD_DIM <= 256);
    static_assert(HEAD_DIM % 32 == 0);

    __shared__ float block_partial_m[NUM_WARPS];
    __shared__ float block_partial_l[NUM_WARPS];
    __shared__ float2 block_partial_acc[NUM_WARPS][HEAD_DIM / 2];

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
    const int base_blocks = num_logical_blocks / num_splits;
    const int extra_blocks = num_logical_blocks % num_splits;
    const int block_this_cta = base_blocks + (blockIdx.y < extra_blocks ? 1 : 0);
    const int block_logical_start = blockIdx.y * base_blocks + min(blockIdx.y, extra_blocks);
    const int base_warp_blocks = block_this_cta / NUM_WARPS;
    const int extra_warp_blocks = block_this_cta % NUM_WARPS;
    const int block_this_warp = base_warp_blocks + (warp_id < extra_warp_blocks ? 1 : 0);
    const int warp_logical_start = block_logical_start
      + warp_id * base_warp_blocks
      + min(warp_id, extra_warp_blocks);
    const int warp_logical_end = warp_logical_start + block_this_warp;

    for (int logical_block = warp_logical_start; logical_block < warp_logical_end; logical_block++) {
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
      block_partial_m[warp_id] = m_g;
      block_partial_l[warp_id] = l_g;
    }

    #pragma unroll
    for (int step = 0; step < kVecSteps; step++) {
      int vec_idx = lane + step * 32;
      if (vec_idx < kNumVecs) {
        block_partial_acc[warp_id][vec_idx] = acc[step];
      }
    }

    __syncthreads();

    if (warp_id == 0) {
      float m = -INFINITY;
      float l = 0.0f;
      float2 merged_acc[kVecSteps] = {};

      for (int w = 0; w < NUM_WARPS; w++) {
        float w_l = block_partial_l[w];

        // warp without context tokens
        if (w_l == 0) {
          continue;
        }

        float alpha = 0.0f;
        float beta = 0.0f;
        float w_m = block_partial_m[w];

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
            merged_acc[step].x = alpha * merged_acc[step].x + beta * block_partial_acc[w][vec_idx].x;
            merged_acc[step].y = alpha * merged_acc[step].y + beta * block_partial_acc[w][vec_idx].y;
          }
        }
      }

      const int64_t state_idx = q_head * num_splits + blockIdx.y;

      if (lane == 0) {
        partial_m[state_idx] = m;
        partial_l[state_idx] = l;
      }

      for (int step = 0; step < kVecSteps; step++) {
        const int vec_idx = lane + step * 32;
        if (vec_idx < kNumVecs) {
          const int dim = vec_idx * 2;
          const int acc_base = state_idx * HEAD_DIM;
          partial_acc[acc_base + dim] = merged_acc[step].x;
          partial_acc[acc_base + dim + 1] = merged_acc[step].y;
        }
      }
    }
  }


// Stage 2: one CTA owns one Query head and merges all Context splits.
//
// For every non-empty split state (m_s, l_s, acc_s), preserve the same
// online-softmax merge invariant already used by the four-Warp baseline:
//   m_new = max(m, m_s)
//   alpha = exp(m - m_new)
//   beta = exp(m_s - m_new)
//   l = alpha * l + beta * l_s
//   acc = alpha * acc + beta * acc_s
// Finally write output = acc / l in scalar_t.
template <typename scalar_t, int HEAD_DIM>
__global__ void paged_decode_attention_split_kv_reduce_kernel(
    const float* partial_m,
    const float* partial_l,
    const float* partial_acc,
    scalar_t* output,
    int64_t num_attention_heads,
    int64_t num_splits) {
  // TODO(learner): merge the per-split FP32 states and normalize once.
  // partial_m / partial_l -> [Hq, num_splits]
  // partial_acc -> [Hq, num_splits, head_dim]
  // output -> [Hq, head_dim]

  int q_head = blockIdx.x;

  if (q_head >= num_attention_heads) {
    return;
  }

  int lane = threadIdx.x % 32;

  constexpr int kVecSteps = (HEAD_DIM / 32 + 1) / 2;

  float2 acc[kVecSteps] = {};

  float m = -INFINITY;
  float l = 0.0f;

  for (int n = 0; n < num_splits; n++) {
    const int64_t ml_idx = blockIdx.x * num_splits + n;
    float l_n = partial_l[ml_idx];

    if (l_n == 0) {
      continue;
    }

    float m_n = partial_m[ml_idx];

    float alpha = 0.0f, beta = 0.0f;
    if (lane == 0) {
      float m_new = fmaxf(m_n, m);
      alpha = expf(m - m_new);
      beta = expf(m_n - m_new);
      l = l * alpha + l_n * beta;
      m = m_new;
    }

    alpha = __shfl_sync(0xffffffff, alpha, 0);
    beta = __shfl_sync(0xffffffff, beta, 0);

    const int64_t acc_base = ml_idx * HEAD_DIM;

    #pragma unroll
    for (int step = 0; step < kVecSteps; step++) {
      int vec_idx = lane + step * 32;
      if (vec_idx < HEAD_DIM / 2) {
        const int dim = vec_idx * 2;
        const float2 v = load_vec2_as_float(partial_acc + acc_base + dim);
        acc[step].x = acc[step].x * alpha + v.x * beta;
        acc[step].y = acc[step].y * alpha + v.y * beta;
      }
    }
  }

  l = __shfl_sync(0xffffffff, l, 0);

  const int64_t o_base = q_head * HEAD_DIM;

  #pragma unroll
  for (int step = 0; step < kVecSteps; step++) {
    int vec_idx = lane + step * 32;
    if (vec_idx < HEAD_DIM / 2) {
      const int dim = vec_idx * 2;
      output[o_base + dim] = static_cast<scalar_t>(acc[step].x / l);
      output[o_base + dim + 1] = static_cast<scalar_t>(acc[step]. y / l);
    }
  }
}

template <typename scalar_t, int HEAD_DIM, int NUM_WARPS>
void launch_paged_decode_attention_split_kv_kernels(
    const scalar_t* Q,
    const scalar_t* K_cache,
    const scalar_t* V_cache,
    const int64_t* block_table,
    float* partial_m,
    float* partial_l,
    float* partial_acc,
    scalar_t* output,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t block_len,
    int64_t context_len,
    int64_t num_splits,
    float scale,
    cudaStream_t stream) {
  constexpr int partial_threads = 32 * NUM_WARPS;
  const dim3 partial_blocks(
      static_cast<unsigned int>(num_attention_heads),
      static_cast<unsigned int>(num_splits));
  paged_decode_attention_split_kv_partial_kernel<
      scalar_t,
      HEAD_DIM,
      NUM_WARPS><<<partial_blocks, partial_threads, 0, stream>>>(
      Q,
      K_cache,
      V_cache,
      block_table,
      partial_m,
      partial_l,
      partial_acc,
      num_attention_heads,
      num_kv_heads,
      block_len,
      context_len,
      num_splits,
      scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int reduce_threads = 32;
  const dim3 reduce_blocks(static_cast<unsigned int>(num_attention_heads));
  paged_decode_attention_split_kv_reduce_kernel<scalar_t, HEAD_DIM>
      <<<reduce_blocks, reduce_threads, 0, stream>>>(
          partial_m,
          partial_l,
          partial_acc,
          output,
          num_attention_heads,
          num_splits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor paged_decode_attention_split_kv_cuda(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    int64_t num_splits,
    double scale) {
  check_paged_decode_attention_split_kv_inputs(
      Q,
      K_cache,
      V_cache,
      block_table,
      context_len,
      num_splits,
      scale);
  TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
  TORCH_CHECK(K_cache.is_cuda(), "K_cache must be a CUDA tensor");
  TORCH_CHECK(V_cache.is_cuda(), "V_cache must be a CUDA tensor");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(Q.device());

  const int64_t num_attention_heads = Q.size(0);
  const int64_t num_kv_heads = K_cache.size(2);
  const int64_t head_dim = Q.size(1);
  const int64_t block_len = K_cache.size(1);
  TORCH_CHECK(
      head_dim <= 256,
      "paged_decode_attention_split_kv requires head_dim <= 256");
  TORCH_CHECK(
      head_dim % 32 == 0,
      "paged_decode_attention_split_kv requires head_dim divisible by 32");

  auto output = at::empty_like(Q);
  const auto workspace_options = Q.options().dtype(at::kFloat);
  auto partial_m = at::empty(
      {num_attention_heads, num_splits}, workspace_options);
  auto partial_l = at::empty_like(partial_m);
  auto partial_acc = at::empty(
      {num_attention_heads, num_splits, head_dim}, workspace_options);

  constexpr int num_warps = 4;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      Q.scalar_type(),
      "paged_decode_attention_split_kv_cuda",
      [&] {
        switch (head_dim) {
          case 32:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 32, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          case 64:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 64, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          case 96:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 96, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          case 128:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 128, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          case 160:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 160, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          case 192:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 192, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          case 224:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 224, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          case 256:
            launch_paged_decode_attention_split_kv_kernels<
                scalar_t, 256, num_warps>(
                Q.data_ptr<scalar_t>(), K_cache.data_ptr<scalar_t>(),
                V_cache.data_ptr<scalar_t>(), block_table.data_ptr<int64_t>(),
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
                partial_acc.data_ptr<float>(), output.data_ptr<scalar_t>(),
                num_attention_heads, num_kv_heads, block_len, context_len,
                num_splits, static_cast<float>(scale), stream);
            break;
          default:
            TORCH_CHECK(false, "unsupported head_dim: ", head_dim);
        }
      });

  return output;
}

}  // namespace einf::ops
