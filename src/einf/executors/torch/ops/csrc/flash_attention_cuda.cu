#include "flash_attention.h"

#include "contiguous_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace einf::ops {

template <typename scalar_t, int BLOCK_M, int BLOCK_N>
__global__ void flash_attention_forward_kernel(
    const scalar_t* Q,
    const scalar_t* K,
    const scalar_t* V,
    scalar_t* output,
    int64_t q_len,
    int64_t kv_len,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_pos,
    float scale) {
      // BLOCK_M for Q, BLOCK_N for K, V
      // Q -> [q_len, num_attention_heads, head_dim]
      // K, V -> [kv_len, num_kv_heads, head_dim]
      // total slots -> q (BLOCK_M * head_dim) + k (BLOCK_N * head_dim) + v (BLOCK_N * head_dim)
      // each block -> layout: (?, 32) [q_len, head_dim] * [kv_len, head_dim]^T

      extern __shared__ float shared[];

      int q_tile_idx = blockIdx.x;
      int q_head_idx = blockIdx.y;

      const int kv_tile_size = BLOCK_N * head_dim;

      if (q_head_idx >= num_attention_heads || q_tile_idx >= (q_len + BLOCK_M - 1) / BLOCK_M) {
        return;
      }

      float* q_shared = shared;
      float* k_shared = shared + BLOCK_M * head_dim;
      float* v_shared = shared + BLOCK_M * head_dim + BLOCK_N * head_dim;

      int64_t q_global_row = q_tile_idx * BLOCK_M + threadIdx.y;
      int q_shared_base = threadIdx.y * head_dim;

      // Step 1: load Q to shared memory
      if (q_global_row < q_len) {
        int64_t q_row_base = q_global_row * num_attention_heads * head_dim + q_head_idx * head_dim;
        for (int i = threadIdx.x; i < head_dim; i += 32) {
          q_shared[q_shared_base + i] = Q[q_row_base + i];
        }
      } else {
        for (int i = threadIdx.x; i < head_dim; i += 32) {
          q_shared[q_shared_base + i] = 0.0f;
        }
      }

      __syncthreads();

      int head_kv = q_head_idx / (num_attention_heads / num_kv_heads);
      int num_threads = blockDim.x * blockDim.y;

      float m_i = -1e20f;
      float l_i = 0.0f;

      // max head_dim = 256
      float accum[8] = {0.0f};
      float S[BLOCK_N];
      int element_per_thread = (head_dim + 31) / 32;

      for (int i = 0; i < kv_len; i += BLOCK_N) {
        // Step 2: load KV to shared memory
        int tid = threadIdx.y * blockDim.x + threadIdx.x;

        for (int j = tid; j < kv_tile_size; j += num_threads) {
          int row = j / head_dim;
          int col = j % head_dim;

          int kv_global_row = i + row;

          if (kv_global_row < kv_len) {
            int64_t kv_global_idx = kv_global_row * num_kv_heads * head_dim + head_kv * head_dim + col;
            k_shared[j] = K[kv_global_idx];
            v_shared[j] = V[kv_global_idx];
          } else {
            k_shared[j] = 0.0f;
            v_shared[j] = 0.0f;
          }
        }

        __syncthreads();

        for (int j = 0; j < BLOCK_N; j++) {
          float sum = 0.0f;

          for (int k = threadIdx.x; k < head_dim; k += 32) {
            sum += q_shared[threadIdx.y * head_dim + k] * k_shared[j * head_dim + k];
          }

          for (int offset = 16; offset >= 1; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
          }

          const int64_t absolute_query_pos = start_pos + q_global_row;
          const int64_t global_key_idx = i + j;

          if (global_key_idx <= absolute_query_pos && global_key_idx < kv_len) {
            S[j] = sum * scale;
          } else {
            S[j] = -1e20f;
          }
        }

        float m_j = -1e20f;
        for (int j = 0; j < BLOCK_N; j++) {
          if (S[j] > m_j) {
            m_j = S[j];
          }
        }

        float m_new = fmaxf(m_j, m_i);
        float sum_j = 0.0f;
        for (int j = 0; j < BLOCK_N; j++) {
          sum_j += expf(S[j] - m_new);
          S[j] = expf(S[j] - m_new);
        }

        float alpha = expf(m_i - m_new);
        float l_new = l_i * alpha + sum_j;

        for (int step = 0; step < element_per_thread; step++) {
          int d = threadIdx.x + step * 32;

          if (d < head_dim) {
            accum[step] *= alpha;

            float pv_sum = 0.0f;

            for (int k = 0; k < BLOCK_N; k++) {
              pv_sum += S[k] * v_shared[k * head_dim + d];
            }

            accum[step] += pv_sum;
          }
        }

        m_i = m_new;
        l_i = l_new;

        __syncthreads();
      }

      int64_t out_global_row = q_tile_idx * BLOCK_M + threadIdx.y;

      if (out_global_row < q_len) {
        int64_t out_row_base = out_global_row * num_attention_heads * head_dim + q_head_idx * head_dim;

        for (int step = 0; step < element_per_thread; step++) {
          int d = threadIdx.x + step * 32;

          if (d < head_dim) {
            output[out_row_base + d] = static_cast<scalar_t>(accum[step] / l_i);
          }
        }
      }
}

at::Tensor flash_attention_cuda(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale) {
  check_contiguous_attention_inputs(Q, K, V, start_pos, scale);
  TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
  TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
  TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
  const c10::cuda::CUDAGuard device_guard(Q.device());

  const int64_t q_len = Q.size(0);
  const int64_t num_attention_heads = Q.size(1);
  const int64_t head_dim = Q.size(2);
  const int64_t kv_len = K.size(0);
  const int64_t num_kv_heads = K.size(1);
  auto output = at::empty_like(Q);

  constexpr int block_m = 4;
  constexpr int block_n = 16;
  constexpr int warp_size = 32;

  TORCH_CHECK(
      head_dim == 64,
      "einf::flash_attention v0 requires head_dim == 64, got ",
      head_dim);

  const dim3 threads(warp_size, block_m);
  const dim3 blocks(
      static_cast<unsigned int>((q_len + block_m - 1) / block_m),
      static_cast<unsigned int>(num_attention_heads));
  const size_t shared_memory_bytes =
      static_cast<size_t>(block_m + 2 * block_n) *
      static_cast<size_t>(head_dim) * sizeof(float);
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      Q.scalar_type(),
      "flash_attention_cuda",
      [&] {
        flash_attention_forward_kernel<scalar_t, block_m, block_n>
            <<<blocks, threads, shared_memory_bytes, stream>>>(
                Q.data_ptr<scalar_t>(),
                K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                q_len,
                kv_len,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                start_pos,
                static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

}  // namespace einf::ops
