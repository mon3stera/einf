#include "flash_attention.h"

#include "contiguous_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace einf::ops {

template <typename scalar_t, int BLOCK_M, int BLOCK_N, int HEAD_DIM, int NUM_WARPS>
__global__ __launch_bounds__(256, 2) void flash_attention_forward_kernel(
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

      int q_tile_idx = blockIdx.x;
      int q_head_idx = blockIdx.y;
      int warp_id = threadIdx.x / 32;
      int lane = threadIdx.x % 32;

      constexpr int Q_ROWS_PER_WARP = BLOCK_M / NUM_WARPS;
      constexpr int kv_tile_size = BLOCK_N * HEAD_DIM;

      if (q_head_idx >= num_attention_heads || q_tile_idx >= (q_len + BLOCK_M - 1) / BLOCK_M) {
        return;
      }

      __shared__ scalar_t q_shared[BLOCK_M * HEAD_DIM];
      __shared__ scalar_t k_shared[BLOCK_N * HEAD_DIM];
      __shared__ scalar_t v_shared[BLOCK_N * HEAD_DIM];

      static_assert(BLOCK_M % NUM_WARPS == 0);
      constexpr int ELEMENTS_PER_LANE = (HEAD_DIM + 31) / 32;

      float m_g[Q_ROWS_PER_WARP];
      float l_g[Q_ROWS_PER_WARP] = {0.0f};
      float acc[Q_ROWS_PER_WARP][ELEMENTS_PER_LANE] = {0.0f};

      #pragma unroll
      for (int i = 0; i < Q_ROWS_PER_WARP; i++) {
        m_g[i] = -INFINITY;
      }

      #pragma unroll
      for (int i = 0; i < Q_ROWS_PER_WARP; i++) {
        int q_local_row = warp_id * Q_ROWS_PER_WARP + i;
        int64_t q_global_row = q_tile_idx * BLOCK_M + i + warp_id * Q_ROWS_PER_WARP;
        int q_shared_base = q_local_row * HEAD_DIM;

        // Step 1: load Q to shared memory
        if (q_global_row < q_len) {
          int64_t q_row_base = q_global_row * num_attention_heads * head_dim + q_head_idx * head_dim;
          for (int i = lane; i < HEAD_DIM; i += 32) {
            q_shared[q_shared_base + i] = Q[q_row_base + i];
          }
        } else {
          for (int i = lane; i < HEAD_DIM; i += 32) {
            q_shared[q_shared_base + i] = 0.0f;
          }
        }
      }

      __syncthreads();

      int head_kv = q_head_idx / (num_attention_heads / num_kv_heads);
      int num_threads = blockDim.x;

      const int64_t q_tile_start =
          static_cast<int64_t>(q_tile_idx) * BLOCK_M;
      const int64_t q_tile_end =
          q_tile_start + BLOCK_M < q_len
          ? q_tile_start + BLOCK_M
          : q_len;
      const int64_t last_absolute_q_idx =
          start_pos + q_tile_end - 1;
      const int64_t kv_end =
          last_absolute_q_idx + 1 < kv_len
          ? last_absolute_q_idx + 1
          : kv_len;
      for (int i = 0; i < kv_end; i += BLOCK_N) {
        // Step 2: load KV to shared memory
        int tid = threadIdx.x;

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

        #pragma unroll
        for (int row = 0; row < Q_ROWS_PER_WARP; row++) {
          int q_local_row = warp_id * Q_ROWS_PER_WARP + row;
          int64_t q_global_row = q_tile_idx * BLOCK_M + q_local_row;

          const int64_t absolute_query_pos = start_pos + q_global_row;
          if (i > absolute_query_pos) {
            continue;
          }

          float S[BLOCK_N] = {0.0f};

          for (int j = 0; j < BLOCK_N; j++) {
            float sum = 0.0f;

            for (int k = lane; k < HEAD_DIM; k += 32) {
              sum = fmaf(
                  static_cast<float>(q_shared[q_local_row * HEAD_DIM + k]),
                  static_cast<float>(k_shared[j * HEAD_DIM + k]),
                  sum);
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

          #pragma unroll
          for (int j = 0; j < BLOCK_N; j++) {
            if (S[j] > m_j) {
              m_j = S[j];
            }
          }

          float m_new = fmaxf(m_j, m_g[row]);
          float sum_j = 0.0f;

          #pragma unroll
          for (int j = 0; j < BLOCK_N; j++) {
            const float prob = expf(S[j] - m_new);
            sum_j += prob;
            S[j] = prob;
          }

          float alpha = expf(m_g[row] - m_new);
          float l_new = l_g[row] * alpha + sum_j;

          #pragma unroll
          for (int step = 0; step < ELEMENTS_PER_LANE; step++) {
            int d = lane + step * 32;

            if (d < head_dim) {
              acc[row][step] *= alpha;

              float pv_sum = 0.0f;

              for (int k = 0; k < BLOCK_N; k++) {
                pv_sum = fmaf(
                    S[k],
                    static_cast<float>(v_shared[k * HEAD_DIM + d]),
                    pv_sum);
              }

              acc[row][step] += pv_sum;
            }
          }

          m_g[row] = m_new;
          l_g[row] = l_new;
        }

        __syncthreads();
      }

      #pragma unroll
      for (int row = 0; row < Q_ROWS_PER_WARP; row++) {
        int q_local_row = warp_id * Q_ROWS_PER_WARP + row;
        int64_t out_global_row = q_tile_idx * BLOCK_M + q_local_row;

        if (out_global_row < q_len) {
          int64_t out_row_base = out_global_row * num_attention_heads * head_dim + q_head_idx * HEAD_DIM;

          #pragma unroll
          for (int step = 0; step < ELEMENTS_PER_LANE; step++) {
            int d = lane + step * 32;

            if (d < HEAD_DIM) {
              output[out_row_base + d] = static_cast<scalar_t>(acc[row][step] / l_g[row]);
            }
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

  constexpr int block_m = 16;
  constexpr int block_n = 16;
  constexpr int compile_head_dim = 64;
  constexpr int num_warps = 8;
  constexpr int warp_size = 32;

  TORCH_CHECK(
      head_dim == 64,
      "einf::flash_attention v0 requires head_dim == 64, got ",
      head_dim);

  const dim3 threads(warp_size * num_warps);
  const dim3 blocks(
      static_cast<unsigned int>((q_len + block_m - 1) / block_m),
      static_cast<unsigned int>(num_attention_heads));
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      Q.scalar_type(),
      "flash_attention_cuda",
      [&] {
        flash_attention_forward_kernel<
            scalar_t,
            block_m,
            block_n,
            compile_head_dim,
            num_warps>
            <<<blocks, threads, 0, stream>>>(
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
