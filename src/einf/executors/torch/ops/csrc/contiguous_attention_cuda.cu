#include "contiguous_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace einf::ops {

template <typename scalar_t>
__global__ void qk_scores_kernel(
    const scalar_t* Q,
    const scalar_t* K,
    float* scores,
    int64_t q_len,
    int64_t kv_len,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_pos,
    float scale) {
  // TODO(user): compute scaled QK dot products in FP32, map GQA heads,
  // and write -infinity for causally invisible key positions.

  // scores = Q @ K^T, Q -> [q_len, num_attention_heads, head_dim], KV -> [kv_len, num_kv_heads, head_dim]
  // scores -> [q_len, num_attention_heads, kv_len]
  // x -> q_len, y -> kv_len, z -> num_attention_heads

  const int64_t head_q = blockIdx.z;

  if (head_q >= num_attention_heads) {
    return;
  }

  const int64_t gqa_ratio = num_attention_heads / num_kv_heads;
  const int64_t head_kv = head_q / gqa_ratio;

  const int64_t q_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t kv_idx =
      static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;

  if (q_idx >= q_len || kv_idx >= kv_len) {
    return;
  }

  const int64_t total_attention_head_dims =
      num_attention_heads * head_dim;
  const int64_t total_kv_head_dims = num_kv_heads * head_dim;

  const int64_t q_base =
      q_idx * total_attention_head_dims + head_q * head_dim;
  const int64_t kv_base =
      kv_idx * total_kv_head_dims + head_kv * head_dim;

  const int64_t tgt =
      q_idx * num_attention_heads * kv_len + head_q * kv_len + kv_idx;

  const int64_t global_q_idx = start_pos + q_idx;

  if (global_q_idx < kv_idx) {
    scores[tgt] = -1e20f;
    return;
  }

  float sum = 0.0f;
  for (int64_t i = 0; i < head_dim; ++i) {
    sum += static_cast<float>(Q[q_base + i]) * static_cast<float>(K[kv_base + i]);
  }

  scores[tgt] = sum * scale;
}

__global__ void softmax_inplace_kernel(
    float* scores,
    int64_t q_len,
    int64_t num_attention_heads,
    int64_t kv_len) {
  // TODO(user): perform a numerically stable row-wise softmax in place.
  // One row is scores[query_index, attention_head, :].

  // scores -> [q_len, num_attention_heads, kv_len]

  const int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < q_len * num_attention_heads) {
    float cmax = -1e20f;
    float esum = 0;

    int64_t base = tid * kv_len;

    for (int64_t i = 0; i < kv_len; ++i) {
      float s = scores[base + i];

      float new_cmax = fmaxf(cmax, s);

      esum = esum * expf(cmax - new_cmax) + expf(s - new_cmax);
      cmax = new_cmax;
    }

    for (int64_t i = 0; i < kv_len; ++i) {
      float s = scores[base + i];
      scores[base + i] = expf(s - cmax) / esum;
    }
  }
}

template <typename scalar_t>
__global__ void pv_output_kernel(
    const float* probabilities,
    const scalar_t* V,
    scalar_t* output,
    int64_t q_len,
    int64_t kv_len,
    int64_t num_attention_heads,
    int64_t num_kv_heads,
    int64_t head_dim) {
  // TODO(user): accumulate probability * V in FP32 and cast once when
  // writing output. Use the same GQA head mapping as qk_scores_kernel.

  // probs -> [q_len, num_attention_heads, kv_len]
  // V -> [kv_len, num_kv_heads, head_dim]
  // output -> [q_len, num_attention_heads, head_dim]

  const int64_t head_q = blockIdx.z;

  if (head_q >= num_attention_heads) {
    return;
  }

  const int64_t q_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t dim_idx =
      static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t head_kv =
      head_q / (num_attention_heads / num_kv_heads);

  if (q_idx >= q_len || dim_idx >= head_dim) {
    return;
  }

  const int64_t tgt =
      q_idx * num_attention_heads * head_dim + head_q * head_dim + dim_idx;
  const int64_t probs_base =
      q_idx * num_attention_heads * kv_len + head_q * kv_len;

  float sum = 0;
  for (int64_t i = 0; i < kv_len; ++i) {
    const int64_t V_idx =
        i * num_kv_heads * head_dim + head_kv * head_dim + dim_idx;
    sum += static_cast<float>(V[V_idx]) * probabilities[probs_base + i];
  }
  output[tgt] = static_cast<scalar_t>(sum);
}

at::Tensor contiguous_attention_cuda(
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

  auto scores = at::empty(
      {q_len, num_attention_heads, kv_len},
      Q.options().dtype(at::kFloat));
  auto output = at::empty_like(Q);

  const auto stream = at::cuda::getCurrentCUDAStream();

  const dim3 qk_threads(16, 16);
  const dim3 qk_blocks(
      static_cast<unsigned int>((q_len + qk_threads.x - 1) / qk_threads.x),
      static_cast<unsigned int>((kv_len + qk_threads.y - 1) / qk_threads.y),
      static_cast<unsigned int>(num_attention_heads));

  constexpr int softmax_threads = 256;
  const int64_t num_rows = q_len * num_attention_heads;
  const int softmax_blocks =
      static_cast<int>((num_rows + softmax_threads - 1) / softmax_threads);

  const dim3 pv_threads(8, 32);
  const dim3 pv_blocks(
      static_cast<unsigned int>((q_len + pv_threads.x - 1) / pv_threads.x),
      static_cast<unsigned int>((head_dim + pv_threads.y - 1) / pv_threads.y),
      static_cast<unsigned int>(num_attention_heads));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      Q.scalar_type(),
      "contiguous_attention_cuda",
      [&] {
        qk_scores_kernel<scalar_t><<<qk_blocks, qk_threads, 0, stream>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            scores.data_ptr<float>(),
            q_len,
            kv_len,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            start_pos,
            static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        softmax_inplace_kernel
            <<<softmax_blocks, softmax_threads, 0, stream>>>(
                scores.data_ptr<float>(),
                q_len,
                num_attention_heads,
                kv_len);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        pv_output_kernel<scalar_t><<<pv_blocks, pv_threads, 0, stream>>>(
            scores.data_ptr<float>(),
            V.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            q_len,
            kv_len,
            num_attention_heads,
            num_kv_heads,
            head_dim);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return output;
}

}  // namespace einf::ops
