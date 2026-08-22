#include "tensor_core_qk.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace einf::ops {

namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 16;
constexpr int kHeadDim = 64;
constexpr int kNumWarps = 8;
constexpr int kWarpSize = 32;
constexpr int kQueriesPerWarp = 16;

static_assert(kBlockM == kNumWarps * kQueriesPerWarp);

namespace wmma = nvcuda::wmma;

__global__ void tensor_core_qk_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    float* __restrict__ scores,
    int64_t q_len,
    int64_t kv_len,
    int64_t num_attention_heads,
    int64_t num_kv_heads) {
  __shared__ __align__(32) __nv_bfloat16 q_shared[kBlockM * kHeadDim];
  __shared__ __align__(32) __nv_bfloat16 k_shared[kBlockN * kHeadDim];

  using QueryFragment = wmma::fragment<
      wmma::matrix_a,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::row_major>;
  using KeyFragment = wmma::fragment<
      wmma::matrix_b,
      16,
      16,
      16,
      __nv_bfloat16,
      wmma::col_major>;
  using ScoreFragment =
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

  const int thread_idx = threadIdx.x;
  const int warp_id = thread_idx / kWarpSize;
  const int lane = thread_idx % kWarpSize;
  const int64_t q_tile_start =
      static_cast<int64_t>(blockIdx.x) * kBlockM;
  const int64_t q_head = blockIdx.y;
  const int64_t kv_tile_start =
      static_cast<int64_t>(blockIdx.z) * kBlockN;
  const int64_t group_size = num_attention_heads / num_kv_heads;
  const int64_t kv_head = q_head / group_size;
  const int64_t warp_q_local_start = static_cast<int64_t>(warp_id) * kQueriesPerWarp;
  const int64_t warp_q_start =
      q_tile_start + static_cast<int64_t>(warp_id) * kQueriesPerWarp;

  constexpr int kQRowStep = kBlockM / kNumWarps;
  constexpr int kKRowStep = kBlockN / kNumWarps;
  constexpr int kElementsPerBlock = kNumWarps * 32;
  
  for (int i = threadIdx.x; i < kBlockM * kHeadDim; i += kElementsPerBlock) {
    const int row = i / kHeadDim;
    const int col = i % kHeadDim;
    const int64_t q_global_row = q_tile_start + row;
    const int q_local_idx = row * kHeadDim + col;
    
    if (q_global_row < q_len) {
      const int64_t q_global_idx = q_global_row * num_attention_heads * kHeadDim + q_head * kHeadDim + col;
      q_shared[q_local_idx] = Q[q_global_idx];
    } else {
      q_shared[q_local_idx] = __float2bfloat16(0.0f);
    }
  }

  for (int i = threadIdx.x; i < kBlockN * kHeadDim; i += kElementsPerBlock) {
    const int row = i / kHeadDim;
    const int col = i % kHeadDim;
    const int64_t k_global_row = kv_tile_start + row;
    const int k_local_idx = row * kHeadDim + col;

    if (k_global_row < kv_len) {
      const int64_t k_global_idx = k_global_row * num_kv_heads * kHeadDim + kv_head * kHeadDim + col;
      k_shared[k_local_idx] = K[k_global_idx];
    } else {
      k_shared[k_local_idx] = __float2bfloat16(0.0f);
    }
  }

  __syncthreads();

  if (warp_q_start >= q_len) {
    return;
  }

  QueryFragment q_frag;
  KeyFragment k_frag;
  ScoreFragment acc;

  wmma::fill_fragment(acc, 0.0f);
   
  for (int k_step = 0; k_step < kHeadDim / 16; k_step++) {
    const __nv_bfloat16* q_tile_ptr = q_shared + warp_q_local_start * kHeadDim + k_step * 16;
    wmma::load_matrix_sync(q_frag, q_tile_ptr, kHeadDim);

    const __nv_bfloat16* k_tile_ptr = k_shared + k_step * 16;
    wmma::load_matrix_sync(k_frag, k_tile_ptr, kHeadDim);

    wmma::mma_sync(acc, q_frag, k_frag, acc);
  }

  float* write_ptr = scores 
    + (warp_q_start * num_attention_heads + q_head) * kv_len
    + kv_tile_start;
  wmma::store_matrix_sync(write_ptr, acc, kv_len * num_attention_heads, wmma::mem_row_major);
}

}  // namespace

at::Tensor tensor_core_qk_cuda(
    const at::Tensor& Q,
    const at::Tensor& K) {
  const c10::cuda::CUDAGuard device_guard(Q.device());

  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(
      properties->major >= 8,
      "einf::tensor_core_qk requires compute capability 8.0 or newer");

  auto scores = at::empty(
      {Q.size(0), Q.size(1), K.size(0)},
      Q.options().dtype(at::kFloat));

  const dim3 threads(kNumWarps * kWarpSize);
  const dim3 blocks(
      static_cast<unsigned int>((Q.size(0) + kBlockM - 1) / kBlockM),
      static_cast<unsigned int>(Q.size(1)),
      static_cast<unsigned int>(K.size(0) / kBlockN));
  const auto stream = at::cuda::getCurrentCUDAStream(Q.get_device());

  tensor_core_qk_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>()),
      scores.data_ptr<float>(),
      Q.size(0),
      K.size(0),
      Q.size(1),
      K.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

}  // namespace einf::ops
