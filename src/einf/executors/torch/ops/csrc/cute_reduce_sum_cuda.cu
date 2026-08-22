#include "cute_reduce_sum.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace einf::ops {

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kThreads / kWarpSize;

template <class InputTensor>
__global__ void cute_reduce_sum_kernel(
    InputTensor input,
    float* output,
    int64_t numel) {
  using namespace cute;

  __shared__ float warp_sums[kNumWarps];
  
  auto cI = make_identity_tensor(make_shape(numel));

  auto cta_layout = make_layout(make_shape(numel), LayoutRight{});
  auto gI_tile = local_tile(input, cta_layout, blockIdx.x);
  auto cI_tile = local_tile(cI, cta_layout, blockIdx.x);
  
  auto thr_layout = make_layout(make_shape(Int<kThreads>{}), LayoutRight{});
  auto tIgI = local_partition(gI_tile, thr_layout, threadIdx.x);
  auto cIgI = local_partition(cI_tile, thr_layout, threadIdx.x);

  float sum = 0.0f;
  for (int i = 0; i < size(tIgI); i++) {
    if (get<0>(cIgI(i)) < numel) {
      sum += tIgI(i);
    }
  }

  #pragma unroll 
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }

  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  if (lane == 0) {
    warp_sums[warp_id] = sum;
  }

  __syncthreads();

  if (warp_id == 0) {
    float s = lane < kNumWarps ? warp_sums[lane] : 0.0f;
    
    for (int offset = 8; offset > 0; offset >>= 1) {
      s += __shfl_down_sync(0xffffffff, s, offset);
    }

    if (lane == 0) {
      *output = s;
    }
  }
  }
}  // namespace

at::Tensor cute_reduce_sum_cuda(const at::Tensor& input) {
  using namespace cute;

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = at::zeros({}, input.options());
  const int64_t numel = input.numel();

  auto flat_shape = make_shape(numel);
  auto flat_layout = make_layout(flat_shape, make_stride(Int<1>{}));
  Tensor input_tensor =
      make_tensor(make_gmem_ptr(input.data_ptr<float>()), flat_layout);
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  const dim3 block(kThreads);
  const dim3 grid(1);

  if (numel != 0) {
    cute_reduce_sum_kernel<<<grid, block, 0, stream>>>(input_tensor, output.data_ptr<float>(), numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return output;
}

}  // namespace einf::ops
