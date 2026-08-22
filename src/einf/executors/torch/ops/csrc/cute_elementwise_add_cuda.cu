#include "cute_elementwise_add.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

constexpr int kTileSize = 256;
constexpr int kNumThreads = 256;

namespace einf::ops {

namespace {


template <class XTensor, class YTensor, class OutputTensor>
__global__ void cute_elementwise_add_kernel(
    XTensor X,
    YTensor Y,
    OutputTensor output,
    int64_t numel) {
  using namespace cute;

  auto cta_layout = make_layout(make_shape(Int<kTileSize>{}), make_stride(Int<1>{}));
  auto cY = make_identity_tensor(make_shape(numel));
  
  auto gX_tile = local_tile(X, cta_layout, blockIdx.x);
  auto gY_tile = local_tile(Y, cta_layout, blockIdx.x);
  auto gO_tile = local_tile(output, cta_layout, blockIdx.x);
  auto cY_tile = local_tile(cY, cta_layout, blockIdx.x);

  auto thr_layout = make_layout(make_shape(Int<kNumThreads>{}), make_stride(Int<1>{}));
  auto tXgX = local_partition(gX_tile, thr_layout, threadIdx.x);
  auto tYgY = local_partition(gY_tile, thr_layout, threadIdx.x);
  auto cYgY = local_partition(cY_tile, thr_layout, threadIdx.x);
  auto tOgO = local_partition(gO_tile, thr_layout, threadIdx.x);
 
  for (int i = 0; i < size(tXgX); i++) {
    if (get<0>(cYgY(i)) < numel) {
        tOgO(i) = tXgX(i) + tYgY(i);
    }
  }
}

}  // namespace

at::Tensor cute_elementwise_add_cuda(
    const at::Tensor& X,
    const at::Tensor& Y) {
  using namespace cute;

  const c10::cuda::CUDAGuard device_guard(X.device());
  auto output = at::empty_like(X);
  const int64_t numel = X.numel();

  auto flat_shape = make_shape(numel);
  auto flat_layout = make_layout(flat_shape, make_stride(Int<1>{}));
  Tensor x_tensor =
      make_tensor(make_gmem_ptr(X.data_ptr<float>()), flat_layout);
  Tensor y_tensor =
      make_tensor(make_gmem_ptr(Y.data_ptr<float>()), flat_layout);
  Tensor output_tensor =
      make_tensor(make_gmem_ptr(output.data_ptr<float>()), flat_layout);
  const dim3 block(kNumThreads);
  const dim3 grid(ceil_div(numel, kTileSize));
  const auto stream = at::cuda::getCurrentCUDAStream(X.get_device());

  if (numel != 0) {
    cute_elementwise_add_kernel<<<grid, block, 0, stream>>>(x_tensor, y_tensor, output_tensor, numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return output;
}

}  // namespace einf::ops
