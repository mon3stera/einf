#include "cute_shared_copy.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace einf::ops {

namespace {

constexpr int kBlockRows = 128;
constexpr int kBlockCols = 64;

template <class SourceTensor, class DestinationTensor, class ThreadLayout>
__global__ void cute_shared_copy_kernel(
    SourceTensor source,
    DestinationTensor destination,
    ThreadLayout) {
  using namespace cute;

  Tensor source_tile =
      source(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor destination_tile =
      destination(make_coord(_, _), blockIdx.x, blockIdx.y);

  using SharedLayout = decltype(make_layout(
      make_shape(Int<kBlockRows>{}, Int<kBlockCols>{}),
      make_stride(Int<kBlockCols>{}, Int<1>{})));
  constexpr SharedLayout shared_layout{};
  __shared__ float shared_storage[kBlockRows * kBlockCols];
  Tensor shared_tile =
      make_tensor(make_smem_ptr(shared_storage), shared_layout);

  auto partition_src =
      local_partition(source_tile, ThreadLayout{}, threadIdx.x);
  auto partition_shared =
      local_partition(shared_tile, ThreadLayout{}, threadIdx.x);
  auto partition_dst =
      local_partition(destination_tile, ThreadLayout{}, threadIdx.x);

  copy(partition_src, partition_shared);

  __syncthreads();

  copy(partition_shared, partition_dst);
}

}  // namespace

at::Tensor cute_shared_copy_cuda(const at::Tensor& input) {
  using namespace cute;

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = at::empty_like(input);

  const auto tensor_shape = make_shape(input.size(0), input.size(1));
  const auto tensor_layout = make_layout(
      tensor_shape,
      make_stride(input.size(1), Int<1>{}));
  Tensor source = make_tensor(
      make_gmem_ptr(input.data_ptr<float>()),
      tensor_layout);
  Tensor destination = make_tensor(
      make_gmem_ptr(output.data_ptr<float>()),
      tensor_layout);

  const auto block_shape =
      make_shape(Int<kBlockRows>{}, Int<kBlockCols>{});
  Tensor tiled_source = tiled_divide(source, block_shape);
  Tensor tiled_destination = tiled_divide(destination, block_shape);

  using ThreadLayout = decltype(make_layout(
      make_shape(Int<8>{}, Int<32>{}),
      make_stride(Int<32>{}, Int<1>{})));
  constexpr ThreadLayout thread_layout{};

  const dim3 blocks(
      static_cast<unsigned int>(size<1>(tiled_destination)),
      static_cast<unsigned int>(size<2>(tiled_destination)));
  const dim3 threads(static_cast<unsigned int>(size(thread_layout)));
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  cute_shared_copy_kernel<<<blocks, threads, 0, stream>>>(
      tiled_source,
      tiled_destination,
      thread_layout);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace einf::ops
