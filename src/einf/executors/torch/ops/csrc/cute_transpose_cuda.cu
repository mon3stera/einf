#include "cute_transpose.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace einf::ops {

namespace {

constexpr int kBlockRows = 64;
constexpr int kBlockCols = 64;
constexpr int kPaddedBlockCols = 65;

template <class SourceTensor, class DestinationTensor, class ThreadLayout>
__global__ void cute_transpose_kernel(
    SourceTensor source,
    DestinationTensor destination,
    ThreadLayout) {
  using namespace cute;

  Tensor source_tile =
      source(make_coord(_, _), blockIdx.x, blockIdx.y);
  Tensor destination_tile =
      destination(make_coord(_, _), blockIdx.y, blockIdx.x);

  using SharedWriteLayout = decltype(make_layout(
      make_shape(Int<kBlockRows>{}, Int<kBlockCols>{}),
      make_stride(Int<kPaddedBlockCols>{}, Int<1>{})));
  using SharedTransposeLayout = decltype(make_layout(
      make_shape(Int<kBlockCols>{}, Int<kBlockRows>{}),
      make_stride(Int<1>{}, Int<kPaddedBlockCols>{})));
  constexpr SharedWriteLayout shared_write_layout{};
  constexpr SharedTransposeLayout shared_transpose_layout{};

  __shared__ float shared_storage[kBlockRows * kPaddedBlockCols];
  Tensor shared_write_tile =
      make_tensor(make_smem_ptr(shared_storage), shared_write_layout);
  Tensor shared_transpose_tile =
      make_tensor(make_smem_ptr(shared_storage), shared_transpose_layout);

  auto tile_src =
      local_partition(source_tile, ThreadLayout{}, threadIdx.x);
  auto tile_dst =
      local_partition(destination_tile, ThreadLayout{}, threadIdx.x);
  auto tile_shared_src =
      local_partition(shared_write_tile, ThreadLayout{}, threadIdx.x);
  auto tile_shared_transpose =
      local_partition(shared_transpose_tile, ThreadLayout{}, threadIdx.x);

  copy(tile_src, tile_shared_src);

  __syncthreads();

  copy(tile_shared_transpose, tile_dst);

  // Both Shared tensors view the same padded storage. The row-major write view
  // uses stride [65,1], while the transposed read view uses [1,65]. The extra
  // column changes a Warp's transposed Shared access from one bank to 32 banks.
}

}  // namespace

at::Tensor cute_transpose_cuda(const at::Tensor& input) {
  using namespace cute;

  const c10::cuda::CUDAGuard device_guard(input.device());
  auto output = at::empty(
      {input.size(1), input.size(0)},
      input.options());

  const auto source_shape = make_shape(input.size(0), input.size(1));
  const auto source_layout = make_layout(
      source_shape,
      make_stride(input.size(1), Int<1>{}));
  const auto destination_shape = make_shape(input.size(1), input.size(0));
  const auto destination_layout = make_layout(
      destination_shape,
      make_stride(input.size(0), Int<1>{}));
  Tensor source = make_tensor(
      make_gmem_ptr(input.data_ptr<float>()),
      source_layout);
  Tensor destination = make_tensor(
      make_gmem_ptr(output.data_ptr<float>()),
      destination_layout);

  const auto block_shape =
      make_shape(Int<kBlockRows>{}, Int<kBlockCols>{});
  Tensor tiled_source = tiled_divide(source, block_shape);
  Tensor tiled_destination = tiled_divide(destination, block_shape);

  using ThreadLayout = decltype(make_layout(
      make_shape(Int<8>{}, Int<32>{}),
      make_stride(Int<32>{}, Int<1>{})));
  constexpr ThreadLayout thread_layout{};

  const dim3 blocks(
      static_cast<unsigned int>(size<1>(tiled_source)),
      static_cast<unsigned int>(size<2>(tiled_source)));
  const dim3 threads(static_cast<unsigned int>(size(thread_layout)));
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  cute_transpose_kernel<<<blocks, threads, 0, stream>>>(
      tiled_source,
      tiled_destination,
      thread_layout);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace einf::ops
