#include "cute_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace einf::ops {

namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kBlockK = 16;
constexpr int kNumThreads = 256;
constexpr int kCopyTileAM = 64;
constexpr int kCopyTileAK = 16;
constexpr int kCopyTileBK = 16;
constexpr int kCopyTileBN = 64;
constexpr int kMMATileM = 16;
constexpr int kMMATileN = 16;
constexpr int kStages = 2;

template <int VectorWidth, class ThrCopy, class GlobalTile, class CoordTile>
CUTE_DEVICE auto build_copy_source_and_predicate(
    const ThrCopy &thr_copy, const GlobalTile &global_tile,
    const CoordTile &coord_tile, int64_t extent_0, int64_t extent_1) {
  using namespace cute;

  auto tXgX = thr_copy.partition_S(global_tile);
  auto tXcX = thr_copy.partition_S(coord_tile);
  auto tXpX =
      make_tensor<bool>(make_shape(size<1>(tXgX), size<2>(tXgX)));

  for (int i = 0; i < size<0>(tXpX); ++i) {
    for (int j = 0; j < size<1>(tXpX); ++j) {
      auto coord = tXcX(0, i, j);
      const int64_t coord_0 = get<0>(coord);
      const int64_t coord_1 = get<1>(coord);
      tXpX(i, j) = coord_0 < extent_0 &&
                    coord_1 + VectorWidth - 1 < extent_1;
    }
  }

  return make_tuple(tXpX, tXgX);
}

template <int BlockM, int BlockN, int BlockK, class ATensor, class BTensor,
          class CTensor>
__global__ void cute_gemm_kernel(ATensor A, BTensor B, CTensor C, int64_t M,
                                 int64_t N, int64_t K) {
  using namespace cute;

  __shared__ __align__(16) float shared_A[kStages * BlockM * BlockK];
  __shared__ __align__(16) float shared_B[kStages * BlockK * BlockN];

  constexpr int kVectorWidth = 4;

  using GEMM_Copy_Atom =
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, float>;

  int tid = threadIdx.x;

  auto copy_a_thr_layout = make_layout(
      make_shape(Int<kCopyTileAM>{}, Int<kCopyTileAK / kVectorWidth>{}),
      LayoutRight{});
  auto copy_a_val_layout =
      make_layout(make_shape(_1{}, Int<kVectorWidth>{}), LayoutRight{});
  auto copy_b_thr_layout = make_layout(
      make_shape(Int<kCopyTileBK>{}, Int<kCopyTileBN / kVectorWidth>{}),
      LayoutRight{});
  auto copy_b_val_layout =
      make_layout(make_shape(_1{}, Int<kVectorWidth>{}), LayoutRight{});
  auto mma_thr_layout =
      make_layout(make_shape(Int<kMMATileM>{}, Int<kMMATileN>{}, _1{}),
                  make_stride(Int<kMMATileN>{}, _1{}, _0{}));

  auto tiled_copy_a =
      make_tiled_copy(GEMM_Copy_Atom{}, copy_a_thr_layout, copy_a_val_layout);
  auto tiled_copy_b =
      make_tiled_copy(GEMM_Copy_Atom{}, copy_b_thr_layout, copy_b_val_layout);

  auto thr_copy_a = tiled_copy_a.get_slice(tid);
  auto thr_copy_b = tiled_copy_b.get_slice(tid);

  auto sA_shape = make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<kStages>{});
  auto sB_shape = make_shape(Int<BlockK>{}, Int<BlockN>{}, Int<kStages>{});
  auto gA_shape = make_shape(Int<BlockM>{}, Int<BlockK>{});
  auto gB_shape = make_shape(Int<BlockK>{}, Int<BlockN>{});
  auto gC_shape = make_shape(Int<BlockM>{}, Int<BlockN>{});
  auto sA_layout = make_layout(
      sA_shape, make_stride(Int<BlockK>{}, _1{}, Int<BlockM * BlockK>{}));
  auto sB_layout = make_layout(
      sB_shape, make_stride(Int<BlockN>{}, _1{}, Int<BlockK * BlockN>{}));

  auto sA_tensor = make_tensor(make_smem_ptr(shared_A), sA_layout);
  auto cA_tensor = make_identity_tensor(make_shape(M, K));
  auto sB_tensor = make_tensor(make_smem_ptr(shared_B), sB_layout);
  auto cB_tensor = make_identity_tensor(make_shape(K, N));

  auto cC_tensor = make_identity_tensor(make_shape(M, N));
  auto gC_tile = local_tile(C, gC_shape, make_coord(blockIdx.x, blockIdx.y));
  auto cC_tile =
      local_tile(cC_tensor, gC_shape, make_coord(blockIdx.x, blockIdx.y));

  auto tiled_mma =
      make_tiled_mma(MMA_Atom<UniversalFMA<float>>{}, mma_thr_layout);
  auto thr_mma = tiled_mma.get_slice(tid);

  auto tCsA = thr_mma.partition_A(sA_tensor);
  auto sB_mma = make_tensor(
      make_smem_ptr(shared_B),
      make_layout(make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<kStages>{}),
                  make_stride(_1{}, Int<BlockN>{}, Int<BlockK * BlockN>{})));
  auto tCsB = thr_mma.partition_B(sB_mma);
  auto tCgC = thr_mma.partition_C(gC_tile);
  auto tCrC = make_tensor_like(tCgC);
  auto tCcC = thr_mma.partition_C(cC_tile);
  clear(tCrC);

  auto tAsA = thr_copy_a.partition_D(sA_tensor);
  auto tBsB = thr_copy_b.partition_D(sB_tensor);
  auto tCrA = make_fragment_like(tCsA(_, _, _, 0));
  auto tCrB = make_fragment_like(tCsB(_, _, _, 0));

  int write_stage = 0;
  int read_stage = 0;
  int64_t k_next = 1;
  const int64_t num_k_tiles = ceil_div(K, int64_t{BlockK});

  // Prologue: populate the first read stage. Each following iteration issues
  // the next Global-to-Shared copy before computing the current register tile.
  if (num_k_tiles > 0) {
    auto gA_tile = local_tile(A, gA_shape, make_coord(blockIdx.x, 0));
    auto gB_tile = local_tile(B, gB_shape, make_coord(0, blockIdx.y));
    auto cA_tile = local_tile(cA_tensor, gA_shape, make_coord(blockIdx.x, 0));
    auto cB_tile = local_tile(cB_tensor, gB_shape, make_coord(0, blockIdx.y));

    auto [tApA, tAgA] = build_copy_source_and_predicate<kVectorWidth>(
        thr_copy_a, gA_tile, cA_tile, M, K);
    auto [tBpB, tBgB] = build_copy_source_and_predicate<kVectorWidth>(
        thr_copy_b, gB_tile, cB_tile, K, N);

    copy_if(tiled_copy_a, tApA, tAgA, tAsA(_, _, _, 0));
    copy_if(tiled_copy_b, tBpB, tBgB, tBsB(_, _, _, 0));

    cp_async_fence();
    write_stage = 1;
  }

  for (int64_t k = 0; k < num_k_tiles; ++k) {
    cp_async_wait<0>();
    __syncthreads();

    if (k_next < num_k_tiles) {
      auto gA_tile = local_tile(A, gA_shape, make_coord(blockIdx.x, k_next));
      auto gB_tile = local_tile(B, gB_shape, make_coord(k_next, blockIdx.y));
      auto cA_tile =
          local_tile(cA_tensor, gA_shape, make_coord(blockIdx.x, k_next));
      auto cB_tile =
          local_tile(cB_tensor, gB_shape, make_coord(k_next, blockIdx.y));

      auto [tApA, tAgA] = build_copy_source_and_predicate<kVectorWidth>(
          thr_copy_a, gA_tile, cA_tile, M, K);
      auto [tBpB, tBgB] = build_copy_source_and_predicate<kVectorWidth>(
          thr_copy_b, gB_tile, cB_tile, K, N);

      copy_if(tiled_copy_a, tApA, tAgA, tAsA(_, _, _, write_stage));
      copy_if(tiled_copy_b, tBpB, tBgB, tBsB(_, _, _, write_stage));

      cp_async_fence();
      k_next++;
    }

    copy(tCsA(_, _, _, read_stage), tCrA);
    copy(tCsB(_, _, _, read_stage), tCrB);

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    write_stage = read_stage;
    read_stage ^= 1;

    __syncthreads();
  }

  for (int i = 0; i < size(tCgC); i++) {
    if (get<0>(tCcC(i)) < M && get<1>(tCcC(i)) < N) {
      tCgC(i) = tCrC(i);
    }
  }
}

} // namespace

at::Tensor cute_gemm_cuda(const at::Tensor &A, const at::Tensor &B) {
  using namespace cute;

  const c10::cuda::CUDAGuard device_guard(A.device());
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(1);
  auto C = at::empty({M, N}, A.options());

  auto a_layout = make_layout(make_shape(M, K), make_stride(K, Int<1>{}));
  auto b_layout = make_layout(make_shape(K, N), make_stride(N, Int<1>{}));
  auto c_layout = make_layout(make_shape(M, N), make_stride(N, Int<1>{}));
  Tensor a_tensor = make_tensor(make_gmem_ptr(A.data_ptr<float>()), a_layout);
  Tensor b_tensor = make_tensor(make_gmem_ptr(B.data_ptr<float>()), b_layout);
  Tensor c_tensor = make_tensor(make_gmem_ptr(C.data_ptr<float>()), c_layout);
  const auto stream = at::cuda::getCurrentCUDAStream(A.get_device());

  const dim3 block(kNumThreads);
  const dim3 grid(ceil_div(M, kBlockM), ceil_div(N, kBlockN));

  if (M != 0 && N != 0) {
    cute_gemm_kernel<kBlockM, kBlockN, kBlockK>
        <<<grid, block, 0, stream>>>(a_tensor, b_tensor, c_tensor, M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  return C;
}

} // namespace einf::ops
