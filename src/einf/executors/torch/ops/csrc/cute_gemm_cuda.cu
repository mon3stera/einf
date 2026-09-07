#include "cute_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <type_traits>

#include <cute/tensor.hpp>

namespace einf::ops {

namespace {

constexpr int kBlockM = 128;
constexpr int kBlockN = 256;
constexpr int kBlockK = 8;
constexpr int kNumThreads = 256;
constexpr int kVectorWidth = 4;
constexpr int kMMATileM = 8;
constexpr int kMMATileN = 32;
constexpr int kStages = 3;
constexpr int kL2TileSize = 8;
// CUTLASS SIMT transposes row-major A into M-major smem with
// simt_transpose_padding(32, BlockK, 32) = 4 so 4x4 lane LDS stays aligned.
constexpr int kPadAM = 4;
constexpr int kSmemAM = kBlockM + kPadAM;
static_assert(kPadAM % kVectorWidth == 0,
              "A M-padding must keep 128-bit LDS alignment");

template <int VectorWidth, class PredTensor, class CoordTensor>
CUTE_DEVICE void fill_copy_predicate(PredTensor &pred, const CoordTensor &tXcX,
                                     int64_t origin_0, int64_t origin_1,
                                     int64_t extent_0, int64_t extent_1,
                                     bool enabled) {
  using namespace cute;

  if (!enabled) {
    clear(pred);
    return;
  }

  CUTE_UNROLL
  for (int i = 0; i < size<0>(pred); ++i) {
    CUTE_UNROLL
    for (int j = 0; j < size<1>(pred); ++j) {
      auto coord = tXcX(0, i, j);
      pred(i, j) = origin_0 + get<0>(coord) < extent_0 &&
                   origin_1 + get<1>(coord) + VectorWidth - 1 < extent_1;
    }
  }
}

template <int BlockM, int BlockN, int BlockK, int Stages, bool Predicated,
          class ATensor, class BTensor, class CTensor>
__global__ void cute_gemm_kernel(ATensor A, BTensor B, CTensor C, int64_t M,
                                 int64_t N, int64_t K, int64_t num_m_tiles,
                                 int64_t num_n_tiles) {
  using namespace cute;
  static_assert(Stages >= 2,
                "CpAsync mainloop follows CUTLASS SM80 and needs Stages >= 2");

  // Group CTAs into an 8-tile L2 macro-tile along the longer grid axis.
  // This changes only traversal order: every logical (block_m, block_n) tile
  // is still visited exactly once, while reusable A/B panels are revisited
  // sooner than with a full row/column sweep.
  const int64_t pid = static_cast<int64_t>(blockIdx.x) +
                      static_cast<int64_t>(gridDim.x) * blockIdx.y;
  int64_t block_m;
  int64_t block_n;
  if (num_m_tiles >= num_n_tiles) {
    const int64_t group_span = int64_t{kL2TileSize} * num_n_tiles;
    const int64_t first_m = (pid / group_span) * kL2TileSize;
    const int64_t actual_m =
        min(int64_t{kL2TileSize}, num_m_tiles - first_m);
    const int64_t pid_in_group = pid % group_span;
    block_m = first_m + pid_in_group % actual_m;
    block_n = pid_in_group / actual_m;
  } else {
    const int64_t group_span = int64_t{kL2TileSize} * num_m_tiles;
    const int64_t first_n = (pid / group_span) * kL2TileSize;
    const int64_t actual_n =
        min(int64_t{kL2TileSize}, num_n_tiles - first_n);
    const int64_t pid_in_group = pid % group_span;
    block_n = first_n + pid_in_group % actual_n;
    block_m = pid_in_group / actual_n;
  }

  __shared__ __align__(16) float shared_A[Stages * kSmemAM * BlockK];
  __shared__ __align__(16) float shared_B[Stages * BlockK * BlockN];

  // A G2S is a transpose: gmem is K-contiguous, smem is M-major, so the
  // 128-bit atom cannot land. Threads walk K (coalesced LDG) and store
  // scalar cp.async. B stays 128-bit into N-major smem.
  using GEMM_Copy_Atom_A = std::conditional_t<
      Predicated, Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<float>, float>,
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<float>, float>>;
  using GEMM_Copy_Atom_B = std::conditional_t<
      Predicated,
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint128_t>, float>,
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, float>>;

  int tid = threadIdx.x;

  static_assert(BlockK % kVectorWidth == 0,
                "BlockK must be a multiple of the 128-bit float4 vector");
  static_assert(kNumThreads % BlockK == 0,
                "BlockK must divide the CTA thread count for the B copy");
  constexpr int kCopyARows = kNumThreads / BlockK;
  constexpr int kCopyAK = BlockK;
  constexpr int kCopyBKrows = BlockK;
  constexpr int kCopyBNvecs = kNumThreads / BlockK;
  constexpr int kCopyBCols = kCopyBNvecs * kVectorWidth;
  constexpr int kValM = BlockM / kMMATileM;
  constexpr int kValN = BlockN / kMMATileN;
  static_assert(kValN == 8, "split-N MMA perm below assumes two float4 groups");
  static_assert(kCopyARows * kCopyAK == kNumThreads);
  static_assert(kCopyBKrows * kCopyBNvecs == kNumThreads);
  static_assert(BlockM % kCopyARows == 0,
                "BlockM must be a multiple of the A copy tile M");
  static_assert(BlockN % kCopyBCols == 0,
                "BlockN must be a multiple of the B copy tile N (1024/BlockK)");

  auto copy_a_thr_layout = make_layout(
      make_shape(Int<kCopyARows>{}, Int<kCopyAK>{}), LayoutRight{});
  auto copy_a_val_layout = Layout<_1>{};
  auto copy_b_thr_layout = make_layout(
      make_shape(Int<kCopyBKrows>{}, Int<kCopyBNvecs>{}), LayoutRight{});
  auto copy_b_val_layout =
      make_layout(make_shape(_1{}, Int<kVectorWidth>{}), LayoutRight{});
  auto mma_thr_layout =
      make_layout(make_shape(Int<kMMATileM>{}, Int<kMMATileN>{}, _1{}),
                  make_stride(Int<kMMATileN>{}, _1{}, _0{}));

  auto tiled_copy_a =
      make_tiled_copy(GEMM_Copy_Atom_A{}, copy_a_thr_layout, copy_a_val_layout);
  auto tiled_copy_b =
      make_tiled_copy(GEMM_Copy_Atom_B{}, copy_b_thr_layout, copy_b_val_layout);

  auto thr_copy_a = tiled_copy_a.get_slice(tid);
  auto thr_copy_b = tiled_copy_b.get_slice(tid);

  auto sA_shape = make_shape(Int<BlockM>{}, Int<BlockK>{}, Int<Stages>{});
  auto sB_shape = make_shape(Int<BlockK>{}, Int<BlockN>{}, Int<Stages>{});
  auto gC_shape = make_shape(Int<BlockM>{}, Int<BlockN>{});
  auto sA_layout = make_layout(
      sA_shape,
      make_stride(_1{}, Int<kSmemAM>{}, Int<kSmemAM * BlockK>{}));
  auto sB_layout = make_layout(
      sB_shape, make_stride(Int<BlockN>{}, _1{}, Int<BlockK * BlockN>{}));

  auto sA_tensor = make_tensor(make_smem_ptr(shared_A), sA_layout);
  auto sB_tensor = make_tensor(make_smem_ptr(shared_B), sB_layout);
  auto gA_shape = make_shape(Int<BlockM>{}, Int<BlockK>{});
  auto gB_shape = make_shape(Int<BlockK>{}, Int<BlockN>{});
  auto cA = make_identity_tensor(gA_shape);
  auto cB = make_identity_tensor(gB_shape);

  auto cC_tensor = make_identity_tensor(make_shape(M, N));
  auto gC_tile = local_tile(C, gC_shape, make_coord(block_m, block_n));
  auto cC_tile =
      local_tile(cC_tensor, gC_shape, make_coord(block_m, block_n));

  // thr=8x32. 16 consecutive M are contiguous in M-major A smem (LDS.128).
  // N is two float4 groups 128 apart so B LDS.128 is a 512B stream.
  auto tiled_mma = make_tiled_mma(
      MMA_Atom<UniversalFMA<float>>{}, mma_thr_layout,
      make_tile(make_layout(Shape<Int<kMMATileM>, Int<kValM>>{},
                            Stride<Int<kValM>, _1>{}),
                make_layout(Shape<Int<kMMATileN>, Shape<_2, _4>>{},
                            Stride<_4, Stride<Int<128>, _1>>{}),
                _));
  auto thr_mma = tiled_mma.get_slice(tid);

  auto tCsA = thr_mma.partition_A(sA_tensor);
  auto sB_mma = make_tensor(
      make_smem_ptr(shared_B),
      make_layout(make_shape(Int<BlockN>{}, Int<BlockK>{}, Int<Stages>{}),
                  make_stride(_1{}, Int<BlockN>{}, Int<BlockK * BlockN>{})));
  auto tCsB = thr_mma.partition_B(sB_mma);
  auto tCgC = thr_mma.partition_C(gC_tile);
  auto tCrC = make_tensor_like(tCgC);
  auto tCcC = thr_mma.partition_C(cC_tile);
  clear(tCrC);

  auto tAsA = thr_copy_a.partition_D(sA_tensor);
  auto tBsB = thr_copy_b.partition_D(sB_tensor);
  auto tAcA = thr_copy_a.partition_S(cA);
  auto tBcB = thr_copy_b.partition_S(cB);
  auto tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)));
  auto tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)));
  // Predicates and identity copy coords are live only on the slow path.

  // Slice K and Stage with compile-time _0 so those modes drop out of the
  // fragment. Runtime 0 leaves a size-1 K mode and make_fragment_like fails.
  auto tCrA0 = make_fragment_like(tCsA(_, _, _0{}, _0{}));
  auto tCrA1 = make_fragment_like(tCsA(_, _, _0{}, _0{}));
  auto tCrB0 = make_fragment_like(tCsB(_, _, _0{}, _0{}));
  auto tCrB1 = make_fragment_like(tCsB(_, _, _0{}, _0{}));

  // CUTLASS SM80 CpAsync collective: unroll prologue to Stages-1, always
  // wait<Stages-2> in the mainloop, and over-run by Stages-1 tiles so the
  // wait path never forks. See sm80_mma_multistage.hpp.
  const int64_t num_k_tiles = ceil_div(K, int64_t{BlockK});
  int k_tile_iter = 0;
  int64_t k_tile_count = num_k_tiles;
  const int64_t origin_m = block_m * BlockM;
  const int64_t origin_n = block_n * BlockN;

  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < Stages - 1; ++k_pipe) {
    const bool enabled = k_tile_count > 0;
    if (enabled) {
      auto gA_tile = local_tile(A, gA_shape, make_coord(block_m, k_tile_iter));
      auto gB_tile = local_tile(B, gB_shape, make_coord(k_tile_iter, block_n));
      auto tAgA = thr_copy_a.partition_S(gA_tile);
      auto tBgB = thr_copy_b.partition_S(gB_tile);
      if constexpr (Predicated) {
        fill_copy_predicate<1>(tApA, tAcA, origin_m,
                               int64_t{k_tile_iter} * BlockK, M, K, true);
        fill_copy_predicate<kVectorWidth>(tBpB, tBcB,
                                          int64_t{k_tile_iter} * BlockK,
                                          origin_n, K, N, true);
        copy_if(tiled_copy_a, tApA, tAgA, tAsA(_, _, _, k_pipe));
        copy_if(tiled_copy_b, tBpB, tBgB, tBsB(_, _, _, k_pipe));
      } else {
        copy(tiled_copy_a, tAgA, tAsA(_, _, _, k_pipe));
        copy(tiled_copy_b, tBgB, tBsB(_, _, _, k_pipe));
      }
    }
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) {
      ++k_tile_iter;
    }
  }

  int smem_pipe_read = 0;
  int smem_pipe_write = Stages - 1;

#pragma unroll 1
  while (k_tile_count > -(Stages - 1)) {
    cp_async_wait<Stages - 2>();
    __syncthreads();

    {
      const bool enabled = k_tile_count > 0;
      if (enabled) {
        auto gA_tile = local_tile(A, gA_shape, make_coord(block_m, k_tile_iter));
        auto gB_tile = local_tile(B, gB_shape, make_coord(k_tile_iter, block_n));
        auto tAgA = thr_copy_a.partition_S(gA_tile);
        auto tBgB = thr_copy_b.partition_S(gB_tile);
        if constexpr (Predicated) {
          fill_copy_predicate<1>(tApA, tAcA, origin_m,
                                 int64_t{k_tile_iter} * BlockK, M, K, true);
          fill_copy_predicate<kVectorWidth>(tBpB, tBcB,
                                            int64_t{k_tile_iter} * BlockK,
                                            origin_n, K, N, true);
          copy_if(tiled_copy_a, tApA, tAgA, tAsA(_, _, _, smem_pipe_write));
          copy_if(tiled_copy_b, tBpB, tBgB, tBsB(_, _, _, smem_pipe_write));
        } else {
          copy(tiled_copy_a, tAgA, tAsA(_, _, _, smem_pipe_write));
          copy(tiled_copy_b, tBgB, tBsB(_, _, _, smem_pipe_write));
        }
      }
      cp_async_fence();
      --k_tile_count;
      if (k_tile_count > 0) {
        ++k_tile_iter;
      }
    }

    // Load k=0 of *this* smem stage only after the wait. Prefetching before
    // cp_async_wait reads stale/unready smem.
    copy(tCsA(_, _, _0{}, smem_pipe_read), tCrA0);
    copy(tCsB(_, _, _0{}, smem_pipe_read), tCrB0);

    CUTE_UNROLL
    for (int k = 0; k < BlockK; ++k) {
      auto &tCrA = (k % 2) ? tCrA1 : tCrA0;
      auto &tCrB = (k % 2) ? tCrB1 : tCrB0;

      if (k + 1 < BlockK) {
        auto &tCrAn = ((k + 1) % 2) ? tCrA1 : tCrA0;
        auto &tCrBn = ((k + 1) % 2) ? tCrB1 : tCrB0;
        copy(tCsA(_, _, k + 1, smem_pipe_read), tCrAn);
        copy(tCsB(_, _, k + 1, smem_pipe_read), tCrBn);
      }

      gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
    }

    smem_pipe_write = smem_pipe_read;
    ++smem_pipe_read;
    smem_pipe_read = (smem_pipe_read == Stages) ? 0 : smem_pipe_read;
  }

  cp_async_wait<0>();
  __syncthreads();

  for (int i = 0; i < size(tCgC); i++) {
    if (get<0>(tCcC(i)) < M && get<1>(tCcC(i)) < N) {
      tCgC(i) = tCrC(i);
    }
  }
}

template <int Stages, bool Predicated, class ATensor, class BTensor,
          class CTensor>
void launch_cute_gemm(ATensor a_tensor, BTensor b_tensor, CTensor c_tensor,
                      int64_t M, int64_t N, int64_t K, int64_t num_m_tiles,
                      int64_t num_n_tiles, dim3 grid, dim3 block,
                      cudaStream_t stream, int device) {
  auto *kernel =
      cute_gemm_kernel<kBlockM, kBlockN, kBlockK, Stages, Predicated, ATensor,
                       BTensor, CTensor>;
  static bool smem_carveout_set = false;
  if (!smem_carveout_set) {
    cudaFuncAttributes func_attr{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&func_attr, kernel));
    int optin_smem = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    const int max_dynamic =
        optin_smem > static_cast<int>(func_attr.sharedSizeBytes)
            ? optin_smem - static_cast<int>(func_attr.sharedSizeBytes)
            : 0;
    if (max_dynamic > 0) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic));
    }
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
    smem_carveout_set = true;
  }
  kernel<<<grid, block, 0, stream>>>(a_tensor, b_tensor, c_tensor, M, N, K,
                                     num_m_tiles, num_n_tiles);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  const int64_t num_m_tiles = ceil_div(M, int64_t{kBlockM});
  const int64_t num_n_tiles = ceil_div(N, int64_t{kBlockN});
  const int64_t num_ctas = num_m_tiles * num_n_tiles;
  constexpr int64_t kMaxGridX = 2147483647;
  const int64_t grid_x = min(num_ctas, kMaxGridX);
  const int64_t grid_y = num_ctas == 0 ? 1 : ceil_div(num_ctas, grid_x);
  const dim3 grid(static_cast<unsigned>(grid_x),
                  static_cast<unsigned>(grid_y));

  if (M != 0 && N != 0) {
    using ATensor = decltype(a_tensor);
    using BTensor = decltype(b_tensor);
    using CTensor = decltype(c_tensor);
    const bool aligned = (M % kBlockM == 0) && (N % kBlockN == 0) &&
                         (K % kBlockK == 0);
    if (aligned) {
      launch_cute_gemm<kStages, false>(a_tensor, b_tensor, c_tensor, M, N, K,
                                       num_m_tiles, num_n_tiles, grid, block,
                                       stream, A.get_device());
    } else {
      launch_cute_gemm<kStages, true>(a_tensor, b_tensor, c_tensor, M, N, K,
                                      num_m_tiles, num_n_tiles, grid, block,
                                      stream, A.get_device());
    }
  }

  return C;
}

} // namespace einf::ops
