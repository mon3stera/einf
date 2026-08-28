#include "cute_mma_qk.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>

#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>

namespace einf::ops {

namespace {

constexpr int kM = 16;
constexpr int kN = 8;
constexpr int kK = 16;
constexpr int kWarpSize = 32;

template <class QTensor, class KTensor, class ScoreTensor>
__global__ void cute_mma_qk_kernel(
    QTensor Q,
    KTensor K,
    ScoreTensor scores) {
  using namespace cute;

  using MmaAtom =
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
  auto tiled_mma = make_tiled_mma(MmaAtom{});
  auto thread_mma = tiled_mma.get_slice(threadIdx.x);

  auto tQgQ = thread_mma.partition_A(Q);
  auto tKgK = thread_mma.partition_B(K);
  auto tSgS = thread_mma.partition_C(scores);

  auto tQrQ = thread_mma.partition_fragment_A(Q);
  auto tKrK = thread_mma.partition_fragment_B(K);
  auto tSrS = thread_mma.partition_fragment_C(scores);

  copy(tQgQ, tQrQ);
  copy(tKgK, tKrK);
  clear(tSrS);

  gemm(tiled_mma, tQrQ, tKrK, tSrS);
  copy(tSrS, tSgS);
  
  // Learning task 4:
  // 1. Partition Global Q/K/scores with thread_mma.partition_A/B/C().
  // 2. Create Register fragments compatible with those partitions.
  // 3. Copy BF16 Q/K Global partitions into Register fragments.
  // 4. Clear the FP32 accumulator fragment.
  // 5. Execute one cute::gemm() backed by m16n8k16 BF16 Tensor Cores.
  // 6. Copy the FP32 accumulator fragment to Global scores.
  //
  // This exercise has one CTA, one Warp, one MMA atom, and one K step. It
  // deliberately excludes Shared Memory, tiling composition, tails, and loops.
}

}  // namespace

at::Tensor cute_mma_qk_cuda(
    const at::Tensor& Q,
    const at::Tensor& K) {
  using namespace cute;

  const c10::cuda::CUDAGuard device_guard(Q.device());
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(
      properties->major >= 8,
      "einf::cute_mma_qk requires compute capability 8.0 or newer");

  auto scores = at::empty(
      {kM, kN},
      Q.options().dtype(at::kFloat));

  static_assert(sizeof(bfloat16_t) == sizeof(at::BFloat16));
  auto* q_ptr = reinterpret_cast<bfloat16_t*>(Q.data_ptr<at::BFloat16>());
  auto* k_ptr = reinterpret_cast<bfloat16_t*>(K.data_ptr<at::BFloat16>());

  auto q_layout = make_layout(
      make_shape(Int<kM>{}, Int<kK>{}),
      make_stride(Int<kK>{}, Int<1>{}));
  auto k_layout = make_layout(
      make_shape(Int<kN>{}, Int<kK>{}),
      make_stride(Int<kK>{}, Int<1>{}));
  auto score_layout = make_layout(
      make_shape(Int<kM>{}, Int<kN>{}),
      make_stride(Int<kN>{}, Int<1>{}));

  Tensor q_tensor = make_tensor(make_gmem_ptr(q_ptr), q_layout);
  Tensor k_tensor = make_tensor(make_gmem_ptr(k_ptr), k_layout);
  Tensor score_tensor =
      make_tensor(make_gmem_ptr(scores.data_ptr<float>()), score_layout);

  const dim3 blocks(1);
  const dim3 threads(kWarpSize);
  const auto stream = at::cuda::getCurrentCUDAStream(Q.get_device());

  cute_mma_qk_kernel<<<blocks, threads, 0, stream>>>(
      q_tensor,
      k_tensor,
      score_tensor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

}  // namespace einf::ops
