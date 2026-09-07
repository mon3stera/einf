#include "marlin_gemm.h"
#include "marlin/marlin_cuda.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace einf::ops {

void marlin_gemm_cuda(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& s,
    at::Tensor& C,
    at::Tensor& workspace,
    int64_t group_size,
    int64_t max_par) {
  check_marlin_gemm_inputs(A, B, s, C);
  const c10::cuda::CUDAGuard device_guard(A.device());

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = C.size(1);
  TORCH_CHECK(n % 128 == 0, "marlin_gemm requires N divisible by 128, got N=", n);
  TORCH_CHECK(k % 128 == 0, "marlin_gemm requires K divisible by 128, got K=", k);
  TORCH_CHECK(group_size == 128 || group_size == -1,
      "marlin_gemm supports group_size 128 (or -1 per-column), got ", group_size);
  TORCH_CHECK(
      workspace.numel() >= n / 128 * max_par,
      "workspace must hold at least N/128*max_par=",
      n / 128 * max_par,
      " int32 entries");

  // The lock buffer doubles as the cross-slice reduction state; zero it every
  // call instead of relying on the kernel to leave it clean.
  workspace.zero_();

  // One configuration everywhere: decode wants the finest (128, 128) split,
  // and prefill keeps it so a single N%128 contract holds for every einf
  // linear (the stock (64, 256) prefill default needs N%256, which Qwen
  // breaks). The vendored kernel gains (thread_n=128, thread_k=128)
  // instantiations for thread_m_blocks 2-4 to cover M > 16.
  const int thread_k = 128;
  const int thread_n = 128;

  const int groupsize = group_size <= 0 ? -1 : static_cast<int>(group_size);
  const int dev = A.get_device();
  const int err = marlin_cuda(
      A.data_ptr(),
      B.data_ptr(),
      C.data_ptr(),
      s.data_ptr(),
      static_cast<int>(m),
      static_cast<int>(n),
      static_cast<int>(k),
      workspace.data_ptr(),
      groupsize,
      dev,
      at::cuda::getCurrentCUDAStream(dev),
      thread_k,
      thread_n,
      /*sms=*/-1,
      static_cast<int>(max_par));
  TORCH_CHECK(err == 0, "marlin_gemm failed with error code ", err);
}

}  // namespace einf::ops
