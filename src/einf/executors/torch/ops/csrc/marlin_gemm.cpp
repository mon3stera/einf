#include "marlin_gemm.h"

#include <torch/library.h>

namespace einf::ops {

void check_marlin_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& s,
    const at::Tensor& C) {
  TORCH_CHECK(A.dim() == 2, "A must have shape [M, K]");
  TORCH_CHECK(A.scalar_type() == at::kHalf, "A must be fp16; cast at the module boundary");
  TORCH_CHECK(B.dim() == 2 && B.scalar_type() == at::kInt, "B must be int32 [K/16, N*2]");
  TORCH_CHECK(s.dim() == 2 && s.scalar_type() == at::kHalf, "s must be fp16 [K/groupsize, N]");
  TORCH_CHECK(C.dim() == 2 && C.scalar_type() == at::kHalf, "C must be fp16 [M, N]");
  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = C.size(1);
  TORCH_CHECK(B.size(0) == k / 16 && B.size(1) == 2 * n, "B shape must be [K/16, N*2]");
  TORCH_CHECK(s.size(1) == n, "scales N must match C");
  TORCH_CHECK(k % s.size(0) == 0, "K must be divisible by the scale group count");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && s.is_contiguous() && C.is_contiguous(),
      "marlin_gemm tensors must be contiguous");
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && s.is_cuda() && C.is_cuda(),
      "marlin_gemm requires CUDA tensors");
  TORCH_CHECK(A.device() == B.device() && A.device() == s.device() && A.device() == C.device(),
      "marlin_gemm tensors must share one device");
}

void marlin_gemm_cpu(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& s,
    at::Tensor& C,
    at::Tensor& workspace,
    int64_t group_size,
    int64_t max_par) {
  check_marlin_gemm_inputs(A, B, s, C);
  (void)workspace;
  (void)group_size;
  (void)max_par;
  TORCH_CHECK(
      false,
      "einf::marlin_gemm is CUDA-only; the reference path is the dequantized "
      "F.linear on CPU");
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def(
      "marlin_gemm(Tensor A, Tensor b_qweight, Tensor scales, Tensor(a!) out, "
      "Tensor(a!) workspace, *, int group_size, int max_par=8) -> ()");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("marlin_gemm", TORCH_FN(einf::ops::marlin_gemm_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("marlin_gemm", TORCH_FN(einf::ops::marlin_gemm_cuda));
}
