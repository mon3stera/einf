#include "cute_gemm.h"

#include <torch/library.h>

namespace einf::ops {

namespace {

void check_cute_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& B) {
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA Tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA Tensor");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(A.device() == B.device(), "A and B must be on the same CUDA device");
  TORCH_CHECK(A.scalar_type() == at::kFloat, "A must have dtype FP32");
  TORCH_CHECK(B.scalar_type() == at::kFloat, "B must have dtype FP32");
  TORCH_CHECK(A.dim() == 2, "A must have shape [M,K]");
  TORCH_CHECK(B.dim() == 2, "B must have shape [K,N]");
  TORCH_CHECK(A.size(1) == B.size(0), "A and B must have matching K dimensions");
  TORCH_CHECK(
      A.size(0) % 4 == 0 && A.size(1) % 4 == 0 && B.size(1) % 4 == 0,
      "M, K, and N must be divisible by 4");
}

}  // namespace

at::Tensor cute_gemm_cpu(
    const at::Tensor& A,
    const at::Tensor& B) {
  TORCH_CHECK(false, "einf::cute_gemm is CUDA-only");
  return {};
}

at::Tensor cute_gemm(
    const at::Tensor& A,
    const at::Tensor& B) {
  check_cute_gemm_inputs(A, B);
  return cute_gemm_cuda(A, B);
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def("cute_gemm(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("cute_gemm", TORCH_FN(einf::ops::cute_gemm_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("cute_gemm", TORCH_FN(einf::ops::cute_gemm));
}
