#include "cute_mma_qk.h"

#include <torch/library.h>

namespace einf::ops {

namespace {

void check_cute_mma_qk_inputs(
    const at::Tensor& Q,
    const at::Tensor& K) {
  TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA Tensor");
  TORCH_CHECK(K.is_cuda(), "K must be a CUDA Tensor");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(Q.device() == K.device(), "Q and K must be on the same CUDA device");
  TORCH_CHECK(
      Q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16,
      "cute_mma_qk requires BF16 Q and K");
  TORCH_CHECK(Q.dim() == 2, "Q must have shape [16,16]");
  TORCH_CHECK(K.dim() == 2, "K must have shape [8,16]");
  TORCH_CHECK(
      Q.size(0) == 16 && Q.size(1) == 16,
      "cute_mma_qk requires Q shape [16,16]");
  TORCH_CHECK(
      K.size(0) == 8 && K.size(1) == 16,
      "cute_mma_qk requires K shape [8,16]");
}

}  // namespace

at::Tensor cute_mma_qk_cpu(
    const at::Tensor& Q,
    const at::Tensor& K) {
  TORCH_CHECK(false, "einf::cute_mma_qk is CUDA-only");
  return {};
}

at::Tensor cute_mma_qk(
    const at::Tensor& Q,
    const at::Tensor& K) {
  check_cute_mma_qk_inputs(Q, K);
  return cute_mma_qk_cuda(Q, K);
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def("cute_mma_qk(Tensor Q, Tensor K) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("cute_mma_qk", TORCH_FN(einf::ops::cute_mma_qk_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("cute_mma_qk", TORCH_FN(einf::ops::cute_mma_qk));
}
