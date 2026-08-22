#include "cute_elementwise_add.h"

#include <torch/library.h>

namespace einf::ops {

namespace {

void check_cute_elementwise_add_inputs(
    const at::Tensor& X,
    const at::Tensor& Y) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA Tensor");
  TORCH_CHECK(Y.is_cuda(), "Y must be a CUDA Tensor");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(Y.is_contiguous(), "Y must be contiguous");
  TORCH_CHECK(X.device() == Y.device(), "X and Y must be on the same CUDA device");
  TORCH_CHECK(X.scalar_type() == at::kFloat, "X must have dtype FP32");
  TORCH_CHECK(Y.scalar_type() == at::kFloat, "Y must have dtype FP32");
  TORCH_CHECK(X.sizes() == Y.sizes(), "X and Y must have the same shape");
}

}  // namespace

at::Tensor cute_elementwise_add_cpu(
    const at::Tensor& X,
    const at::Tensor& Y) {
  TORCH_CHECK(false, "einf::cute_elementwise_add is CUDA-only");
  return {};
}

at::Tensor cute_elementwise_add(
    const at::Tensor& X,
    const at::Tensor& Y) {
  check_cute_elementwise_add_inputs(X, Y);
  return cute_elementwise_add_cuda(X, Y);
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def("cute_elementwise_add(Tensor X, Tensor Y) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl(
      "cute_elementwise_add",
      TORCH_FN(einf::ops::cute_elementwise_add_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl(
      "cute_elementwise_add",
      TORCH_FN(einf::ops::cute_elementwise_add));
}
