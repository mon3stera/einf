#include "cute_reduce_sum.h"

#include <torch/library.h>

namespace einf::ops {

namespace {

void check_cute_reduce_sum_input(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA Tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must have dtype FP32");
}

}  // namespace

at::Tensor cute_reduce_sum_cpu(const at::Tensor& input) {
  TORCH_CHECK(false, "einf::cute_reduce_sum is CUDA-only");
  return {};
}

at::Tensor cute_reduce_sum(const at::Tensor& input) {
  check_cute_reduce_sum_input(input);
  return cute_reduce_sum_cuda(input);
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def("cute_reduce_sum(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("cute_reduce_sum", TORCH_FN(einf::ops::cute_reduce_sum_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("cute_reduce_sum", TORCH_FN(einf::ops::cute_reduce_sum));
}
