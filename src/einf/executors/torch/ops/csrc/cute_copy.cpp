#include "cute_copy.h"

#include <torch/library.h>

namespace einf::ops {

namespace {

void check_cute_copy_input(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA Tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must have shape [rows,cols]");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "cute_copy v0 requires FP32 input");
  TORCH_CHECK(input.size(0) > 0, "rows must be positive");
  TORCH_CHECK(input.size(1) > 0, "cols must be positive");
  TORCH_CHECK(
      input.size(0) % 128 == 0,
      "cute_copy v0 requires rows to be divisible by 128");
  TORCH_CHECK(
      input.size(1) % 64 == 0,
      "cute_copy v0 requires cols to be divisible by 64");
}

}  // namespace

at::Tensor cute_copy_cpu(const at::Tensor& input) {
  TORCH_CHECK(false, "einf::cute_copy is CUDA-only");
  return {};
}

at::Tensor cute_copy(const at::Tensor& input) {
  check_cute_copy_input(input);
  return cute_copy_cuda(input);
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def("cute_copy(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("cute_copy", TORCH_FN(einf::ops::cute_copy_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("cute_copy", TORCH_FN(einf::ops::cute_copy));
}
