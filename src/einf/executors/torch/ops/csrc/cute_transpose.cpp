#include "cute_transpose.h"

#include <torch/library.h>

namespace einf::ops {

namespace {

void check_cute_transpose_input(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA Tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 2, "input must have shape [rows,cols]");
  TORCH_CHECK(
      input.scalar_type() == at::kFloat,
      "cute_transpose v0 requires FP32 input");
  TORCH_CHECK(input.size(0) > 0, "rows must be positive");
  TORCH_CHECK(input.size(1) > 0, "cols must be positive");
  TORCH_CHECK(
      input.size(0) % 64 == 0,
      "cute_transpose v0 requires rows to be divisible by 64");
  TORCH_CHECK(
      input.size(1) % 64 == 0,
      "cute_transpose v0 requires cols to be divisible by 64");
}

}  // namespace

at::Tensor cute_transpose_cpu(const at::Tensor& input) {
  TORCH_CHECK(false, "einf::cute_transpose is CUDA-only");
  return {};
}

at::Tensor cute_transpose(const at::Tensor& input) {
  check_cute_transpose_input(input);
  return cute_transpose_cuda(input);
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def("cute_transpose(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("cute_transpose", TORCH_FN(einf::ops::cute_transpose_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("cute_transpose", TORCH_FN(einf::ops::cute_transpose));
}
