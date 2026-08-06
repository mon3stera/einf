#include "flash_attention.h"

#include "contiguous_attention.h"

#include <torch/library.h>

namespace einf::ops {

at::Tensor flash_attention_cpu(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale) {
  check_contiguous_attention_inputs(Q, K, V, start_pos, scale);
  TORCH_CHECK(!Q.is_cuda(), "CPU kernel received CUDA tensors");
  TORCH_CHECK(
      false,
      "einf::flash_attention CPU kernel is not implemented; Gate 5 focuses on CUDA");
  return {};
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def(
      "flash_attention(Tensor Q, Tensor K, Tensor V, "
      "int start_pos, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl(
      "flash_attention",
      TORCH_FN(einf::ops::flash_attention_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl(
      "flash_attention",
      TORCH_FN(einf::ops::flash_attention_cuda));
}
