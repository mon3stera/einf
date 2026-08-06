#include "contiguous_attention.h"

#include <torch/library.h>

#include <cmath>

namespace einf::ops {

void check_contiguous_attention_inputs(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale) {
  TORCH_CHECK(Q.dim() == 3, "Q must have shape [q_len, num_attention_heads, head_dim]");
  TORCH_CHECK(K.dim() == 3, "K must have shape [kv_len, num_kv_heads, head_dim]");
  TORCH_CHECK(V.sizes() == K.sizes(), "V must match K shape");
  TORCH_CHECK(Q.size(0) > 0, "q_len must be positive");
  TORCH_CHECK(K.size(0) > 0, "kv_len must be positive");
  TORCH_CHECK(Q.size(1) > 0, "num_attention_heads must be positive");
  TORCH_CHECK(K.size(1) > 0, "num_kv_heads must be positive");
  TORCH_CHECK(Q.size(2) > 0, "head_dim must be positive");
  TORCH_CHECK(Q.size(2) == K.size(2), "Q and K head_dim must match");
  TORCH_CHECK(Q.size(1) % K.size(1) == 0, "num_attention_heads must be divisible by num_kv_heads");

  TORCH_CHECK(Q.is_floating_point(), "Q/K/V must use a floating-point dtype");
  TORCH_CHECK(Q.scalar_type() == K.scalar_type(), "Q and K dtype must match");
  TORCH_CHECK(V.scalar_type() == K.scalar_type(), "V and K dtype must match");
  TORCH_CHECK(Q.device() == K.device(), "Q and K devices must match");
  TORCH_CHECK(V.device() == K.device(), "V and K devices must match");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

  TORCH_CHECK(start_pos >= 0, "start_pos must be non-negative");
  TORCH_CHECK(
      start_pos + Q.size(0) == K.size(0),
      "start_pos + q_len must equal kv_len");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "scale must be finite and positive");
}

at::Tensor contiguous_attention_cpu(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale) {
  check_contiguous_attention_inputs(Q, K, V, start_pos, scale);
  TORCH_CHECK(!Q.is_cuda(), "CPU kernel received CUDA tensors");
  TORCH_CHECK(
      false,
      "einf::contiguous_attention CPU kernel is not implemented; Gate 5 focuses on CUDA");
  return {};
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def(
      "contiguous_attention(Tensor Q, Tensor K, Tensor V, "
      "int start_pos, float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl(
      "contiguous_attention",
      TORCH_FN(einf::ops::contiguous_attention_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl(
      "contiguous_attention",
      TORCH_FN(einf::ops::contiguous_attention_cuda));
}
