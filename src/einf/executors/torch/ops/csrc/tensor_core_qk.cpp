#include "tensor_core_qk.h"

#include <torch/library.h>

namespace einf::ops {

namespace {

void check_tensor_core_qk_inputs(
    const at::Tensor& Q,
    const at::Tensor& K) {
  TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA Tensor");
  TORCH_CHECK(K.is_cuda(), "K must be a CUDA Tensor");
  TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(Q.device() == K.device(), "Q and K must be on the same CUDA device");
  TORCH_CHECK(Q.dim() == 3, "Q must have shape [q_len,Hq,D]");
  TORCH_CHECK(K.dim() == 3, "K must have shape [kv_len,Hkv,D]");
  TORCH_CHECK(
      Q.scalar_type() == at::kBFloat16 && K.scalar_type() == at::kBFloat16,
      "tensor_core_qk requires BF16 Q and K");
  TORCH_CHECK(Q.size(0) > 0, "q_len must be positive");
  TORCH_CHECK(K.size(0) > 0, "kv_len must be positive");
  TORCH_CHECK(Q.size(1) > 0, "num_attention_heads must be positive");
  TORCH_CHECK(K.size(1) > 0, "num_kv_heads must be positive");
  TORCH_CHECK(Q.size(2) == 64, "tensor_core_qk v0 requires head_dim == 64");
  TORCH_CHECK(K.size(2) == Q.size(2), "Q and K head dimensions must match");
  TORCH_CHECK(
      Q.size(1) % K.size(1) == 0,
      "num_attention_heads must be divisible by num_kv_heads");
  TORCH_CHECK(
      Q.size(0) % 16 == 0,
      "tensor_core_qk v0 requires q_len to be a multiple of 16");
  TORCH_CHECK(
      K.size(0) % 16 == 0,
      "tensor_core_qk v0 requires kv_len to be a multiple of 16");
}

}  // namespace

at::Tensor tensor_core_qk_cpu(
    const at::Tensor& Q,
    const at::Tensor& K) {
  TORCH_CHECK(false, "einf::tensor_core_qk is CUDA-only");
  return {};
}

at::Tensor tensor_core_qk(
    const at::Tensor& Q,
    const at::Tensor& K) {
  check_tensor_core_qk_inputs(Q, K);
  return tensor_core_qk_cuda(Q, K);
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def("tensor_core_qk(Tensor Q, Tensor K) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("tensor_core_qk", TORCH_FN(einf::ops::tensor_core_qk_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("tensor_core_qk", TORCH_FN(einf::ops::tensor_core_qk));
}
