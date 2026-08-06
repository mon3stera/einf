#include "gather_context.h"

#include <torch/library.h>

namespace einf::ops {

void check_gather_context_inputs(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len) {
  TORCH_CHECK(K_cache.dim() == 4, "K_cache must have shape [num_blocks, block_len, num_kv_heads, head_dim]");
  TORCH_CHECK(V_cache.sizes() == K_cache.sizes(), "V_cache must match K_cache shape");
  TORCH_CHECK(V_cache.scalar_type() == K_cache.scalar_type(), "V_cache dtype must match K_cache");
  TORCH_CHECK(V_cache.device() == K_cache.device(), "V_cache device must match K_cache");
  TORCH_CHECK(K_cache.is_contiguous(), "K_cache must be contiguous");
  TORCH_CHECK(V_cache.is_contiguous(), "V_cache must be contiguous");
  TORCH_CHECK(K_cache.size(1) > 0, "block_len must be positive");

  TORCH_CHECK(block_table.dim() == 1, "block_table must have shape [num_logical_blocks]");
  TORCH_CHECK(block_table.scalar_type() == at::kLong, "block_table must have dtype torch.int64");
  TORCH_CHECK(block_table.device() == K_cache.device(), "block_table device must match cache");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");

  TORCH_CHECK(context_len >= 0, "context_len must be non-negative");
  const int64_t block_len = K_cache.size(1);
  const int64_t required_blocks =
      context_len == 0 ? 0 : (context_len + block_len - 1) / block_len;
  TORCH_CHECK(required_blocks <= block_table.size(0), "block_table is too short for context_len");
}

std::tuple<at::Tensor, at::Tensor> gather_context_cpu(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len) {
  check_gather_context_inputs(K_cache, V_cache, block_table, context_len);
  TORCH_CHECK(!K_cache.is_cuda(), "CPU kernel received a CUDA cache tensor");
  TORCH_CHECK(
      false,
      "einf::gather_context CPU kernel is not implemented; Gate 4 focuses on CUDA");
  return {};
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def(
      "gather_context(Tensor K_cache, Tensor V_cache, Tensor block_table, "
      "int context_len) -> (Tensor K, Tensor V)");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("gather_context", TORCH_FN(einf::ops::gather_context_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("gather_context", TORCH_FN(einf::ops::gather_context_cuda));
}
