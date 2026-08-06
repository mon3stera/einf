#include "paged_attention_split_kv.h"

#include "paged_attention.h"

#include <torch/library.h>

namespace einf::ops {

void check_paged_decode_attention_split_kv_inputs(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    int64_t num_splits,
    double scale) {
  check_paged_decode_attention_inputs(
      Q, K_cache, V_cache, block_table, context_len, scale);

  const int64_t block_len = K_cache.size(1);
  const int64_t num_logical_blocks =
      (context_len + block_len - 1) / block_len;
  TORCH_CHECK(num_splits > 0, "num_splits must be positive");
  TORCH_CHECK(
      num_splits <= num_logical_blocks,
      "num_splits must not exceed the number of logical context blocks");
}

at::Tensor paged_decode_attention_split_kv_cpu(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    int64_t num_splits,
    double scale) {
  check_paged_decode_attention_split_kv_inputs(
      Q,
      K_cache,
      V_cache,
      block_table,
      context_len,
      num_splits,
      scale);
  TORCH_CHECK(!Q.is_cuda(), "CPU kernel received CUDA tensors");
  TORCH_CHECK(
      false,
      "einf::paged_decode_attention_split_kv CPU kernel is not implemented; "
      "Gate 5 focuses on CUDA");
  return {};
}

}  // namespace einf::ops

TORCH_LIBRARY_FRAGMENT(einf, m) {
  m.def(
      "paged_decode_attention_split_kv(Tensor Q, Tensor K_cache, "
      "Tensor V_cache, Tensor block_table, int context_len, int num_splits, "
      "float scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl(
      "paged_decode_attention_split_kv",
      TORCH_FN(einf::ops::paged_decode_attention_split_kv_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl(
      "paged_decode_attention_split_kv",
      TORCH_FN(einf::ops::paged_decode_attention_split_kv_cuda));
}
