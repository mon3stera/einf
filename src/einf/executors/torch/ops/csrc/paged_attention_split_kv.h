#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

void check_paged_decode_attention_split_kv_inputs(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    int64_t num_splits,
    double scale);

at::Tensor paged_decode_attention_split_kv_cpu(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    int64_t num_splits,
    double scale);

at::Tensor paged_decode_attention_split_kv_cuda(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    int64_t num_splits,
    double scale);

}  // namespace einf::ops
