#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

void check_paged_decode_attention_inputs(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    double scale);

at::Tensor paged_decode_attention_cpu(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    double scale);

at::Tensor paged_decode_attention_cuda(
    const at::Tensor& Q,
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len,
    double scale);

}  // namespace einf::ops
