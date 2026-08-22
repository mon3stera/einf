#pragma once

#include <torch/extension.h>

namespace einf::ops {

torch::Tensor paged_decode_attention_batched_cuda(
    const torch::Tensor& query,
    const torch::Tensor& K_cache,
    const torch::Tensor& V_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& single_query_request_indices,
    double scale);

}  // namespace einf::ops
