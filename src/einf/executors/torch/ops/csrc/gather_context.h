#pragma once

#include <ATen/ATen.h>

#include <tuple>

namespace einf::ops {

void check_gather_context_inputs(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len);

std::tuple<at::Tensor, at::Tensor> gather_context_cpu(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len);

std::tuple<at::Tensor, at::Tensor> gather_context_cuda(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& block_table,
    int64_t context_len);

}  // namespace einf::ops
