#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

void check_contiguous_attention_inputs(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale);

at::Tensor contiguous_attention_cpu(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale);

at::Tensor contiguous_attention_cuda(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale);

}  // namespace einf::ops
