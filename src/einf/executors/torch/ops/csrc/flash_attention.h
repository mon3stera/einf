#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

at::Tensor flash_attention_cpu(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale);

at::Tensor flash_attention_cuda(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    int64_t start_pos,
    double scale);

}  // namespace einf::ops
