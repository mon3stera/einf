#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

at::Tensor cute_elementwise_add_cpu(
    const at::Tensor& X,
    const at::Tensor& Y);

at::Tensor cute_elementwise_add_cuda(
    const at::Tensor& X,
    const at::Tensor& Y);

}  // namespace einf::ops
