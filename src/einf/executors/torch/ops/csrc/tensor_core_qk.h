#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

at::Tensor tensor_core_qk_cpu(
    const at::Tensor& Q,
    const at::Tensor& K);

at::Tensor tensor_core_qk_cuda(
    const at::Tensor& Q,
    const at::Tensor& K);

}  // namespace einf::ops
