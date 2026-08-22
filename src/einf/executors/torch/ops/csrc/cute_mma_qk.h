#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

at::Tensor cute_mma_qk_cpu(
    const at::Tensor& Q,
    const at::Tensor& K);

at::Tensor cute_mma_qk_cuda(
    const at::Tensor& Q,
    const at::Tensor& K);

}  // namespace einf::ops
