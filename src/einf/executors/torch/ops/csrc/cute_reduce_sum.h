#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

at::Tensor cute_reduce_sum_cpu(const at::Tensor& input);

at::Tensor cute_reduce_sum_cuda(const at::Tensor& input);

}  // namespace einf::ops
