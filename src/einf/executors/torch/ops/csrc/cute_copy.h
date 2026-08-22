#pragma once

#include <torch/extension.h>

namespace einf::ops {

at::Tensor cute_copy(const at::Tensor& input);
at::Tensor cute_copy_cuda(const at::Tensor& input);

}  // namespace einf::ops
