#pragma once

#include <torch/extension.h>

namespace einf::ops {

at::Tensor cute_transpose(const at::Tensor& input);
at::Tensor cute_transpose_cuda(const at::Tensor& input);

}  // namespace einf::ops
