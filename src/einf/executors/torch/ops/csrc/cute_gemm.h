#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

at::Tensor cute_gemm_cpu(
    const at::Tensor& A,
    const at::Tensor& B);

at::Tensor cute_gemm_cuda(
    const at::Tensor& A,
    const at::Tensor& B);

}  // namespace einf::ops
