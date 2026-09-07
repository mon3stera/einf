#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

void check_marlin_gemm_inputs(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& s,
    const at::Tensor& C);

void marlin_gemm_cpu(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& s,
    at::Tensor& C,
    at::Tensor& workspace,
    int64_t group_size,
    int64_t max_par);

void marlin_gemm_cuda(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& s,
    at::Tensor& C,
    at::Tensor& workspace,
    int64_t group_size,
    int64_t max_par);

}  // namespace einf::ops
