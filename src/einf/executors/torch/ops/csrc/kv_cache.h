#pragma once

#include <ATen/ATen.h>

namespace einf::ops {

void check_write_slots_inputs(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V);

void write_slots_cpu(
    at::Tensor& K_cache,
    at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V,
    double k_scale,
    double v_scale);

void write_slots_cuda(
    at::Tensor& K_cache,
    at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V,
    double k_scale,
    double v_scale);

}  // namespace einf::ops
