#include "kv_cache.h"

#include <torch/library.h>

namespace einf::ops {

void check_write_slots_inputs(
    const at::Tensor& K_cache,
    const at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V) {
  TORCH_CHECK(K_cache.dim() == 4, "K_cache must have shape [num_blocks, block_len, num_kv_heads, head_dim]");
  TORCH_CHECK(V_cache.sizes() == K_cache.sizes(), "V_cache must match K_cache shape");
  TORCH_CHECK(K.dim() == 3, "K must have shape [num_tokens, num_kv_heads, head_dim]");
  TORCH_CHECK(V.sizes() == K.sizes(), "V must match K shape");
  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must have shape [num_tokens]");
  TORCH_CHECK(slot_mapping.size(0) == K.size(0), "slot_mapping length must match num_tokens");
  TORCH_CHECK(K.size(1) == K_cache.size(2), "K num_kv_heads must match K_cache");
  TORCH_CHECK(K.size(2) == K_cache.size(3), "K head_dim must match K_cache");
  TORCH_CHECK(V_cache.scalar_type() == K_cache.scalar_type(), "V_cache dtype must match K_cache");
  TORCH_CHECK(V.scalar_type() == K.scalar_type(), "V dtype must match K");
  // An FP8 cache quantizes BF16/FP16/FP32 K/V on write; a floating-point
  // cache still requires the inputs to share its dtype.
  const bool cache_fp8 = K_cache.scalar_type() == at::ScalarType::Float8_e4m3fn;
  if (cache_fp8) {
    const auto input = K.scalar_type();
    TORCH_CHECK(
        input == at::ScalarType::Half || input == at::ScalarType::BFloat16 ||
            input == at::ScalarType::Float,
        "FP8 KV cache requires Half, BFloat16 or Float K/V inputs");
  } else {
    TORCH_CHECK(K.scalar_type() == K_cache.scalar_type(), "K dtype must match K_cache");
  }
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong, "slot_mapping must have dtype torch.int64");
  TORCH_CHECK(K_cache.is_contiguous(), "K_cache must be contiguous");
  TORCH_CHECK(V_cache.is_contiguous(), "V_cache must be contiguous");
  TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
  TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(K_cache.device() == V_cache.device(), "K/V cache devices must match");
  TORCH_CHECK(K_cache.device() == slot_mapping.device(), "slot_mapping device must match cache");
  TORCH_CHECK(K_cache.device() == K.device(), "K device must match cache");
  TORCH_CHECK(K_cache.device() == V.device(), "V device must match cache");
}

void write_slots_cpu(
    at::Tensor& K_cache,
    at::Tensor& V_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& K,
    const at::Tensor& V,
    double k_scale,
    double v_scale) {
  check_write_slots_inputs(K_cache, V_cache, slot_mapping, K, V);
  TORCH_CHECK(!K_cache.is_cuda(), "CPU kernel received a CUDA cache tensor");
  (void)k_scale;
  (void)v_scale;
  TORCH_CHECK(
      false,
      "einf::write_slots_ CPU kernel is not implemented; implement it in kv_cache.cpp");
}

}  // namespace einf::ops

TORCH_LIBRARY(einf, m) {
  m.def(
      "write_slots_(Tensor(a!) K_cache, Tensor(b!) V_cache, "
      "Tensor slot_mapping, Tensor K, Tensor V, *, "
      "float k_scale=1.0, float v_scale=1.0) -> ()");
}

TORCH_LIBRARY_IMPL(einf, CPU, m) {
  m.impl("write_slots_", TORCH_FN(einf::ops::write_slots_cpu));
}

TORCH_LIBRARY_IMPL(einf, CUDA, m) {
  m.impl("write_slots_", TORCH_FN(einf::ops::write_slots_cuda));
}
