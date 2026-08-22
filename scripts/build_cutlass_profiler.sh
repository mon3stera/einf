#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(dirname -- "$script_dir")"
cutlass_source="${CUTLASS_SOURCE_DIR:-$repo_root/third_party/cutlass}"
build_dir="${CUTLASS_BUILD_DIR:-/tmp/einf-cutlass-build}"

if [[ ! -f "$cutlass_source/include/cutlass/cutlass.h" ]]; then
  printf 'CUTLASS source is missing at %s\n' "$cutlass_source" >&2
  printf 'Run: git submodule update --init third_party/cutlass\n' >&2
  exit 1
fi

cmake \
  -S "$cutlass_source" \
  -B "$build_dir" \
  -GNinja \
  -DCUTLASS_NVCC_ARCHS=89 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_LIBRARY_KERNELS='cutlass_simt_sgemm_*_nn_align1,cutlass_tensorop_s1688gemm_tf32_*_nn_align*' \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

cmake \
  --build "$build_dir" \
  --target cutlass_profiler \
  -j "${BUILD_JOBS:-$(nproc)}"

printf '%s\n' "$build_dir/tools/profiler/cutlass_profiler"
