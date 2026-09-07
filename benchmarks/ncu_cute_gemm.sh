#!/bin/bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.9
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_zzx
export CUDA_VISIBLE_DEVICES=0
cd /media/zzx/新加卷2/aiinfra/projects/einf

python3 - <<'PY'
import torch
from torch.utils.cpp_extension import load
from pathlib import Path
root = Path('/media/zzx/新加卷2/aiinfra/projects/einf')
csrc = root / 'src/einf/executors/torch/ops/csrc'
load(
    name='einf_gemm_only',
    sources=[str(csrc / 'cute_gemm.cpp'), str(csrc / 'cute_gemm_cuda.cu')],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-lineinfo'],
    extra_include_paths=[str(root / 'third_party/cutlass/include')],
    with_cuda=True,
    is_python_module=False,
)
print('loaded', torch.ops.einf.cute_gemm)
PY

cat > /tmp/ncu_cute_gemm_launch.py <<'PY'
import torch
torch.manual_seed(0)
A = torch.randn((4096, 4096), device='cuda', dtype=torch.float32)
B = torch.randn((4096, 4096), device='cuda', dtype=torch.float32)
for _ in range(5):
    C = torch.ops.einf.cute_gemm(A, B)
torch.cuda.synchronize()
C = torch.ops.einf.cute_gemm(A, B)
torch.cuda.synchronize()
print(tuple(C.shape))
PY

echo zzx040506 | sudo -S ncu --target-processes all \
  --kernel-name-base demangled \
  --kernel-name regex:cute_gemm \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_issue_stalled_no_eligible.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__occupancy_per_block_size \
  --csv \
  /home/zzx/anaconda3/bin/python /tmp/ncu_cute_gemm_launch.py
