#!/usr/bin/env bash
set -euo pipefail

Q_LEN="${Q_LEN:-128}"
KV_LEN="${KV_LEN:-32768}"
WARMUP="${WARMUP:-10}"
REPORT="${REPORT:-/tmp/einf-flash-chunk-full}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/python/.venv311/bin/python}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/einf-ncu-torch-extensions}"

sudo env \
  TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR}" \
  /usr/bin/ncu \
  --set full \
  --kernel-name-base function \
  --kernel-name 'regex:.*flash_attention_forward_kernel.*' \
  --launch-skip "${WARMUP}" \
  --launch-count 1 \
  --export "${REPORT}" \
  --force-overwrite \
  "${PYTHON_BIN}" \
    "${REPO_ROOT}/benchmarks/profile_flash_attention.py" \
    --q-len "${Q_LEN}" \
    --kv-len "${KV_LEN}" \
    --warmup "${WARMUP}"

echo "Wrote ${REPORT}.ncu-rep"
