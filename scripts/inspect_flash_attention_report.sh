#!/usr/bin/env bash
set -euo pipefail

REPORT="${1:-/tmp/einf-flash-chunk-full.ncu-rep}"

if [[ ! -f "${REPORT}" ]]; then
  echo "Report not found: ${REPORT}" >&2
  exit 1
fi

exec /usr/bin/ncu \
  --import "${REPORT}" \
  --page details
