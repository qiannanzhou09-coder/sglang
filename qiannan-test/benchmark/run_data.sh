#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs-data"
OUTPUT_ROOT="${SCRIPT_DIR}/../figure-runs"

if [[ "${1:-}" == "list" ]]; then
  exec python3 "${SCRIPT_DIR}/run_benchmark.py" list \
    --config-dir "${CONFIG_DIR}" \
    --output-root "${OUTPUT_ROOT}"
fi

if [[ "${1:-}" == "all" ]]; then
  shift
  exec python3 "${SCRIPT_DIR}/run_benchmark.py" all \
    --config-dir "${CONFIG_DIR}" \
    --output-root "${OUTPUT_ROOT}" \
    "$@"
fi

exec python3 "${SCRIPT_DIR}/run_benchmark.py" run \
  --config-dir "${CONFIG_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"
