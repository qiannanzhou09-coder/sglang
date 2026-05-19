#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-A_tp8_full_workload}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
DATA_PATH="${DATA_PATH:-${TEST_ROOT}/bench/workload_data.jsonl}"
export CONFIG BASE_URL

exec bash "${SCRIPT_DIR}/run_client.sh" "${DATA_PATH}"
