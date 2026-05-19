#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

# Run one dataset against the TP8 server.
#
# Usage:
#   bash qiannan-test/scripts/steps2/client_tp8.sh full
#   bash qiannan-test/scripts/steps2/client_tp8.sh T6
#   bash qiannan-test/scripts/steps2/client_tp8.sh qiannan-test/fixtures/T4_decode_heavy.jsonl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_SCRIPT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_ROOT="$(cd "${PARENT_SCRIPT_DIR}/.." && pwd)"

DATASET="${1:-${DATASET:-full}}"
CONFIG="${CONFIG:-${CONFIG_NAME:-tp8_steps2}}"
BASE_URL="${BASE_URL:-http://localhost:${SGLANG_PORT:-8000}}"
export CONFIG BASE_URL

case "${DATASET}" in
  full|FULL|workload|workload_data)
    DATA_PATH="${TEST_ROOT}/bench/workload_data.jsonl"
    ;;
  T1|t1)
    DATA_PATH="${TEST_ROOT}/fixtures/T1_sysprompt_pure.jsonl"
    ;;
  T2|t2)
    DATA_PATH="${TEST_ROOT}/fixtures/T2_unique_sysprompt.jsonl"
    ;;
  T3|t3)
    DATA_PATH="${TEST_ROOT}/fixtures/T3_burst_long_input.jsonl"
    ;;
  T4|t4)
    DATA_PATH="${TEST_ROOT}/fixtures/T4_decode_heavy.jsonl"
    ;;
  T5|t5)
    DATA_PATH="${TEST_ROOT}/fixtures/T5_long_wait.jsonl"
    ;;
  T6|t6)
    DATA_PATH="${TEST_ROOT}/fixtures/T6_official_50.jsonl"
    ;;
  *.jsonl)
    DATA_PATH="${DATASET}"
    ;;
  *)
    echo "Unknown dataset '${DATASET}'. Use full, T1-T6, or a .jsonl path." >&2
    exit 2
    ;;
esac

exec bash "${PARENT_SCRIPT_DIR}/run_client.sh" "${DATA_PATH}"
