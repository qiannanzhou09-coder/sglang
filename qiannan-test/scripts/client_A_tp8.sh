#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-A_tp8}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
export CONFIG BASE_URL

exec bash "${SCRIPT_DIR}/run_client.sh" "${1:-${FIXTURE:-T1}}"
