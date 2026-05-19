#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_sglang.sh"

echo "[A] Starting pure TP baseline on :${SGLANG_PORT}"
echo "    Client target: http://<host>:${SGLANG_PORT}"

python -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --tp-size 8
