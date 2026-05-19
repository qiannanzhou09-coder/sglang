#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_sglang.sh"

echo "[C] Starting DPA no-router control TP=8 DP=8 on :${SGLANG_PORT}"
echo "    Client target: http://<host>:${SGLANG_PORT}"
echo "    This intentionally does not use cache-aware routing."
echo "    Note: SGLang DPA internally divides chunked-prefill-size by dp-size."

python -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention
