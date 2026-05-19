#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_sglang.sh"

echo "[B] Starting DPA server TP=8 DP=8 on :${SGLANG_PORT}"
echo "    Start router in another shell: bash ${SCRIPT_DIR}/router_B_dp8_cache_aware.sh"
echo "    Client target for B: http://<host>:${ROUTER_PORT:-30000}"
echo "    Note: SGLang DPA internally divides chunked-prefill-size by dp-size."

python -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention
