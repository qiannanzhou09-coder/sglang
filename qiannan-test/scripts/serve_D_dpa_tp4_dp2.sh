#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common_sglang.sh"

echo "[D] Starting medium DPA server TP=4 DP=2 on :${SGLANG_PORT}"
echo "    Start router in another shell: bash ${SCRIPT_DIR}/router_D_dp2_cache_aware.sh"
echo "    Client target for D: http://<host>:${ROUTER_PORT:-30000}"
echo "    Note: SGLang DPA internally divides chunked-prefill-size by dp-size."

python -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --tp-size 4 \
  --dp-size 2 \
  --enable-dp-attention

