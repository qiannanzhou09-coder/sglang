#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compensate SGLang DPA's internal adjustments:
#   chunked_prefill_size /= dp_size  -> 16384 / 2 = 8192 effective
#   schedule_conservativeness *= 0.3 -> 3.33 * 0.3 ~= 1.0 effective
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-3.33}"
export CHUNKED_PREFILL_SIZE SCHEDULE_CONSERVATIVENESS

source "${SCRIPT_DIR}/_common_sglang.sh"

echo "[E] Starting DPA server TP=8 DP=2 on :${SGLANG_PORT}"
echo "    Attention: dp=2, tp=4; MoE/dense TP remains 8."
echo "    Start router in another shell: bash ${SCRIPT_DIR}/router_E_dp2_cache_aware.sh"
echo "    Client target for E: http://<host>:${ROUTER_PORT:-30000}"
echo "    DPA effective chunked-prefill-size: ${CHUNKED_PREFILL_SIZE} / 2 = 8192"

python -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --tp-size 8 \
  --dp-size 2 \
  --enable-dp-attention
