#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compensate SGLang DPA's internal adjustments:
#   chunked_prefill_size /= dp_size  -> 65536 / 8 = 8192 effective
#   schedule_conservativeness *= 0.3 -> 3.33 * 0.3 ~= 1.0 effective
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-65536}"
SCHEDULE_CONSERVATIVENESS="${SCHEDULE_CONSERVATIVENESS:-3.33}"
export CHUNKED_PREFILL_SIZE SCHEDULE_CONSERVATIVENESS
EFFECTIVE_CHUNKED_PREFILL_SIZE="$((CHUNKED_PREFILL_SIZE / 8))"

source "${SCRIPT_DIR}/_common_sglang.sh"

echo "[F] Starting DPA server TP=8 DP=8 on :${SGLANG_PORT}"
echo "    Attention: dp=8, tp=1; MoE/dense TP remains 8."
echo "    Start router in another shell: DP_SIZE=8 bash ${SCRIPT_DIR}/router_F_dp4_cache_aware.sh"
echo "    Client target for F: http://<host>:${ROUTER_PORT:-30000}"
echo "    DPA effective chunked-prefill-size: ${CHUNKED_PREFILL_SIZE} / 8 = ${EFFECTIVE_CHUNKED_PREFILL_SIZE}"

python -m sglang.launch_server \
  "${COMMON_SGLANG_ARGS[@]}" \
  --tp-size 8 \
  --dp-size 8 \
  --enable-dp-attention
