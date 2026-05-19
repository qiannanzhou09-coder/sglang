#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DP_SIZE="${DP_SIZE:-2}"
source "${SCRIPT_DIR}/_common_router.sh"

echo "[E router] Starting dp-aware cache_aware router on :${ROUTER_PORT}"
printf "    worker: %s\n" "${WORKER_URLS[@]}"

python -m sglang_router.launch_router \
  --worker-urls "${WORKER_URLS[@]}" \
  --policy "${ROUTER_POLICY}" \
  --dp-aware \
  --host "${ROUTER_HOST}" \
  --port "${ROUTER_PORT}"
