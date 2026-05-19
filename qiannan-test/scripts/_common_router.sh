#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

# Shared defaults for routing one DPA SGLang server through dp-aware cache_aware SMG.

ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
WORKER_URL_BASE="${WORKER_URL_BASE:-http://127.0.0.1:${SGLANG_PORT}}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
DP_SIZE="${DP_SIZE:?DP_SIZE must be set before sourcing _common_router.sh}"

WORKER_URLS=()
for ((rank = 0; rank < DP_SIZE; rank++)); do
  WORKER_URLS+=("${WORKER_URL_BASE}@${rank}")
done
