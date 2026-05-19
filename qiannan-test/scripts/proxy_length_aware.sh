#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
PROXY_PORT="${PROXY_PORT:-8888}"
DIRECT_URL="${DIRECT_URL:-http://127.0.0.1:8000}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:30000}"
PROXY_POLICY="${PROXY_POLICY:-length_aware}"
LENGTH_THRESHOLD="${LENGTH_THRESHOLD:-2048}"
CHARS_PER_TOKEN="${CHARS_PER_TOKEN:-2.0}"
PROXY_METRICS_PATH="${PROXY_METRICS_PATH:-${TEST_ROOT}/runs/proxy_length_aware_metrics.jsonl}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-}"

PROXY_ARGS=(
  --host "${PROXY_HOST}"
  --port "${PROXY_PORT}"
  --direct-url "${DIRECT_URL}"
  --router-url "${ROUTER_URL}"
  --policy "${PROXY_POLICY}"
  --length-threshold "${LENGTH_THRESHOLD}"
  --chars-per-token "${CHARS_PER_TOKEN}"
  --metrics-path "${PROXY_METRICS_PATH}"
)

if [[ -n "${MAX_INFLIGHT_REQUESTS}" ]]; then
  PROXY_ARGS+=(--max-inflight-requests "${MAX_INFLIGHT_REQUESTS}")
fi

python3 "${TEST_ROOT}/proxy/main.py" "${PROXY_ARGS[@]}"
