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
MAX_ACTIVE_SESSIONS="${MAX_ACTIVE_SESSIONS:-}"
SESSION_IDLE_TIMEOUT="${SESSION_IDLE_TIMEOUT:-}"
SESSION_CLEANUP_INTERVAL="${SESSION_CLEANUP_INTERVAL:-}"
DYNAMIC_ADMISSION="${DYNAMIC_ADMISSION:-}"
DYNAMIC_MIN_INFLIGHT_REQUESTS="${DYNAMIC_MIN_INFLIGHT_REQUESTS:-}"
DYNAMIC_INITIAL_INFLIGHT_REQUESTS="${DYNAMIC_INITIAL_INFLIGHT_REQUESTS:-}"
DYNAMIC_MAX_INFLIGHT_REQUESTS="${DYNAMIC_MAX_INFLIGHT_REQUESTS:-}"
DYNAMIC_CONTROL_INTERVAL="${DYNAMIC_CONTROL_INTERVAL:-}"
DYNAMIC_MIN_SAMPLES="${DYNAMIC_MIN_SAMPLES:-}"
DYNAMIC_LOW_CACHE_HIT="${DYNAMIC_LOW_CACHE_HIT:-}"
DYNAMIC_HIGH_CACHE_HIT="${DYNAMIC_HIGH_CACHE_HIT:-}"
DYNAMIC_HIGH_TTFT="${DYNAMIC_HIGH_TTFT:-}"
DYNAMIC_HIGH_QUEUE_WAIT="${DYNAMIC_HIGH_QUEUE_WAIT:-}"
DYNAMIC_ADDITIVE_STEP="${DYNAMIC_ADDITIVE_STEP:-}"
DYNAMIC_DECREASE_FACTOR="${DYNAMIC_DECREASE_FACTOR:-}"

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

if [[ -n "${MAX_ACTIVE_SESSIONS}" ]]; then
  PROXY_ARGS+=(--max-active-sessions "${MAX_ACTIVE_SESSIONS}")
fi

if [[ -n "${SESSION_IDLE_TIMEOUT}" ]]; then
  PROXY_ARGS+=(--session-idle-timeout "${SESSION_IDLE_TIMEOUT}")
fi

if [[ -n "${SESSION_CLEANUP_INTERVAL}" ]]; then
  PROXY_ARGS+=(--session-cleanup-interval "${SESSION_CLEANUP_INTERVAL}")
fi

case "${DYNAMIC_ADMISSION}" in
  1|true|TRUE|True|on|ON|On|yes|YES|Yes)
    PROXY_ARGS+=(--dynamic-admission)
    ;;
esac

if [[ -n "${DYNAMIC_MIN_INFLIGHT_REQUESTS}" ]]; then
  PROXY_ARGS+=(--dynamic-min-inflight-requests "${DYNAMIC_MIN_INFLIGHT_REQUESTS}")
fi

if [[ -n "${DYNAMIC_INITIAL_INFLIGHT_REQUESTS}" ]]; then
  PROXY_ARGS+=(--dynamic-initial-inflight-requests "${DYNAMIC_INITIAL_INFLIGHT_REQUESTS}")
fi

if [[ -n "${DYNAMIC_MAX_INFLIGHT_REQUESTS}" ]]; then
  PROXY_ARGS+=(--dynamic-max-inflight-requests "${DYNAMIC_MAX_INFLIGHT_REQUESTS}")
fi

if [[ -n "${DYNAMIC_CONTROL_INTERVAL}" ]]; then
  PROXY_ARGS+=(--dynamic-control-interval "${DYNAMIC_CONTROL_INTERVAL}")
fi

if [[ -n "${DYNAMIC_MIN_SAMPLES}" ]]; then
  PROXY_ARGS+=(--dynamic-min-samples "${DYNAMIC_MIN_SAMPLES}")
fi

if [[ -n "${DYNAMIC_LOW_CACHE_HIT}" ]]; then
  PROXY_ARGS+=(--dynamic-low-cache-hit "${DYNAMIC_LOW_CACHE_HIT}")
fi

if [[ -n "${DYNAMIC_HIGH_CACHE_HIT}" ]]; then
  PROXY_ARGS+=(--dynamic-high-cache-hit "${DYNAMIC_HIGH_CACHE_HIT}")
fi

if [[ -n "${DYNAMIC_HIGH_TTFT}" ]]; then
  PROXY_ARGS+=(--dynamic-high-ttft "${DYNAMIC_HIGH_TTFT}")
fi

if [[ -n "${DYNAMIC_HIGH_QUEUE_WAIT}" ]]; then
  PROXY_ARGS+=(--dynamic-high-queue-wait "${DYNAMIC_HIGH_QUEUE_WAIT}")
fi

if [[ -n "${DYNAMIC_ADDITIVE_STEP}" ]]; then
  PROXY_ARGS+=(--dynamic-additive-step "${DYNAMIC_ADDITIVE_STEP}")
fi

if [[ -n "${DYNAMIC_DECREASE_FACTOR}" ]]; then
  PROXY_ARGS+=(--dynamic-decrease-factor "${DYNAMIC_DECREASE_FACTOR}")
fi

python3 "${TEST_ROOT}/proxy/main.py" "${PROXY_ARGS[@]}"
