#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

# Generic workload client wrapper.
#
# Usage:
#   CONFIG=A_tp8 BASE_URL=http://localhost:8000 bash qiannan-test/scripts/run_client.sh T1
#   CONFIG=B_dpa_tp8_dp8_router BASE_URL=http://localhost:30000 bash qiannan-test/scripts/run_client.sh T6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCH_DIR="${TEST_ROOT}/bench"
RUNS_DIR="${RUNS_DIR:-${TEST_ROOT}/runs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FIXTURE_ARG="${1:-${FIXTURE:-T1}}"
CONFIG="${CONFIG:-manual}"
BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-default}"
TIMEOUT="${TIMEOUT:-1800}"
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_DURATION="${MAX_DURATION:-}"
SUMMARY_INTERVAL="${SUMMARY_INTERVAL:-}"

case "${FIXTURE_ARG}" in
  T1|t1)
    FIXTURE_NAME="T1_sysprompt_pure"
    DATA_PATH="${TEST_ROOT}/fixtures/T1_sysprompt_pure.jsonl"
    ;;
  T6|t6)
    FIXTURE_NAME="T6_official_50"
    DATA_PATH="${TEST_ROOT}/fixtures/T6_official_50.jsonl"
    ;;
  *.jsonl)
    FIXTURE_NAME="$(basename "${FIXTURE_ARG}" .jsonl)"
    DATA_PATH="${FIXTURE_ARG}"
    ;;
  *)
    echo "Unknown fixture '${FIXTURE_ARG}'. Use T1, T6, or a .jsonl path." >&2
    exit 2
    ;;
esac

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Fixture not found: ${DATA_PATH}" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${TS}_${CONFIG}_${FIXTURE_NAME}}"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/config.yaml" <<EOF
run_id: ${RUN_ID}
config: ${CONFIG}
fixture: ${FIXTURE_NAME}
data: ${DATA_PATH}
base_url: ${BASE_URL}
model: ${MODEL}
timeout: ${TIMEOUT}
temperature: ${TEMPERATURE}
max_duration: ${MAX_DURATION:-null}
summary_interval: ${SUMMARY_INTERVAL:-null}
EOF

echo "[client] run_id: ${RUN_ID}"
echo "[client] config: ${CONFIG}"
echo "[client] fixture: ${FIXTURE_NAME}"
echo "[client] target: ${BASE_URL}"
echo "[client] output: ${RUN_DIR}"

CLIENT_ARGS=(
  "${BENCH_DIR}/run_workload.py"
  --data "${DATA_PATH}"
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --timeout "${TIMEOUT}"
  --temperature "${TEMPERATURE}"
  --output "${RUN_DIR}/client_metrics.jsonl"
)

if [[ -n "${MAX_SESSIONS:-}" ]]; then
  CLIENT_ARGS+=(--max-sessions "${MAX_SESSIONS}")
fi

if [[ -n "${MAX_DURATION}" ]]; then
  CLIENT_ARGS+=(--max-duration "${MAX_DURATION}")
fi

if [[ -n "${SUMMARY_INTERVAL}" ]]; then
  CLIENT_ARGS+=(
    --summary-interval "${SUMMARY_INTERVAL}"
    --summary-output "${RUN_DIR}/interval_summaries.jsonl"
  )
fi

if [[ "${NO_IGNORE_EOS:-0}" == "1" ]]; then
  CLIENT_ARGS+=(--no-ignore-eos)
fi

"${PYTHON_BIN}" "${CLIENT_ARGS[@]}" 2>&1 | tee "${RUN_DIR}/client_stdout.log"

"${PYTHON_BIN}" - "${RUN_DIR}/client_metrics.jsonl" "${RUN_DIR}/summary.json" "${RUN_ID}" "${CONFIG}" "${FIXTURE_NAME}" "${BASE_URL}" <<'PY'
import json
import sys

metrics_path, summary_path, run_id, config, fixture, base_url = sys.argv[1:]
summary = None
with open(metrics_path) as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "summary":
            summary = obj

if summary is None:
    raise SystemExit(f"no summary found in {metrics_path}")

summary.update({
    "run_id": run_id,
    "config": config,
    "fixture": fixture,
    "base_url": base_url,
})

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY

echo "[client] summary: ${RUN_DIR}/summary.json"
