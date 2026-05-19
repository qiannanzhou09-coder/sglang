#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This wrapper serves an already-built pruned checkpoint. It intentionally does
# not set PRUNE_MODE, so startup will not rebuild or mutate the model.
PRUNED_MODEL_PATH="${PRUNED_MODEL_PATH:-/home/qwen3.5-pruned-models/Qwen3.5-122B-A10B-REAP-20-text-only}"

if [[ ! -f "${PRUNED_MODEL_PATH}/config.json" ]]; then
  echo "Pruned model not found or incomplete: ${PRUNED_MODEL_PATH}" >&2
  echo "Set PRUNED_MODEL_PATH=/home/qwen3.5-pruned-models/<model-dir> if your directory name is different." >&2
  exit 2
fi

export MODEL_PATH="${PRUNED_MODEL_PATH}"
export SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NONE}"
export CONFIG="${CONFIG:-E_dpa_tp8_dp2_router_pruned_spec_off}"

exec bash "${SCRIPT_DIR}/serve_E_dpa_tp8_dp2.sh"
