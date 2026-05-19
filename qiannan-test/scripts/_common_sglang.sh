#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

# Shared defaults for Qwen3.5-122B SGLang serving experiments.
# Override any value from the shell, e.g. MODEL_PATH=/path/to/model bash serve_A_tp8.sh

export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"

MODEL_PATH="${MODEL_PATH:-/home/Qwen3.5-122B-A10B}"
HOST="${HOST:-0.0.0.0}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
SPECULATIVE_ALGO="${SPECULATIVE_ALGO-NEXTN}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"

COMMON_SGLANG_ARGS=(
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${SGLANG_PORT}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --context-length "${CONTEXT_LENGTH}"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --mamba-scheduler-strategy "${MAMBA_SCHEDULER_STRATEGY}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --enable-mixed-chunk
  --schedule-policy "${SCHEDULE_POLICY}"
  --enable-cache-report
  --enable-metrics
)

case "${SPECULATIVE_ALGO}" in
  ""|NONE|none|None|OFF|off|Off|FALSE|false|False|0|null|NULL|Null)
    ;;
  *)
    COMMON_SGLANG_ARGS+=(
      --speculative-algo "${SPECULATIVE_ALGO}"
      --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
      --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
      --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    )
    ;;
esac

if [[ -n "${SCHEDULE_CONSERVATIVENESS:-}" ]]; then
  COMMON_SGLANG_ARGS+=(--schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}")
fi
