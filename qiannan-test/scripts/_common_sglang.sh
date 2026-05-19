#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

# Shared defaults for Qwen3.5-122B SGLang serving experiments.
# Override any value from the shell, e.g. MODEL_PATH=/path/to/model bash serve_A_tp8.sh

COMMON_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/Qwen3.5-122B-A10B}"
MODEL_PATH="${MODEL_PATH:-${BASE_MODEL_PATH}}"
PRUNE_MODE="${PRUNE_MODE:-none}"
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
MIXED_CHUNK="${MIXED_CHUNK:-1}"

case "${PRUNE_MODE}" in
  ""|none|NONE|None|off|OFF|Off|0|false|FALSE|False)
    PRUNE_MODE="none"
    ;;
  *)
    MODEL_PATH="$(
      PRUNE_MODE="${PRUNE_MODE}" \
      BASE_MODEL_PATH="${BASE_MODEL_PATH}" \
      PRUNE_SOURCE_MODEL_PATH="${PRUNE_SOURCE_MODEL_PATH:-}" \
      PRUNE_PLAN_PATH="${PRUNE_PLAN_PATH:-}" \
      PRUNED_MODEL_ROOT="${PRUNED_MODEL_ROOT:-}" \
      PRUNED_MODEL_PATH="${PRUNED_MODEL_PATH:-}" \
      PRUNE_WORKERS="${PRUNE_WORKERS:-}" \
      PRUNE_MTP_POLICY="${PRUNE_MTP_POLICY:-}" \
      FORCE_REBUILD_PRUNED_MODEL="${FORCE_REBUILD_PRUNED_MODEL:-}" \
      PYTHON_BIN="${PYTHON_BIN:-}" \
      PRUNE_TOOL_PATH="${PRUNE_TOOL_PATH:-}" \
      "${COMMON_SCRIPT_DIR}/prepare_pruned_model.sh"
    )"
    ;;
esac

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
  --schedule-policy "${SCHEDULE_POLICY}"
  --enable-cache-report
  --enable-metrics
)

case "${MIXED_CHUNK}" in
  ""|OFF|off|Off|FALSE|false|False|0|null|NULL|Null)
    ;;
  *)
    COMMON_SGLANG_ARGS+=(--enable-mixed-chunk)
    ;;
esac

case "${SPECULATIVE_ALGO}" in
  ""|NONE|none|None|OFF|off|Off|FALSE|false|False|0|null|NULL|Null)
    ;;
  *)
    COMMON_SGLANG_ARGS+=(
      --speculative-algorithm "${SPECULATIVE_ALGO}"
      --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
      --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
      --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
    )
    ;;
esac

if [[ -n "${SCHEDULE_CONSERVATIVENESS:-}" ]]; then
  COMMON_SGLANG_ARGS+=(--schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}")
fi

if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
  COMMON_SGLANG_ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi

if [[ -n "${MAX_PREFILL_TOKENS:-}" ]]; then
  COMMON_SGLANG_ARGS+=(--max-prefill-tokens "${MAX_PREFILL_TOKENS}")
fi

if [[ -n "${KV_CACHE_DTYPE:-}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  COMMON_SGLANG_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

case "${ENABLE_DP_LM_HEAD:-0}" in
  1|true|TRUE|True|on|ON|On|yes|YES|Yes)
    COMMON_SGLANG_ARGS+=(--enable-dp-lm-head)
    ;;
esac

if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  COMMON_SGLANG_ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi

if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
  COMMON_SGLANG_ARGS+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
fi

if [[ "${ENABLE_TORCH_COMPILE:-0}" == "1" ]]; then
  COMMON_SGLANG_ARGS+=(--enable-torch-compile)
fi

if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  COMMON_SGLANG_ARGS+=(--disable-radix-cache)
fi

case "${ENABLE_HICACHE:-0}" in
  1|true|TRUE|True|on|ON|On|yes|YES|Yes)
    COMMON_SGLANG_ARGS+=(--enable-hierarchical-cache)
    if [[ -n "${HICACHE_RATIO:-}" ]]; then
      COMMON_SGLANG_ARGS+=(--hicache-ratio "${HICACHE_RATIO}")
    fi
    if [[ -n "${HICACHE_IO_BACKEND:-}" ]]; then
      COMMON_SGLANG_ARGS+=(--hicache-io-backend "${HICACHE_IO_BACKEND}")
    fi
    if [[ -n "${HICACHE_STORAGE_PREFETCH_POLICY:-}" ]]; then
      COMMON_SGLANG_ARGS+=(--hicache-storage-prefetch-policy "${HICACHE_STORAGE_PREFETCH_POLICY}")
    fi
    if [[ -n "${HICACHE_WRITE_POLICY:-}" ]]; then
      COMMON_SGLANG_ARGS+=(--hicache-write-policy "${HICACHE_WRITE_POLICY}")
    fi
    ;;
esac
