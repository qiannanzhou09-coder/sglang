#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

# Generic TP8 server entry for stage-2 tuning.
# Override knobs with environment variables, for example:
#   CHUNKED_PREFILL_SIZE=16384 CONFIG_NAME=tp8_chunk16384 bash serve_tp8_variant.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"

CONFIG_NAME="${CONFIG_NAME:-tp8_steps2}"
MODEL_PATH="${MODEL_PATH:-/home/Qwen3.5-122B-A10B}"
HOST="${HOST:-0.0.0.0}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
MIXED_CHUNK="${MIXED_CHUNK:-1}"

# Speculative decoding. Set SPECULATIVE_ALGO=none to disable.
SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NEXTN}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-4}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --host "${HOST}"
  --port "${SGLANG_PORT}"
  --tp-size 8
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

if [[ "${MIXED_CHUNK}" == "1" || "${MIXED_CHUNK}" == "true" || "${MIXED_CHUNK}" == "on" ]]; then
  ARGS+=(--enable-mixed-chunk)
fi

if [[ -n "${SCHEDULE_CONSERVATIVENESS:-}" ]]; then
  ARGS+=(--schedule-conservativeness "${SCHEDULE_CONSERVATIVENESS}")
fi

if [[ "${SPECULATIVE_ALGO}" != "none" && "${SPECULATIVE_ALGO}" != "off" ]]; then
  ARGS+=(
    --speculative-algorithm "${SPECULATIVE_ALGO}"
    --speculative-num-steps "${SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk "${SPECULATIVE_EAGLE_TOPK}"
    --speculative-num-draft-tokens "${SPECULATIVE_NUM_DRAFT_TOKENS}"
  )
fi

if [[ -n "${KV_CACHE_DTYPE:-}" && "${KV_CACHE_DTYPE}" != "auto" ]]; then
  ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi

if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
  ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
fi
if [[ -n "${PREFILL_ATTENTION_BACKEND:-}" ]]; then
  ARGS+=(--prefill-attention-backend "${PREFILL_ATTENTION_BACKEND}")
fi
if [[ -n "${DECODE_ATTENTION_BACKEND:-}" ]]; then
  ARGS+=(--decode-attention-backend "${DECODE_ATTENTION_BACKEND}")
fi

if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
  ARGS+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
fi
if [[ -n "${MAX_PREFILL_TOKENS:-}" ]]; then
  ARGS+=(--max-prefill-tokens "${MAX_PREFILL_TOKENS}")
fi
if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
  ARGS+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
fi

if [[ "${ENABLE_TORCH_COMPILE:-0}" == "1" ]]; then
  ARGS+=(--enable-torch-compile)
fi
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  ARGS+=(--disable-radix-cache)
fi

if [[ "${ENABLE_HICACHE:-0}" == "1" ]]; then
  ARGS+=(--enable-hierarchical-cache)
  if [[ -n "${HICACHE_RATIO:-}" ]]; then
    ARGS+=(--hicache-ratio "${HICACHE_RATIO}")
  fi
  if [[ -n "${HICACHE_IO_BACKEND:-}" ]]; then
    ARGS+=(--hicache-io-backend "${HICACHE_IO_BACKEND}")
  fi
  if [[ -n "${HICACHE_WRITE_POLICY:-}" ]]; then
    ARGS+=(--hicache-write-policy "${HICACHE_WRITE_POLICY}")
  fi
  if [[ -n "${HICACHE_STORAGE_PREFETCH_POLICY:-}" ]]; then
    ARGS+=(--hicache-storage-prefetch-policy "${HICACHE_STORAGE_PREFETCH_POLICY}")
  fi
fi

if [[ -n "${EXTRA_SGLANG_ARGS:-}" ]]; then
  # Intentionally simple whitespace splitting for ad-hoc flags.
  # Prefer explicit env vars above for values containing spaces.
  EXTRA_ARGS=(${EXTRA_SGLANG_ARGS})
  ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "[steps2:${CONFIG_NAME}] Starting TP8 variant on :${SGLANG_PORT}"
echo "    Client target: http://<host>:${SGLANG_PORT}"
echo "    Dataset is selected on the client side, e.g. bash ${SCRIPT_DIR}/client_tp8.sh full"

python -m sglang.launch_server "${ARGS[@]}"
