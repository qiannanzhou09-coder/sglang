#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PRUNE_TOOL_PATH="${PRUNE_TOOL_PATH:-${ROOT_DIR}/tools/prune_reap_ckpt.py}"
PRUNE_PLAN_PATH="${PRUNE_PLAN_PATH:-${ROOT_DIR}/data/targeted_refusal_analysis.json}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/Qwen3.5-122B-A10B}"
SOURCE_MODEL_PATH="${PRUNE_SOURCE_MODEL_PATH:-${BASE_MODEL_PATH}}"
PRUNED_MODEL_ROOT="${PRUNED_MODEL_ROOT:-/home/qwen3.5-pruned-models}"
PRUNE_WORKERS="${PRUNE_WORKERS:-1}"
PRUNE_MTP_POLICY="${PRUNE_MTP_POLICY:-prune-with-lm0}"
FORCE_REBUILD_PRUNED_MODEL="${FORCE_REBUILD_PRUNED_MODEL:-0}"

log() {
  printf '[prepare_pruned_model] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

truthy() {
  case "${1:-}" in
    1|true|TRUE|True|on|ON|On|yes|YES|Yes)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_mode() {
  case "${1:-none}" in
    ""|none|NONE|None|off|OFF|Off|0|false|FALSE|False)
      printf 'none\n'
      ;;
    strip_visual|strip-visual|text_only|text-only)
      printf 'strip_visual\n'
      ;;
    reap|reap_only|reap-only)
      printf 'reap\n'
      ;;
    reap_text_only|reap-text-only|reap_strip_visual|reap-strip-visual)
      printf 'reap_text_only\n'
      ;;
    *)
      die "unsupported PRUNE_MODE='${1}'"
      ;;
  esac
}

validate_model_dir() {
  local model_dir="$1"
  [[ -d "${model_dir}" ]] || die "model directory does not exist: ${model_dir}"
  [[ -f "${model_dir}/config.json" ]] || die "missing config.json under ${model_dir}"
  [[ -f "${model_dir}/model.safetensors.index.json" ]] || die "missing model.safetensors.index.json under ${model_dir}"

  shopt -s nullglob
  local shards=("${model_dir}"/*.safetensors)
  shopt -u nullglob
  (( ${#shards[@]} > 0 )) || die "no .safetensors shards found under ${model_dir}"
}

expected_reap_num_experts() {
  "${PYTHON_BIN}" -c '
import json
import sys

plan_path, config_path = sys.argv[1:3]
plan = json.load(open(plan_path))
cfg = json.load(open(config_path))

def num_experts(config):
    vals = []
    for parent in (config, config.get("text_config"), config.get("language_model_config")):
        if not isinstance(parent, dict):
            continue
        for key in ("num_experts", "num_routed_experts", "n_routed_experts"):
            if key in parent:
                vals.append(int(parent[key]))
    if not vals:
        raise SystemExit("missing num_experts in source config")
    if len(set(vals)) != 1:
        raise SystemExit(f"inconsistent source num_experts values: {vals}")
    return vals[0]

counts = {len(layer["pruned_indices"]) for layer in plan["layers"]}
if len(counts) != 1:
    raise SystemExit(f"non-uniform pruned_indices counts: {sorted(counts)}")

print(num_experts(cfg) - counts.pop())
' "${PRUNE_PLAN_PATH}" "${SOURCE_MODEL_PATH}/config.json"
}

actual_num_experts() {
  "${PYTHON_BIN}" -c '
import json
import sys

cfg = json.load(open(sys.argv[1]))
vals = []
for parent in (cfg, cfg.get("text_config"), cfg.get("language_model_config")):
    if not isinstance(parent, dict):
        continue
    for key in ("num_experts", "num_routed_experts", "n_routed_experts"):
        if key in parent:
            vals.append(int(parent[key]))
if not vals:
    raise SystemExit("missing num_experts in pruned config")
print(vals[0])
' "$1/config.json"
}

validate_reap_config() {
  local model_dir="$1"
  local expected
  local actual
  expected="$(expected_reap_num_experts)"
  actual="$(actual_num_experts "${model_dir}")"
  [[ "${actual}" == "${expected}" ]] || die "num_experts mismatch for ${model_dir}: expected ${expected}, got ${actual}"
}

mode="$(normalize_mode "${PRUNE_MODE:-none}")"
if [[ "${mode}" == "none" ]]; then
  printf '%s\n' "${SOURCE_MODEL_PATH}"
  exit 0
fi

[[ -f "${PRUNE_TOOL_PATH}" ]] || die "missing prune tool: ${PRUNE_TOOL_PATH}"
validate_model_dir "${SOURCE_MODEL_PATH}"

source_model_name="${SOURCE_MODEL_PATH##*/}"
case "${mode}" in
  strip_visual)
    default_target="${PRUNED_MODEL_ROOT}/${source_model_name}-text-only"
    ;;
  reap)
    default_target="${PRUNED_MODEL_ROOT}/${source_model_name}-REAP-20"
    ;;
  reap_text_only)
    default_target="${PRUNED_MODEL_ROOT}/${source_model_name}-REAP-20-text-only"
    ;;
esac

TARGET_MODEL_PATH="${PRUNED_MODEL_PATH:-${default_target}}"

if [[ -d "${TARGET_MODEL_PATH}" ]] && ! truthy "${FORCE_REBUILD_PRUNED_MODEL}"; then
  log "reusing existing ${mode} model: ${TARGET_MODEL_PATH}"
  validate_model_dir "${TARGET_MODEL_PATH}"
  case "${mode}" in
    reap|reap_text_only)
      validate_reap_config "${TARGET_MODEL_PATH}"
      ;;
  esac
  printf '%s\n' "${TARGET_MODEL_PATH}"
  exit 0
fi

mkdir -p "${PRUNED_MODEL_ROOT}"

cmd=(
  "${PYTHON_BIN}" "${PRUNE_TOOL_PATH}"
  --src "${SOURCE_MODEL_PATH}"
  --dst "${TARGET_MODEL_PATH}"
  --workers "${PRUNE_WORKERS}"
)

case "${mode}" in
  strip_visual)
    cmd+=(--strip-visual)
    ;;
  reap)
    [[ -f "${PRUNE_PLAN_PATH}" ]] || die "missing REAP plan: ${PRUNE_PLAN_PATH}"
    cmd+=(--plan "${PRUNE_PLAN_PATH}" --mtp-policy "${PRUNE_MTP_POLICY}")
    ;;
  reap_text_only)
    [[ -f "${PRUNE_PLAN_PATH}" ]] || die "missing REAP plan: ${PRUNE_PLAN_PATH}"
    cmd+=(--plan "${PRUNE_PLAN_PATH}" --mtp-policy "${PRUNE_MTP_POLICY}" --strip-visual)
    ;;
esac

if truthy "${FORCE_REBUILD_PRUNED_MODEL}"; then
  cmd+=(--force)
fi

log "building ${mode} model"
log "source: ${SOURCE_MODEL_PATH}"
log "target: ${TARGET_MODEL_PATH}"
"${cmd[@]}"

validate_model_dir "${TARGET_MODEL_PATH}"
case "${mode}" in
  reap|reap_text_only)
    validate_reap_config "${TARGET_MODEL_PATH}"
    ;;
esac

printf '%s\n' "${TARGET_MODEL_PATH}"
