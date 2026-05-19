#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NONE}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.92}"
export ENABLE_HICACHE="${ENABLE_HICACHE:-1}"
export HICACHE_RATIO="${HICACHE_RATIO:-2}"
export HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
export HICACHE_STORAGE_PREFETCH_POLICY="${HICACHE_STORAGE_PREFETCH_POLICY:-wait_complete}"
export HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"

exec bash "${SCRIPT_DIR}/serve_E_dpa_tp8_dp2.sh"
