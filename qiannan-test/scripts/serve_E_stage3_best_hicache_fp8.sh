#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage 3: reuse the Stage 2 best HiCache config, but force FP8 KV cache.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

exec bash "${SCRIPT_DIR}/serve_E_stage2_best_hicache.sh"
