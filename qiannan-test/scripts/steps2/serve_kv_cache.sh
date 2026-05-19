#!/usr/bin/env bash
set -euo pipefail

VALUE="${1:?usage: serve_kv_cache.sh <auto|fp8_e4m3|fp8_e5m2|bf16>}"
export KV_CACHE_DTYPE="${VALUE}"
export CONFIG_NAME="${CONFIG_NAME:-tp8_kv_${VALUE}}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/serve_tp8_variant.sh"
