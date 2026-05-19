#!/usr/bin/env bash
set -euo pipefail

VALUE="${1:?usage: serve_mem.sh <0.8|0.85|0.9|0.92|0.95>}"
SAFE_VALUE="${VALUE//./p}"
export MEM_FRACTION_STATIC="${VALUE}"
export CONFIG_NAME="${CONFIG_NAME:-tp8_mem${SAFE_VALUE}}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/serve_tp8_variant.sh"
