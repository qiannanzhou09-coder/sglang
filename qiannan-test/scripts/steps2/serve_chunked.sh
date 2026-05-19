#!/usr/bin/env bash
set -euo pipefail

VALUE="${1:?usage: serve_chunked.sh <4096|8192|16384|32768>}"
export CHUNKED_PREFILL_SIZE="${VALUE}"
export CONFIG_NAME="${CONFIG_NAME:-tp8_chunk${VALUE}}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/serve_tp8_variant.sh"
