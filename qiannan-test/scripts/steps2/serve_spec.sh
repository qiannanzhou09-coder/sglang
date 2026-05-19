#!/usr/bin/env bash
set -euo pipefail

VALUE="${1:?usage: serve_spec.sh <off|1|3|5|7>}"

if [[ "${VALUE}" == "off" || "${VALUE}" == "none" ]]; then
  export SPECULATIVE_ALGO="none"
  export CONFIG_NAME="${CONFIG_NAME:-tp8_spec_off}"
else
  export SPECULATIVE_ALGO="${SPECULATIVE_ALGO:-NEXTN}"
  export SPECULATIVE_NUM_STEPS="${VALUE}"
  export SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-$((VALUE + 1))}"
  export CONFIG_NAME="${CONFIG_NAME:-tp8_spec${VALUE}}"
fi

exec bash "$(dirname "${BASH_SOURCE[0]}")/serve_tp8_variant.sh"
