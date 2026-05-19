#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-E_dpa_tp8_dp2_router_pruned_spec_off_full_workload}"
RUN_ID="${RUN_ID:-E_pruned_dpa_tp8_dp2_router_spec_off_full_$(date +%Y%m%d_%H%M%S)}"
export CONFIG RUN_ID

exec bash "${SCRIPT_DIR}/client_E_dpa_tp8_dp2_router_full.sh"
