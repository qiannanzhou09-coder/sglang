#!/usr/bin/env bash
set -euo pipefail

VALUE="${1:?usage: serve_schedule.sh <lpm|fcfs|lof>}"
export SCHEDULE_POLICY="${VALUE}"
export CONFIG_NAME="${CONFIG_NAME:-tp8_sched_${VALUE}}"

exec bash "$(dirname "${BASH_SOURCE[0]}")/serve_tp8_variant.sh"
