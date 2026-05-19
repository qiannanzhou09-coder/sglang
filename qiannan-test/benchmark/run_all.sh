#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == "list" ]]; then
  exec python3 "${SCRIPT_DIR}/run_benchmark.py" list
fi
exec python3 "${SCRIPT_DIR}/run_benchmark.py" all "$@"
