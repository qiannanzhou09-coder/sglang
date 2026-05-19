#!/usr/bin/env bash
set -euo pipefail
PS4='+ ${BASH_SOURCE}:${LINENO}: '
set -x

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python3 -m sglang.launch_server \
--model-path /home/Qwen3.5-122B-A10B \
--host 0.0.0.0 \
--port 8000 \
--tp-size 8 \
--mem-fraction-static 0.8 \
--context-length 262144 \
--reasoning-parser qwen3 \
--speculative-algo NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4 \
--tool-call-parser qwen3_coder \
--mamba-scheduler-strategy extra_buffer \
--chunked-prefill-size 8192 \
--enable-mixed-chunk \
--schedule-policy lpm \
--enable-cache-report



python3 run_workload.py \
--data workload_data.jsonl \
--base-url http://localhost:8000 \
--model default \
--max-sessions 5


python - <<'PY'
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor
print("ok")
PY

python - <<'PY'
import deep_gemm, importlib.util
print("deep_gemm __file__:", deep_gemm.__file__)
print("deep_gemm __path__:", list(deep_gemm.__path__))
print("layout spec:", importlib.util.find_spec("deep_gemm.utils.layout"))
from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor
print("ok")
PY
