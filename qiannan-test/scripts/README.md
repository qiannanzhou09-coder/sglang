# SGLang Architecture Startup Scripts

These scripts start the four architecture candidates used by the T1/T6 benchmark.

## A. Pure TP baseline

```bash
bash qiannan-test/scripts/serve_A_tp8.sh
```

Client target: `http://localhost:8000`

## B. DPA + cache-aware router

Shell 1:

```bash
bash qiannan-test/scripts/serve_B_dpa_tp8_dp8.sh
```

Shell 2, after the server is healthy:

```bash
bash qiannan-test/scripts/router_B_dp8_cache_aware.sh
```

Client target: `http://localhost:30000`

## C. DPA without router

```bash
bash qiannan-test/scripts/serve_C_dpa_tp8_dp8_no_router.sh
```

Client target: `http://localhost:8000`

## D. Medium DPA + cache-aware router

Shell 1:

```bash
bash qiannan-test/scripts/serve_D_dpa_tp4_dp2.sh
```

Shell 2, after the server is healthy:

```bash
bash qiannan-test/scripts/router_D_dp2_cache_aware.sh
```

Client target: `http://localhost:30000`

## E. TP8 + DP2 DPA + cache-aware router

This keeps the full 8-GPU TP world for dense/MoE layers while using attention
DP=2 and attention TP=4. The server script defaults to compensated DPA values:
effective `chunked-prefill-size=8192` and effective
`schedule-conservativeness~=1.0`.

Shell 1:

```bash
bash qiannan-test/scripts/serve_E_dpa_tp8_dp2.sh
```

Shell 2, after the server is healthy:

```bash
bash qiannan-test/scripts/router_E_dp2_cache_aware.sh
```

Client target: `http://localhost:30000`

## F. TP8 + DP4 DPA + cache-aware router

This keeps the full 8-GPU TP world for dense/MoE layers while using attention
DP=4 and attention TP=2. The server script defaults to compensated DPA values:
effective `chunked-prefill-size=8192` and effective
`schedule-conservativeness~=1.0`.

Shell 1:

```bash
bash qiannan-test/scripts/serve_F_dpa_tp8_dp4.sh
```

Shell 2, after the server is healthy:

```bash
bash qiannan-test/scripts/router_F_dp4_cache_aware.sh
```

Client target: `http://localhost:30000`

## Common Overrides

```bash
MODEL_PATH=/path/to/model bash qiannan-test/scripts/serve_A_tp8.sh
SGLANG_PORT=8100 bash qiannan-test/scripts/serve_A_tp8.sh
ROUTER_PORT=31000 SGLANG_PORT=8100 bash qiannan-test/scripts/router_B_dp8_cache_aware.sh
```

All server scripts enable `--enable-metrics` and `--enable-cache-report`.

## Startup-Time Pruning

All server scripts source `_common_sglang.sh`, so they can prepare and serve a
pruned checkpoint before launching SGLang:

```bash
PRUNE_MODE=strip_visual bash qiannan-test/scripts/serve_A_tp8.sh
PRUNE_MODE=reap bash qiannan-test/scripts/serve_A_tp8.sh
PRUNE_MODE=reap_text_only bash qiannan-test/scripts/serve_A_tp8.sh
```

Defaults:

```text
BASE_MODEL_PATH=/home/Qwen3.5-122B-A10B
PRUNE_PLAN_PATH=qiannan-test/document/targeted_refusal_analysis.json
PRUNED_MODEL_ROOT=/home/qwen3.5-pruned-models
```

Supported `PRUNE_MODE` values:

```text
none            Serve BASE_MODEL_PATH directly.
strip_visual    Drop model.visual.* and vision config fields.
reap            Apply the REAP expert plan.
reap_text_only  Apply REAP and drop visual weights.
```

The pruned checkpoint is reused if it already exists. Set
`FORCE_REBUILD_PRUNED_MODEL=1` to rebuild it. When `PRUNE_MODE` is set, use
`BASE_MODEL_PATH` for the source checkpoint and `PRUNED_MODEL_PATH` only when
you want to force a specific output checkpoint directory.

## Client Workload Scripts

Run one fixture against the matching endpoint:

```bash
bash qiannan-test/scripts/client_A_tp8.sh T1
bash qiannan-test/scripts/client_A_tp8.sh T6

bash qiannan-test/scripts/client_B_dpa_tp8_dp8_router.sh T1
bash qiannan-test/scripts/client_B_dpa_tp8_dp8_router.sh T6

bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T1
bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T6

bash qiannan-test/scripts/client_D_dpa_tp4_dp2_router.sh T1
bash qiannan-test/scripts/client_D_dpa_tp4_dp2_router.sh T6

bash qiannan-test/scripts/client_E_dpa_tp8_dp2_router.sh T1
bash qiannan-test/scripts/client_E_dpa_tp8_dp2_router.sh T6

bash qiannan-test/scripts/client_F_dpa_tp8_dp4_router.sh T1
bash qiannan-test/scripts/client_F_dpa_tp8_dp4_router.sh T6
```

Each run writes:

```text
qiannan-test/runs/<run_id>/
  config.yaml
  client_stdout.log
  client_metrics.jsonl
  summary.json
```

Useful quick-test overrides:

```bash
MAX_SESSIONS=2 bash qiannan-test/scripts/client_A_tp8.sh T1
RUN_ID=A_t1_smoke bash qiannan-test/scripts/client_A_tp8.sh T1
BASE_URL=http://localhost:31000 bash qiannan-test/scripts/client_B_dpa_tp8_dp8_router.sh T6
```

## Full Workload Client Scripts

These wrappers run `qiannan-test/bench/workload_data.jsonl` instead of T1/T6.

```bash
bash qiannan-test/scripts/client_A_tp8_full.sh
bash qiannan-test/scripts/client_B2_dpa_tp8_dp8_router_compensated_full.sh
bash qiannan-test/scripts/client_E_dpa_tp8_dp2_router_full.sh
bash qiannan-test/scripts/client_F_dpa_tp8_dp4_router_full.sh
```

## Stage 2 Best Config

Server:

```bash
SPECULATIVE_ALGO=NONE MEM_FRACTION_STATIC=0.92 ENABLE_HICACHE=1 HICACHE_RATIO=2 HICACHE_IO_BACKEND=kernel HICACHE_STORAGE_PREFETCH_POLICY=wait_complete HICACHE_WRITE_POLICY=write_through bash qiannan-test/scripts/serve_E_dpa_tp8_dp2.sh
```

Equivalent wrapper:

```bash
bash qiannan-test/scripts/serve_E_stage2_best_hicache.sh
```

Router:

```bash
bash qiannan-test/scripts/router_E_dp2_cache_aware.sh
```

Full workload client:

```bash
SUMMARY_INTERVAL=60 bash qiannan-test/scripts/client_E_dpa_tp8_dp2_router_full.sh
```

`RUN_DIR/interval_summaries.jsonl` 的顶层字段仍是 0 到当前时间的累计统计；
每条记录里的 `interval_delta` 是上一次 summary 到本次 summary 之间新完成 round 的窗口统计，
包括该窗口内的 tokens、吞吐、TTFT 和 round latency 分位数。

Machine-readable record: `qiannan-test/results/stage2_best_config.json`.
