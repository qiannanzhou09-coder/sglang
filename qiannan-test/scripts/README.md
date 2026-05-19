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

Machine-readable record: `qiannan-test/results/stage2_best_config.json`.
