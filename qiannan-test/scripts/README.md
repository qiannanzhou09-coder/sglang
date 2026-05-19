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

## Common Overrides

```bash
MODEL_PATH=/path/to/model bash qiannan-test/scripts/serve_A_tp8.sh
SGLANG_PORT=8100 bash qiannan-test/scripts/serve_A_tp8.sh
ROUTER_PORT=31000 SGLANG_PORT=8100 bash qiannan-test/scripts/router_B_dp8_cache_aware.sh
```

All server scripts enable `--enable-metrics` and `--enable-cache-report`.

