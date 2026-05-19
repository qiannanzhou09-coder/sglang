# Benchmark Runner

The benchmark logic is self-contained in this directory. It does not call the
older startup shell wrappers.

Run all cases:

```bash
bash qiannan-test/benchmark/run_all.sh
```

Run one case:

```bash
bash qiannan-test/benchmark/run_one.sh 01_dp2tp8_hicache
```

List cases:

```bash
bash qiannan-test/benchmark/run_one.sh list
```

All outputs are written under:

```text
qiannan-test/benchmark_runs/<case_id>/
```

If that case directory already exists and is not empty, the runner appends a
timestamp, for example `04_admission_pruned_dp2tp8_hicache_ratio4_20260519_164704`.

Each run directory contains `resolved_config.json`, `commands.json`, logs for
the started components, `client_metrics.jsonl`, `interval_summaries.jsonl`, and
`summary.json`.

## Config Layout

`profiles/stage2_hicache.json` holds the shared baseline copied from
`results/stage2_best_config.json` and the existing SGLang startup defaults
needed to run the server.

`configs/*.json` holds the ordered cases and only the differences from the
shared profile:

| Order | Config file | Difference |
| --- | --- | --- |
| 1 | `01_dp2tp8_hicache.json` | Stage-2 best config. |
| 2 | `02_admission_dp2tp8_hicache.json` | Adds dynamic admission proxy. |
| 3 | `03_dp2tp8_hicache_kvfp8.json` | Adds `kv_cache_dtype=fp8_e4m3`. |
| 4 | `04_admission_pruned_dp2tp8_hicache_ratio4.json` | Adds pruned model and `hicache.ratio=4`. |
| 5 | `05_tp8_hicache.json` | Disables DP/router; keeps TP8 + HiCache. |
| 6 | `06_dp8tp8_hicache.json` | Changes DP from 2 to 8. |
| 7 | `07_admission_dp8tp8_hicache.json` | Case 6 plus dynamic admission proxy. |
| 8 | `08_admission_pruned_dp2tp8_hicache_ratio4_kvfp8.json` | Case 4 plus `kv_cache_dtype=fp8_e4m3`. |

The workload always uses `summary_interval=30` from the profile unless a case
explicitly overrides it.
