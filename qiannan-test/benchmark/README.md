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

Run the figure-data cases:

```bash
bash qiannan-test/benchmark/run_data.sh list
bash qiannan-test/benchmark/run_data.sh d02_t6_dp2_router_hicache
bash qiannan-test/benchmark/run_data.sh all --continue-on-error
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
timestamp, for example `04_admission_pruned_dp2tp8_hicache_20260519_164704`.

Each throughput run directory contains `resolved_config.json`, `commands.json`,
logs for the started components, `client_metrics.jsonl`,
`interval_summaries.jsonl`, and `summary.json`. Loss-only runs replace the
client metrics with `loss_metrics.jsonl` and `loss_summary.json`.

Figure-data cases live under `configs-data/` and write outputs under:

```text
qiannan-test/figure-runs/<case_id>/
```

These cases enable the SGLang load sampler by default. It polls
`/v1/loads?include=all` on the direct SGLang server and writes
`load_metrics.jsonl`. The sampler does not call `nvidia-smi`; it records
SGLang-reported per-DP metrics such as `num_running_reqs`,
`num_total_tokens`, `token_usage`, `cache_hit_rate`, and the nested
`memory.weight_gb`, `memory.kv_cache_gb`, `memory.graph_gb`, and
`memory.token_capacity` fields.

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
| 4 | `04_admission_pruned_dp2tp8_hicache.json` | Adds pruned model; keeps profile `hicache.ratio=1.72`. |
| 5 | `05_tp8_hicache.json` | Disables DP/router; keeps TP8 + HiCache. |
| 6 | `06_dp8tp8_hicache.json` | Changes DP from 2 to 8. |
| 7 | `07_admission_dp8tp8_hicache.json` | Case 6 plus dynamic admission proxy. |
| 8 | `08_admission_pruned_dp2tp8_hicache_ratio4_kvfp8.json` | Pruned model with `hicache.ratio=4` plus `kv_cache_dtype=fp8_e4m3`. |
| 9 | `09_loss_base_dp2tp8_hicache.json` | Loss-only scoring for the base model. |
| 10 | `10_loss_pruned_dp2tp8_hicache_ratio4.json` | Loss-only scoring for the pruned model. |

The workload always uses `summary_interval=30` from the profile unless a case
explicitly overrides it.

## Figure-Data Configs

`configs-data/*.json` is separated from the main 01-10 cases so diagnostic
figures do not pollute the primary benchmark matrix.

| Config | Purpose |
| --- | --- |
| `d01_t6_tp8_hicache` | TP8 T6 baseline. |
| `d02_t6_dp2_router_hicache` | DP2 + router T6 baseline. |
| `d03_t6_dp8_router_hicache` | DP8 + router T6 comparison and DP8 heatmap source. |
| `d04_t6_dp8_no_router_hicache` | DP8 without router for router-value comparison. |
| `d05_t6_dp2_router_no_hicache` | HiCache-off T6 comparison. |
| `d06`-`d09` | `mem_fraction_static` sweep: 0.80, 0.88, 0.92, 0.95. |
| `d10_memory_base_dp2tp8_hicache` | Server-ready memory sample for the base model. |
| `d11_memory_pruned_dp2tp8_hicache` | Server-ready memory sample for the pruned model. |
| `d12_t6_dp2_router_kvfp8_hicache` | T6 FP8 KV cache comparison. |
| `d13_t5_dp2_router_hicache` | T5 long-wait HiCache-on late-cache-hit source. |
| `d14_t5_dp2_router_no_hicache` | T5 long-wait HiCache-off late-cache-hit source. |

## Loss Scoring

Loss cases reuse the same runner, but disable the throughput client and call
`benchmark/run_loss.py` after the SGLang server is ready. The loss client reads
fixed rows with `prompt_ids` and `gen_ids`, sends `prompt_ids + gen_ids` to the
server with:

```json
{
  "sampling_params": {"temperature": 0, "max_new_tokens": 0},
  "return_logprob": true,
  "return_text_in_logprobs": false
}
```

It then computes NLL only on the continuation tokens:

```text
nll = mean_t -log p_model(gen_ids[t] | prompt_ids + gen_ids[:t])
```

By default the profile points to:

```text
qiannan-test/data/generations.jsonl
```

and takes `per_domain=20` rows from each domain. Override these in a config's
`loss` block if you want a different sample set, larger coverage, or lower
concurrency. For a pruning-quality comparison in the same style as
`eval_qwen3.5_122b`, generate this file with the candidate pruned model first;
otherwise the result is a fixed-corpus NLL score rather than NLL on that
candidate model's own samples.

The loss-only cases also lower server memory pressure with
`mem_fraction_static=0.70`, `context_length=8192`, `max_running_requests=32`,
and `prefill_max_requests=4`. Keep loss concurrency conservative because
`return_logprob=true` materializes logprob buffers that normal generation does
not need.

Run the two loss cases:

```bash
bash qiannan-test/benchmark/run_one.sh 09_loss_base_dp2tp8_hicache
bash qiannan-test/benchmark/run_one.sh 10_loss_pruned_dp2tp8_hicache_ratio4
```

Each run directory gets `loss_metrics.jsonl`, `loss_summary.json`, and
`loss_console.log`. Compare base and pruned outputs with:

```bash
python3 qiannan-test/benchmark/compare_loss.py \
  --base qiannan-test/benchmark_runs/09_loss_base_dp2tp8_hicache/loss_metrics.jsonl \
  --candidate qiannan-test/benchmark_runs/10_loss_pruned_dp2tp8_hicache_ratio4/loss_metrics.jsonl \
  --markdown-output qiannan-test/benchmark_runs/loss_compare.md
```
