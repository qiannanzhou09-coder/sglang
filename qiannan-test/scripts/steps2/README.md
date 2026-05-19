# Stage 2 TP8 Tuning Scripts

Stage 1 now points to pure `TP8` as the main line. These scripts only tune TP8
server parameters. They do not start DPA or the router.

Each client run takes exactly one dataset argument, same as the earlier scripts.
`full` is supported and maps to `qiannan-test/bench/workload_data.jsonl`; every
candidate that looks good on fixtures should be confirmed with `full`.

## Start A Variant

Generic entry:

```bash
bash qiannan-test/scripts/steps2/serve_tp8_variant.sh
```

Common single-knob wrappers:

```bash
bash qiannan-test/scripts/steps2/serve_chunked.sh 16384
bash qiannan-test/scripts/steps2/serve_mem.sh 0.9
bash qiannan-test/scripts/steps2/serve_spec.sh 5
bash qiannan-test/scripts/steps2/serve_spec.sh off
bash qiannan-test/scripts/steps2/serve_schedule.sh fcfs
bash qiannan-test/scripts/steps2/serve_kv_cache.sh fp8_e4m3
```

## Run One Dataset

```bash
bash qiannan-test/scripts/steps2/client_tp8.sh full
bash qiannan-test/scripts/steps2/client_tp8.sh T6
bash qiannan-test/scripts/steps2/client_tp8.sh T4
bash qiannan-test/scripts/steps2/client_tp8.sh qiannan-test/fixtures/T3_burst_long_input.jsonl
```

Useful smoke-test override:

```bash
MAX_SESSIONS=2 bash qiannan-test/scripts/steps2/client_tp8.sh full
```

## Recommended Order

1. Keep the current TP8 baseline and run `full`.
2. Sweep `chunked-prefill-size`: `4096`, `8192`, `16384`, `32768`.
3. Sweep `mem-fraction-static`: `0.8`, `0.85`, `0.9`, then `0.92/0.95` only if stable.
4. Sweep NEXTN steps on decode-heavy and full: `off`, `1`, `3`, `5`, `7`.
5. Compare scheduler policy: `lpm`, `fcfs`, `lof`.
6. Try `fp8_e4m3` KV only with accuracy guardrails; if it fails, try `fp8_e5m2`.

Primary decision metric stays `avg_session_time` on `full`. Fixture runs are for
diagnosis and cheaper iteration.

## Advanced Environment Knobs

`serve_tp8_variant.sh` accepts these env vars:

```bash
MEM_FRACTION_STATIC=0.9
CHUNKED_PREFILL_SIZE=16384
SCHEDULE_POLICY=lpm
SCHEDULE_CONSERVATIVENESS=1.5
SPECULATIVE_ALGO=NEXTN
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_NUM_DRAFT_TOKENS=4
MIXED_CHUNK=0
KV_CACHE_DTYPE=fp8_e4m3
MAX_RUNNING_REQUESTS=256
MAX_PREFILL_TOKENS=32768
CUDA_GRAPH_MAX_BS=512
ATTENTION_BACKEND=fa3
ENABLE_HICACHE=1
HICACHE_RATIO=4
HICACHE_IO_BACKEND=kernel
HICACHE_WRITE_POLICY=write_through
HICACHE_STORAGE_PREFETCH_POLICY=wait_complete
```
