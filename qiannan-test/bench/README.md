# Qwen3.5-122B Serving Workload Test

LLM Serving 系统的性能测试与评测工具集，针对 Qwen3.5-122B-A10B-FP8 模型，分别使用 SGLang 和 vLLM 两种 Serving 框架。

## 文件说明

### Serving 启动

| 文件 | 说明 |
|------|------|
| `sglang_serve.sh` | 使用 SGLang 拉起模型 Serving（TP=8，262k context，NEXTN 投机解码，Prefix Cache，Chunked Prefill） |
| `vllm_serve.sh` | 使用 vLLM 拉起模型 Serving（TP=8，262k context，MTP 投机解码） |

### Workload 压测
`run_workload.py` 输出结果可参考 `workload_metrics.jsonl`。课题优化目标主要为 `avg_session_time`（单个 session 内所有轮的**执行时间**之和，不含轮间 `wait` 等待；所有 session 取平均，单位：秒），同时也需关注 `TTFT`、`round_latency` 等其他指标。

| 文件 | 说明 |
|------|------|
| `workload_data.jsonl` | 压测数据文件，共 500 个 session（每行一行 JSON），包含 `system_prompt` 与多轮 `{wait, input, output}` |
| `run_workload.py` | 异步并发压测客户端，模拟多 session 多轮对话，统计 TTFT/吞吐/延迟，支持 streaming、重试、ignore_eos |
| `workload_metrics.jsonl` | 压测结果输出文件，包含每轮详细指标和整体 summary |

### Benchmark 评测
用于验证模型正确性。如果`sglang_bench.sh`和`vllm_bench.sh`的结果loss过大则认为模型不正确，结果无效。具体loss阈值稍后发布。
| 文件 | 说明 |
|------|------|
| `eval_download_dataset.sh` | 提前下载 lm_eval 评测数据集（gsm8k 等），避免运行时联网 |
| `sglang_bench.sh` | 对 SGLang Serving 跑 lm_eval 评测（local-completions 模式，model=default） |
| `vllm_bench.sh` | 对 vLLM Serving 跑 lm_eval 评测（local-completions 模式，需指定完整模型路径） |
| `lm-evaluation-harness/` | lm_eval 框架源码，支持通过 OpenAI-compatible API 连接已拉起的 Serving 进行评测 |

## 使用流程

```bash
# 1. 拉起 Serving（二选一）
bash sglang_serve.sh
# 或
bash vllm_serve.sh

# 2. 跑 Workload 压测
python run_workload.py  # 全量压测

# 3. 跑 Benchmark 评测
bash eval_download_dataset.sh             # 首次需下载数据集
bash sglang_bench.sh                 
# 或 
bash vllm_bench.sh
```

## 关键参数

- `run_workload.py --base-url URL`：Serving 端点（默认 `http://localhost:8000`）
- `run_workload.py --max-sessions N`：限制 session 数（快速验证用）
- `run_workload.py --no-ignore-eos`：不强制输出长度（vLLM 不支持 ignore_eos 时使用）
- `run_workload.py --model NAME`：指定模型名（vLLM 需要匹配注册的模型路径）
