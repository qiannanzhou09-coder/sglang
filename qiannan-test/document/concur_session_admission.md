# CONCUR-style session admission 改造说明

本文记录 `qiannan-test/proxy` 本次准入控制改造。目标是把之前的“请求级动态限流”改成更接近 CONCUR 论文的“智能体/session 级动态准入控制”。

## 结论

已经把动态准入的控制对象从 request inflight window 改为 active session window。

新的动态窗口含义是：

```text
current_session_limit = 当前允许继续发 LLM generation request 的 active session 数上限
```

例如：

```text
active_sessions = 64
current_session_limit 从 64 乘法降低到 32
需要从当前 active session 集合中挑 32 个 session 暂停
```

暂停不是杀请求，也不是删除 session。已经发到 SGLang 的 round 会继续跑完；被选中的 session 在当前 round 完成后进入 paused 状态，下一轮请求会在 proxy 内等待，直到动态窗口恢复并允许 resume。

## 为什么删掉动态 request-level 控制

之前的 `--dynamic-admission` 调的是 `current_inflight_limit`，也就是同时转发到 upstream 的 request 数。这个粒度和 CONCUR 的目标不一致。

原因：

1. 中期坍塌的根因是长期活跃 agent/session 的 KV working set 太大，不是瞬时 HTTP request 数本身。
2. 一个 session 会连续多轮复用历史 prompt/KV；request-level 控制看不到“这个 request 属于哪个长期 agent”。
3. request 窗口降低以后，只能阻止新的 request 进入，不能决定“哪些 session 继续跑、哪些 session 在 round 边界暂停”。
4. 如果 64 个 session 已经进入系统，request-level window 即使降低，也不会把 active session working set 降到 32，因此不能主动退出 KV thrashing 区间。

所以这次删除了动态 request-level AIMD 路径：

```text
旧：dynamic_admission -> acquire_dynamic_admission() -> active_requests / current_inflight_limit
新：dynamic_admission -> current_session_limit -> active_sessions / paused sessions
```

保留了静态 `--max-inflight-requests`。它现在只是 upstream 安全阀，用来避免 aiohttp/SGLang 被瞬时请求数打爆，不再参与 CONCUR-style 动态控制。

## 新控制流

请求进入 `proxy_handler` 后顺序是：

```text
1. 解析 x-session-id / round 信息
2. acquire_session_admission(session_id, prompt_tokens_estimate)
3. acquire_admission() 只做可选静态 request cap
4. 转发给 direct/router upstream
5. round 完成后 release_session_if_done(...)
```

动态模式下必须有 `x-session-id`。没有 session id 的请求会返回 400，因为 agent-level 控制无法判断它属于哪个智能体。

## Session 状态

每个 session 现在维护这些关键状态：

```text
in_flight                 当前是否有 round 正在 upstream 跑
paused                    是否已经暂停
pause_requested           是否已选为 victim，等当前 round 结束后暂停
prompt_tokens_estimate    近似 KV 压力
last_cache_hit_rate       最近一次 round 的 cache hit
last_seen                 最近进入/完成时间
last_round_done_at        最近 round 完成时间
pause_count               被暂停次数
resume_not_before         最早恢复时间
```

`active_sessions` 的含义变为：没有 `paused=True` 的 session 数。`pause_requested=True` 的 session 还算 active，因为它的当前 round 还没结束，KV 压力尚未真正释放。

## 动态 AIMD 规则

proxy 会尝试从 SGLang Prometheus metrics 读取：

```text
sglang:token_usage
sglang:full_token_usage
sglang:cache_hit_rate
```

默认 metrics 地址：

```text
--direct-url + /metrics
```

可显式覆盖：

```text
--server-metrics-url http://127.0.0.1:8000/metrics
```

动态规则：

```text
if KV usage < U_low:
    W = min(W + additive_step, W_max)

elif KV usage > U_high and cache_hit_rate < H_thresh:
    W = max(floor(W * decrease_factor), W_min)

else:
    hold
```

默认参数已经调成更贴近论文：

```text
U_low = 0.20
U_high = 0.50
H_thresh = 0.20
additive_step = 2
decrease_factor = 0.50
W_min = 16
```

如果 SGLang metrics 暂时不可用，proxy 会退化到旧的 response usage 反馈：

```text
cache_hit_rate 低 或 upstream_ttft_p99 高 -> 降 session window
健康且有 session 排队 -> 加 session window
```

这个 fallback 不如真实 KV usage 精准，但比 request-level 动态窗口更贴近 agent/session 工作集控制。

## Victim 选择

当窗口缩小，例如：

```text
active_sessions = 64
new_session_limit = 32
```

proxy 会选出 `64 - 32 = 32` 个 session 作为 victim。选择规则不是随机，而是按暂停代价排序。

优先暂停：

1. 当前没有 in-flight round 的 session；
2. idle 时间更久的 session；
3. 最近 cache hit 更差的 session；
4. prompt_tokens_estimate 更大的 session；
5. pause_count 更少的 session。

已经 in-flight 的 victim 不会被强杀，只会标记：

```text
pause_requested = true
```

等它当前 round 完成后再变成：

```text
paused = true
```

这样避免打断正在运行的 SGLang 请求，也避免浪费已经做完的 prefill/decode。

## Resume 行为

paused session 的下一轮请求会停在 proxy 的 `acquire_session_admission`。当满足下面条件时恢复：

```text
active_sessions < current_session_limit
now >= resume_not_before
```

恢复时只改 proxy 状态：

```text
paused = false
last_resumed_at = now
```

注意：当前实现没有真正 pin SGLang KV cache。暂停时间太久时，SGLang 内部 LRU 仍可能淘汰它的 KV。这个版本解决的是 agent/session 准入和暂停策略，不是 KV pin/offload。

## 配置变化

下面几个 benchmark admission 配置已经同步：

```text
qiannan-test/benchmark/configs/02_admission_dp2tp8_hicache.json
qiannan-test/benchmark/configs/04_admission_pruned_dp2tp8_hicache.json
qiannan-test/benchmark/configs/07_admission_dp8tp8_hicache.json
qiannan-test/benchmark/configs/08_admission_pruned_dp2tp8_hicache_ratio4_kvfp8.json
```

旧配置里：

```text
dynamic_min_inflight_requests = 64
dynamic_initial_inflight_requests = 128
```

这会导致 session 窗口最低只能降到 64，无法出现 `64 -> 32` 这种乘法减。现在改成：

```text
dynamic_min_inflight_requests = 16
dynamic_initial_inflight_requests = 64
dynamic_max_inflight_requests = 256
```

参数名里仍有 `inflight` 是为了兼容旧脚本；在 `--dynamic-admission` 下这些参数现在表示 active session window。

## 新增参数

```text
--dynamic-min-active-sessions
--dynamic-initial-active-sessions
--dynamic-max-active-sessions
```

这三个是旧 `--dynamic-*-inflight-requests` 的别名。

新增 KV feedback 参数：

```text
--dynamic-low-kv-usage
--dynamic-high-kv-usage
--dynamic-cache-hit-threshold
--server-metrics-url
--server-metrics-interval
```

新增 pause 控制：

```text
--session-pause-min-secs
```

## 日志和观测

`proxy_metrics.jsonl` 新增或调整了这些事件/字段：

```text
dynamic_admission_interval:
  old_session_limit
  new_session_limit
  active_sessions
  paused_sessions
  pause_requested_sessions
  waiting_session_requests
  kv_cache_usage
  server_cache_hit_rate
  pause_victims

session_pause_mark:
  session_id
  state = paused | pause_requested
  prompt_tokens_estimate
  last_cache_hit_rate

session_paused:
  session_id
  current_session_limit

session_resume:
  session_id
  current_session_limit
```

`/health` 也会返回：

```text
session_admission.current_session_limit
session_admission.active_sessions
session_admission.paused_sessions
session_admission.pause_requested_sessions
server_feedback.kv_cache_usage
server_feedback.cache_hit_rate
```

## 当前限制

1. 暂停 session 不等于释放或 pin KV。真正 KV 是否保留仍由 SGLang radix/LRU/HiCache 决定。
2. 如果 SGLang `/metrics` 不可用，控制会退化到 response usage + TTFT fallback，准确性下降。
3. Victim 选择目前用 `prompt_tokens_estimate` 近似 KV 压力，不是 per-session 真实 KV 占用。
4. 这个实现不强杀 in-flight request，只在 round 边界暂停；因此窗口降低后需要等 victim 当前 round 完成，active session 数才会逐步降到新窗口。

## 修改文件

```text
qiannan-test/proxy/main.py
qiannan-test/scripts/proxy_length_aware.sh
qiannan-test/benchmark/run_benchmark.py
qiannan-test/benchmark/configs/02_admission_dp2tp8_hicache.json
qiannan-test/benchmark/configs/04_admission_pruned_dp2tp8_hicache.json
qiannan-test/benchmark/configs/07_admission_dp8tp8_hicache.json
qiannan-test/benchmark/configs/08_admission_pruned_dp2tp8_hicache_ratio4_kvfp8.json
```

## 验证

已执行：

```text
python3 -m py_compile qiannan-test/proxy/main.py qiannan-test/benchmark/run_benchmark.py
python3 -m json.tool <四个 admission config>
```

本地直接运行 `python3 qiannan-test/proxy/main.py --help` 失败，因为当前 Codex 环境没有安装 `aiohttp`。这是运行时依赖缺失，不是语法错误；部署/benchmark 环境需要确保安装 `aiohttp`。
