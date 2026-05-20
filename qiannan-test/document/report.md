## 一、初步策略确定
### 数据集
**T1 sysprompt_pure：测纯 sysprompt+session 双前缀缓存上限。跑不快 = radix/lpm/router 坏了**

**T6 official_50：对齐官方分布，最终对外结果出这个数。修正：分层抽样而非随机抽，否则 sysprompt 比例可能严重失衡（5 种分布 23/21/20/19/16 抽 50 个，随机抽极端情况某种可能 0 个）**

T1_sysprompt_pure

+ 20 sessions × 100 rounds；每轮 1k input / 128 output；system prompt 高复用
+ 目的：测 prefix cache 上限和架构是否保 cache。
+ 预期：
    - A cache hit 非常高
    - B 如果 router 工作，也应很高
    - C 如果不走 router，cache hit 会明显掉

T1 更像“显微镜”，专门看 cache/router 是否正确。

T6_official_50：

+ 目的：对齐官方/真实 workload 分布，是最终对外比较数字。
+ 预期：
    - cache hit 低于 T1
    - latency 更有代表性
    - winner 主要看 T6 的 avg_session_time

T6 更像“最终成绩单”。

### A = 纯 TP8 baseline
+ T1_sysprompt_pure
    - 目的：测 prefix cache 上限。  
    - 预期：
        * cache_hit_rate 很高，约 97%~99%  
        * cached_tokens_total / prompt_tokens_total 很高
        * num_queue_reqs 接近 0
        * failed_rounds = 0
        * actual_output_tokens = expected_output_tokens
+ T6_official_50
    - 目的：对齐官方分布，作为对外报告数字。PLAN 里说 T6 是“最终对外结果出这个数”。
    - 预期：
        * cache_hit_rate 仍应较高，但通常低于 T1
        * avg_session_time 比 T1 更有代表性
        * A 会成为 B/C/D 对比基线

**重点不是 A 自己多快，而是后面比较：**

**B vs A  => ADP 是否带来吞吐收益  
****A vs C  => cache 命中被打烂后会差多少  
****B vs C  => router 是否保住 cache**

**测试commit：4058afd5c7192c11f0545a183c5097e47b9eaeeb**

```plain
# server 命令
bash qiannan-test/scripts/serve_A_tp8.sh
# client
bash client_A_tp8.sh T1
bash client_A_tp8.sh T6
```

```json
{
  "type": "summary",
  "wall_time": 320.14,
  "successful_rounds": 2000,
  "failed_rounds": 0,
  "total_sessions": 20,
  "avg_session_time": 315.8958,
  "total_input_tokens": 2000000,
  "total_prompt_tokens": 102367279,
  "total_cached_tokens": 100266560,
  "total_uncached_prompt_tokens": 2100719,
  "cache_hit_rate": 0.979479,
  "total_output_tokens": 256000,
  "output_throughput_tok_s": 799.7,
  "request_throughput_req_s": 6.25,
  "ttft": {
    "avg": 0.4831,
    "p50": 0.3966,
    "p90": 0.9018,
    "p99": 1.3804
  },
  "round_latency": {
    "avg": 3.1589,
    "p50": 3.0455,
    "p90": 4.6262,
    "p99": 5.5772
  },
  "run_id": "20260519_035714_A_tp8_T1_sysprompt_pure",
  "config": "A_tp8",
  "fixture": "T1_sysprompt_pure",
  "base_url": "http://localhost:8000"
}
```

```plain
{
  "type": "summary",
  "wall_time": 1228.1,
  "successful_rounds": 3435,
  "failed_rounds": 0,
  "total_sessions": 50,
  "avg_session_time": 165.6478,
  "total_input_tokens": 6716234,
  "total_prompt_tokens": 221361630,
  "total_cached_tokens": 214104172,
  "total_uncached_prompt_tokens": 7257458,
  "cache_hit_rate": 0.967214,
  "total_output_tokens": 732918,
  "output_throughput_tok_s": 596.8,
  "request_throughput_req_s": 2.8,
  "ttft": {
    "avg": 0.6569,
    "p50": 0.471,
    "p90": 1.1978,
    "p99": 3.8426
  },
  "round_latency": {
    "avg": 2.4112,
    "p50": 1.9145,
    "p90": 4.6952,
    "p99": 8.7406
  },
  "run_id": "20260519_040757_A_tp8_T6_official_50",
  "config": "A_tp8",
  "fixture": "T6_official_50",
  "base_url": "http://localhost:8000"
}

```

### B：DPA + cache-aware router
```plain
-tp-size 8 --dp-size 8 --enable-dp-attention
  client -> sglang_router :30000 -> SGLang :8000
  router: --dp-aware --policy cache_aware
```

目的：验证推荐生产方案。

DPA 的目标是提高 decode/attention 并行吞吐；router 的目标是让同一个 session 或相似 prefix 尽量打到能命中 cache 的 DP rank，避免 cache 被切碎。

B 用来回答：

DPA 在保住 cache 的情况下，是否比 A 更快？

预期：

+ T1: cache_hit_rate 仍应较高，但可能略低于 A
+ T6: avg_session_time 应该优于 A，或至少吞吐更好num_running_reqs/throughput 可能更高

如果 B 比 A 快，说明 DPA 值得用。  
如果 B cache hit 明显低，说明 router 没保住 cache。  
如果 B cache hit 高但不快，说明 DPA 的额外开销抵消了收益。

commit：be49fdd55b2324e79b23f176c7530fa30bf1c17d

```bash
# 1. 先启动 B 的 SGLang server：
bash serve_B_dpa_tp8_dp8.sh
# 默认监听：SGLang server: http://localhost:8000

#2. 另开一个 shell，启动 cache-aware router：
bash router_B_dp8_cache_aware.sh
# 默认监听：router: http://localhost:30000

# 3. client 要打 router，不要直接打 8000：
bash client_B_dpa_tp8_dp8_router.sh T1
bash qiannan-test/scripts/client_B_dpa_tp8_dp8_router.sh T6

# 这个 client 脚本默认：BASE_URL=http://localhost:30000
```

```json
{
  "type": "summary",
  "wall_time": 893.35,
  "successful_rounds": 2000,
  "failed_rounds": 0,
  "total_sessions": 20,
  "avg_session_time": 618.805,
  "total_input_tokens": 2000000,
  "total_prompt_tokens": 102369311,
  "total_cached_tokens": 99930791,
  "total_uncached_prompt_tokens": 2438520,
  "cache_hit_rate": 0.976179,
  "total_output_tokens": 256000,
  "output_throughput_tok_s": 286.6,
  "request_throughput_req_s": 2.24,
  "ttft": {
    "avg": 4.0549,
    "p50": 0.7009,
    "p90": 2.7302,
    "p99": 28.2833
  },
  "round_latency": {
    "avg": 6.188,
    "p50": 2.7874,
    "p90": 4.709,
    "p99": 31.2907
  },
  "run_id": "20260519_044627_B_dpa_tp8_dp8_router_T1_sysprompt_pure",
  "config": "B_dpa_tp8_dp8_router",
  "fixture": "T1_sysprompt_pure",
  "base_url": "http://localhost:30000"
}

```



```bash
# server
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33 bash serve_B_dpa_tp8_dp8.sh
bash router_B_dp8_cache_aware.sh
# client
CONFIG=B2_dpa_tp8_dp8_router_compensated BASE_URL=http://localhost:30000 bash run_client.sh T1

CONFIG=B2_dpa_tp8_dp8_router_compensated BASE_URL=http://localhost:30000 \
bash qiannan-test/scripts/run_client.sh T6
```

```bash
{
  "type": "summary",
  "wall_time": 681.78,
  "successful_rounds": 2000,
  "failed_rounds": 0,
  "total_sessions": 20,
  "avg_session_time": 456.1046,
  "total_input_tokens": 2000000,
  "total_prompt_tokens": 102406130,
  "total_cached_tokens": 100179802,
  "total_uncached_prompt_tokens": 2226328,
  "cache_hit_rate": 0.97826,
  "total_output_tokens": 256000,
  "output_throughput_tok_s": 375.5,
  "request_throughput_req_s": 2.93,
  "ttft": {
    "avg": 2.8613,
    "p50": 0.5617,
    "p90": 1.8648,
    "p99": 12.1619
  },
  "round_latency": {
    "avg": 4.561,
    "p50": 2.3122,
    "p90": 3.6313,
    "p99": 13.8842
  },
  "run_id": "20260519_050851_B2_dpa_tp8_dp8_router_compensated_T1_sysprompt_pure",
  "config": "B2_dpa_tp8_dp8_router_compensated",
  "fixture": "T1_sysprompt_pure",
  "base_url": "http://localhost:30000"
}

```

**<font style="color:#DF2A3F;">B2-compensated  比 B default 更好</font>**

### 结果不符合预期 开始分析：
尝试把请求打到router和直接打到sglang：实验发现，cache aware会让一个负载非常高，其他都是空的。

```bash
  # B2 + router
  CONFIG=B2_router BASE_URL=http://localhost:30000 bash run_client.sh T1

  # B2 no-router，直接打 SGLang
  CONFIG=B2_no_router BASE_URL=http://localhost:8000 bash run_client.sh T1
```

B2 + router的结果

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779167970299-568e4da8-5aa5-47ee-bfb4-f1aa12985360.png" width="1117" title="" crop="0,0,1,1" id="u807f8b70" class="ne-image">

直接用router 负载严重不均衡。

B2 no-router，直接打 SGLang 的结果。

CONFIG=B2_no_router_clean BASE_URL=[http://localhost:8000](http://localhost:8000) bash run_client.sh T1的结果

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779168226300-d83a74bf-c33d-4bbe-84d3-20d46bee5574.png" width="1104" title="" crop="0,0,1,1" id="u9e85442a" class="ne-image">

```bash
B2 + router：严重负载倾斜
dp_rank=0 num_running_reqs 5
dp_rank=0 num_queue_reqs 14
dp_rank=0 gen_throughput ~506.9
dp_rank=0 cache_hit_rate ~0.989

其他 dp_rank 全是 0

这说明 router 把 T1 基本全压到一个 rank 上。cache hit 很高，但 7 个 rank 空闲，rank 0 还在排队，所以吞吐差。

B2 no-router：负载均衡明显正常

每个 dp_rank num_running_reqs = 2
每个 dp_rank num_queue_reqs = 0
每个 dp_rank gen_throughput ~81-104

8 个 rank 都在工作，瞬时总 gen throughput 粗略相加约：

100.7 + 98.2 + 96.1 + 81.4 + 102.2 + 97.8 + 96.0 + 104.2 ≈ 776 tok/s

这已经接近 A T1 summary 里的 799.7 tok/s。但 no-router 的 cache hit 会下降，比如你贴的：

dp_rank=4 cache_hit_rate ~0.909
dp_rank=2 cache_hit_rate ~0.634

所以结论很明确：

T1 上 B 慢的主因是 router cache_aware 过度追求 cache locality，导致 DP rank 热点集中。不是 B2 参数没生效，也不是 DPA 一定完全不行。

现在可以这样写阶段性结论：

| 配置 | 负载 | cache | 预期性能 |
|---|---|---|---|
| B2 + router | 极度不均衡，单 rank 热点，有 queue | 很高，~99% | 慢 |
| B2 no-router | 8 rank 均衡，无 queue | 降低 | 可能明显更快 |
| A TP8 | 全卡统一服务 | 高 | 当前最稳 |

下一步最关键是等 B2_no_router_clean T1 的完整 client summary。如果它接近 A，就说明 DPA 本身不是主要问题，router 策略需要加 load-balance 约束。
如果它仍明显慢于 A，才说明 DPA 本身还有额外开销。

后续策略上，T1 这种“所有请求同 sysprompt”的 fixture 不适合纯 cache-aware router。需要测 T6，因为 T6 有 5 种 sysprompt 和真实分布，router 可能会
自然分散到多个 rank。若 T6 仍单 rank 热点，就要考虑不用 cache_aware，或者改成“cache-aware + max load / queue penalty”的策略。
```





#### 增加长度感知的优化
commit：c0f3251a8a436d4d459d7464d3157cce781e0425

```bash
# client
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33 bash serve_B_dpa_tp8_dp8.sh

 # router
bash router_B_dp8_cache_aware.sh

# length
bash proxy_length_aware.sh

CONFIG=B2_length_aware_proxy BASE_URL=http://localhost:8888 bash run_client.sh T1

```

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779169342766-52609e18-398f-43a9-ac37-6c3bc8576d66.png" width="1093" title="" crop="0,0,1,1" id="u66f8cd54" class="ne-image">

速度会比没length aware会好一点，但是收益不大。

```bash
LENGTH_THRESHOLD=100000000 bash qiannan-test/scripts/proxy_length_aware.sh
```

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779169506254-68569861-70ac-4840-a149-3206a6e21ddb.png" width="1110" title="" crop="0,0,1,1" id="udb2b5212" class="ne-image">



#### 鉴于T1是特殊case，因此先测一下T6再判断是否需要进一步的优化。
```bash
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33 bash serve_B_dpa_tp8_dp8.sh

CONFIG=B2_no_router_T6 BASE_URL=http://localhost:8000 bash run_client.sh T6
```

```bash
  [   10.0s] sessions 0/50 | rounds 0/3435 | failed 0
  [   20.0s] sessions 0/50 | rounds 31/3435 | failed 0
  [   30.0s] sessions 0/50 | rounds 74/3435 | failed 0
  [   40.0s] sessions 0/50 | rounds 109/3435 | failed 0
  [   50.0s] sessions 0/50 | rounds 127/3435 | failed 0
  [   60.0s] sessions 0/50 | rounds 151/3435 | failed 0
  [   70.0s] sessions 0/50 | rounds 174/3435 | failed 0
  [   80.0s] sessions 0/50 | rounds 186/3435 | failed 0
  [   90.0s] sessions 0/50 | rounds 186/3435 | failed 0
  [  100.0s] sessions 0/50 | rounds 205/3435 | failed 0
  [  110.0s] sessions 0/50 | rounds 238/3435 | failed 0
  [  120.0s] sessions 0/50 | rounds 238/3435 | failed 0
  [  130.0s] sessions 0/50 | rounds 256/3435 | failed 0
  [  140.0s] sessions 0/50 | rounds 281/3435 | failed 0
  [  150.0s] sessions 0/50 | rounds 288/3435 | failed 0
  [  160.0s] sessions 0/50 | rounds 313/3435 | failed 0
```

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779170195986-0105e4e7-3ed1-4597-908e-cb861fba9ed9.png" width="1063" title="" crop="0,0,1,1" id="u4c572750" class="ne-image">



```bash
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33 bash serve_B_dpa_tp8_dp8.sh
bash router_B_dp8_cache_aware.sh

CONFIG=B2_router_T6 BASE_URL=http://localhost:30000 bash run_client.sh T6
```

```bash

  [   10.0s] sessions 0/50 | rounds 0/3435 | failed 0
  [   20.0s] sessions 0/50 | rounds 31/3435 | failed 0
  [   30.0s] sessions 0/50 | rounds 72/3435 | failed 0
  [   40.0s] sessions 0/50 | rounds 109/3435 | failed 0
  [   50.0s] sessions 0/50 | rounds 141/3435 | failed 0
  [   60.0s] sessions 0/50 | rounds 180/3435 | failed 0
  [   70.0s] sessions 0/50 | rounds 180/3435 | failed 0
  [   80.0s] sessions 0/50 | rounds 231/3435 | failed 0
  [   90.0s] sessions 0/50 | rounds 260/3435 | failed 0
  [  100.0s] sessions 0/50 | rounds 262/3435 | failed 0
  [  110.0s] sessions 0/50 | rounds 321/3435 | failed 0
  [  120.0s] sessions 0/50 | rounds 359/3435 | failed 0
  [  130.0s] sessions 0/50 | rounds 394/3435 | failed 0
  [  140.0s] sessions 0/50 | rounds 430/3435 | failed 0
  [  150.0s] sessions 0/50 | rounds 476/3435 | failed 0
  [  160.0s] sessions 0/50 | rounds 507/3435 | failed 0
```

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779170487533-85fd583b-48db-4a6f-bd24-1cf3d43735e8.png" width="1103" title="" crop="0,0,1,1" id="ufd3beaf8" class="ne-image">



```bash
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33 bash serve_B_dpa_tp8_dp8.sh
bash router_B_dp8_cache_aware.sh
LENGTH_THRESHOLD=2048 bash proxy_length_aware.sh
CONFIG=B2_length_aware_2k_T6 BASE_URL=http://localhost:8888 bash run_client.sh T6
```

```bash
  [   10.0s] sessions 0/50 | rounds 0/3435 | failed 0
  [   20.0s] sessions 0/50 | rounds 32/3435 | failed 0
  [   30.0s] sessions 0/50 | rounds 75/3435 | failed 0
  [   40.0s] sessions 0/50 | rounds 113/3435 | failed 0
  [   50.0s] sessions 0/50 | rounds 137/3435 | failed 0
  [   60.0s] sessions 0/50 | rounds 182/3435 | failed 0
  [   70.0s] sessions 0/50 | rounds 182/3435 | failed 0
  [   80.0s] sessions 0/50 | rounds 218/3435 | failed 0
  [   90.0s] sessions 0/50 | rounds 222/3435 | failed 0
  [  100.0s] sessions 0/50 | rounds 271/3435 | failed 0
  [  110.0s] sessions 0/50 | rounds 321/3435 | failed 0
  [  120.0s] sessions 0/50 | rounds 351/3435 | failed 0
  [  130.0s] sessions 0/50 | rounds 385/3435 | failed 0
  [  140.0s] sessions 0/50 | rounds 425/3435 | failed 0
  [  150.0s] sessions 0/50 | rounds 473/3435 | failed 0
  [  160.0s] sessions 0/50 | rounds 504/3435 | failed 0
```

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779170744043-3c2cafe2-34a2-4054-9823-4a950ff0c2d1.png" width="1120" title="" crop="0,0,1,1" id="udecb1dbe" class="ne-image">

### 结论：开proxy  length-aware的adp吞吐更高一点。
纯开cache aware在一些特殊case下导致负载不均，length-aware可以缓解这个问题，且对吞吐没啥影响。



### 回归正题：计算avg时间与A进行比较
```bash
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33 bash serve_B_dpa_tp8_dp8.sh

bash router_B_dp8_cache_aware.sh

LENGTH_THRESHOLD=2048 bash proxy_length_aware.sh

CONFIG=B2_length_aware_2k_T6_clean BASE_URL=http://localhost:8888 bash run_client.sh T6
```

```bash
{
  "type": "summary",
  "wall_time": 1591.39,
  "successful_rounds": 3435,
  "failed_rounds": 0,
  "total_sessions": 50,
  "avg_session_time": 419.6623,
  "total_input_tokens": 6716234,
  "total_prompt_tokens": 220456547,
  "total_cached_tokens": 209493691,
  "total_uncached_prompt_tokens": 10962856,
  "cache_hit_rate": 0.950272,
  "total_output_tokens": 732918,
  "output_throughput_tok_s": 460.6,
  "request_throughput_req_s": 2.16,
  "ttft": {
    "avg": 1.3173,
    "p50": 0.6171,
    "p90": 2.3701,
    "p99": 12.4369
  },
  "round_latency": {
    "avg": 6.1086,
    "p50": 4.0039,
    "p90": 13.7051,
    "p99": 26.5467
  },
  "run_id": "20260519_061558_B2_length_aware_2k_T6_clean_T6_official_50",
  "config": "B2_length_aware_2k_T6_clean",
  "fixture": "T6_official_50",
  "base_url": "http://localhost:8888"
}

```

 B2 length-aware T6 clean 还是比A方案慢很多，可能是因为dp=8 在只有50个session的情况下并行度太大了。所以后面可以测一下dp=2的case



```bash
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33   bash serve_B_dpa_tp8_dp8.sh
bash router_B_dp8_cache_aware.sh
CONFIG=B2_router_T6_clean BASE_URL=http://localhost:30000 bash run_client.sh T6
```

```bash
{
  "type": "summary",
  "wall_time": 1473.81,
  "successful_rounds": 3435,
  "failed_rounds": 0,
  "total_sessions": 50,
  "avg_session_time": 346.2887,
  "total_input_tokens": 6716234,
  "total_prompt_tokens": 220716323,
  "total_cached_tokens": 212491908,
  "total_uncached_prompt_tokens": 8224415,
  "cache_hit_rate": 0.962738,
  "total_output_tokens": 732918,
  "output_throughput_tok_s": 497.3,
  "request_throughput_req_s": 2.33,
  "ttft": {
    "avg": 0.9081,
    "p50": 0.5377,
    "p90": 1.6242,
    "p99": 9.5753
  },
  "round_latency": {
    "avg": 5.0406,
    "p50": 3.3376,
    "p90": 11.9441,
    "p99": 21.3973
  },
  "run_id": "20260519_065101_B2_router_T6_clean_T6_official_50",
  "config": "B2_router_T6_clean",
  "fixture": "T6_official_50",
  "base_url": "http://localhost:30000"
}
```

结论：长度aware的策略不如原本的router cache策略。



### C：DPA 裸跑，无 router
--tp-size 8 --dp-size 8 --enable-dp-attention  
client -> SGLang :8000  
**不走 cache-aware router**

目的：反面教材，证明 router 的价值。

DPA 会把 KV pool 切到多个 DP rank。没有 router 时，同一 session 的多轮请求可能分散到不同 rank，前缀 cache 命中会下降。

C 用来回答：只开 DPA，不做 cache-aware routing，会不会把 cache 打烂？

预期：

+ cache_hit_rate 明显低于 B
+ avg_session_time 可能比 A 还差
+ TTFT/round_latency 变差
+ prompt uncached tokens 增多

对比关系：

B vs C = router 的价值  
A vs C = cache 命中被破坏的代价

```bash
#启动 server：
bash qiannan-test/scripts/serve_C_dpa_tp8_dp8_no_router.sh

#C 的 client 默认打：http://localhost:8000
# 然后另一个 shell 跑 T1：
bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T1
#跑 T6：
bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T6
```





### D：中等 DPA
commit:02c257d6b982c54258ee27adfe3e9c8a7d580369

**--tp-size 8 --dp-size 2 --enable-dp-attention**  
client -> router -> SGLang

目的：找 DP size 的甜点。

B 是比较激进的 DPA；D 是中间方案。DP rank 少一些，cache 被切分得没那么碎，但吞吐并行度也少一些。

D 用来回答：dp_size=8 是否太激进？dp_size=2/中等 DPA 是否更稳？

预期：

+ cache_hit_rate 可能高于 B
+ 吞吐可能低于 B
+ avg_session_time 可能介于 A 和 B 之间
+ 如果 B cache 抖动明显，D 可能反而更好

```bash
bash serve_E_dpa_tp8_dp2.sh
bash router_E_dp2_cache_aware.sh
bash client_E_dpa_tp8_dp2_router.sh T6
```

```bash
{
  "type": "summary",
  "wall_time": 1328.85,
  "successful_rounds": 3435,
  "failed_rounds": 0,
  "total_sessions": 50,
  "avg_session_time": 246.7272,
  "total_input_tokens": 6716234,
  "total_prompt_tokens": 220814423,
  "total_cached_tokens": 213136538,
  "total_uncached_prompt_tokens": 7677885,
  "cache_hit_rate": 0.965229,
  "total_output_tokens": 732918,
  "output_throughput_tok_s": 551.5,
  "request_throughput_req_s": 2.58,
  "ttft": {
    "avg": 0.6287,
    "p50": 0.4622,
    "p90": 1.0111,
    "p99": 4.6201
  },
  "round_latency": {
    "avg": 3.5914,
    "p50": 2.7526,
    "p90": 7.0902,
    "p99": 13.1833
  },
  "run_id": "20260519_072137_E_dpa_tp8_dp2_router_T6_official_50",
  "config": "E_dpa_tp8_dp2_router",
  "fixture": "T6_official_50",
  "base_url": "http://localhost:30000"
}

```

能看到其实还是不太均衡

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779175505250-aa0f1be1-1ecd-4e4c-88d0-a9d5c7723a49.png" width="1097" title="" crop="0,0,1,1" id="u42323fc8" class="ne-image">



测数据集qps，所以取一个小数据集似乎不一定能反映最后的数据？

```bash
• 这两个 QPS 是数据集理论到达 QPS（offered load），不是服务端实际完成 QPS。
  T6_official_50
  平均 QPS: 3.40 req/s
  1s 峰值: 15 req/s
  5s 峰值: 9 req/s
  10s 峰值: 8 req/s
  60s 峰值: 5.75 req/s

  含义：T6 这个 50-session 子集，如果不考虑服务端处理时间，只按数据集里的 wait 来发请求，平均每秒约 3.4 个请求，短时间最高会冲到 15 QPS。

  完整 workload_data.jsonl

  平均 QPS: 32.30 req/s
  1s 峰值: 107 req/s
  5s 峰值: 87 req/s
  10s 峰值: 80 req/s
  60s 峰值: 59.52 req/s

  含义：完整 500-session workload 的压力大约是 T6 的 9.5 倍：

  32.30 / 3.40 ≈ 9.49x

  关键解释

  T6 是 50 sessions，完整 workload 是 500 sessions，所以平均 QPS 大致按 10 倍放大，这和结果吻合：

  T6 avg:      3.4 QPS
  full avg:   32.3 QPS

  这也支持你前面的判断：T6 的 50 session 可能偏小，DPA/ADP 在完整 500 session、更高并发压力下可能更容易发挥作用。

  不过要注意最后一行提示：真实 run_workload.py 是 session 内串行的，下一轮要等上一轮响应完成后再 sleep wait，所以当服务端慢时，实际发出的 QPS 会低于这里的 offered QPS。这里的数字主要用于理解数据集本身的目标压力。
```



### 尝试在完整的数据集上测
commit：a5ea9f596f07c66ca3ebcb7769d5823d9369d655

```bash
# A/TP8 跑法：
bash serve_A_tp8.sh
bash client_A_tp8_full.sh
```

```bash
  [   10.0s] sessions 0/500 | rounds 15/36550 | failed 0
  [   20.0s] sessions 0/500 | rounds 94/36550 | failed 0
  [   30.0s] sessions 0/500 | rounds 203/36550 | failed 0
  [   40.0s] sessions 0/500 | rounds 306/36550 | failed 0
  [   50.0s] sessions 0/500 | rounds 408/36550 | failed 0
  [   60.0s] sessions 0/500 | rounds 514/36550 | failed 0
  [   70.0s] sessions 0/500 | rounds 605/36550 | failed 0
  [   80.0s] sessions 0/500 | rounds 687/36550 | failed 0
  [   90.0s] sessions 0/500 | rounds 754/36550 | failed 0
  [  100.0s] sessions 0/500 | rounds 806/36550 | failed 0
  [  110.0s] sessions 0/500 | rounds 887/36550 | failed 0
  [  120.0s] sessions 0/500 | rounds 975/36550 | failed 0
  [  130.0s] sessions 0/500 | rounds 1045/36550 | failed 0
  [  140.0s] sessions 0/500 | rounds 1126/36550 | failed 0
  [  150.0s] sessions 0/500 | rounds 1201/36550 | failed 0
  [  160.0s] sessions 0/500 | rounds 1286/36550 | failed 0
  [  170.0s] sessions 0/500 | rounds 1367/36550 | failed 0
  [  180.0s] sessions 0/500 | rounds 1455/36550 | failed 0
  [  190.0s] sessions 0/500 | rounds 1543/36550 | failed 0
  [  200.0s] sessions 0/500 | rounds 1619/36550 | failed 0
```



```bash
# B2-compensated 跑法：
CHUNKED_PREFILL_SIZE=65536 SCHEDULE_CONSERVATIVENESS=3.33 bash serve_B_dpa_tp8_dp8.sh

bash router_B_dp8_cache_aware.sh

bash client_B2_dpa_tp8_dp8_router_compensated_full.sh
```

```bash
  [   10.0s] sessions 0/500 | rounds 1/36550 | failed 0
  [   20.0s] sessions 0/500 | rounds 27/36550 | failed 0
  [   30.0s] sessions 0/500 | rounds 81/36550 | failed 0
  [   40.0s] sessions 0/500 | rounds 183/36550 | failed 0
  [   50.0s] sessions 0/500 | rounds 254/36550 | failed 0
  [   60.0s] sessions 0/500 | rounds 301/36550 | failed 0
  [   70.0s] sessions 0/500 | rounds 383/36550 | failed 0
  [   80.0s] sessions 0/500 | rounds 471/36550 | failed 0
  [   90.0s] sessions 0/500 | rounds 508/36550 | failed 0
  [  100.0s] sessions 0/500 | rounds 565/36550 | failed 0
  [  110.0s] sessions 0/500 | rounds 568/36550 | failed 0
  [  120.0s] sessions 0/500 | rounds 636/36550 | failed 0
  [  130.0s] sessions 0/500 | rounds 692/36550 | failed 0
  [  140.0s] sessions 0/500 | rounds 720/36550 | failed 0
  [  150.0s] sessions 0/500 | rounds 789/36550 | failed 0
  [  160.0s] sessions 0/500 | rounds 813/36550 | failed 0
  [  170.0s] sessions 0/500 | rounds 893/36550 | failed 0
  [  180.0s] sessions 0/500 | rounds 951/36550 | failed 0
  [  190.0s] sessions 0/500 | rounds 974/36550 | failed 0
  [  200.0s] sessions 0/500 | rounds 1032/36550 | failed 0
  [  210.0s] sessions 0/500 | rounds 1062/36550 | failed 0
  [  220.0s] sessions 0/500 | rounds 1145/36550 | failed 0
  [  230.0s] sessions 0/500 | rounds 1218/36550 | failed 0
  [  240.0s] sessions 0/500 | rounds 1285/36550 | failed 0
  [  250.0s] sessions 0/500 | rounds 1340/36550 | failed 0
  [  260.0s] sessions 0/500 | rounds 1379/36550 | failed 0
  [  270.0s] sessions 0/500 | rounds 1442/36550 | failed 0
  [  280.0s] sessions 0/500 | rounds 1512/36550 | failed 0
  [  290.0s] sessions 0/500 | rounds 1563/36550 | failed 0
```

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779177119372-0d87a649-dc09-4d7d-8324-2e428b1a4f0c.png" width="1102" title="" crop="0,0,1,1" id="u4230d0cc" class="ne-image">



```bash
bash serve_E_dpa_tp8_dp2.sh

bash router_E_dp2_cache_aware.sh

bash client_E_dpa_tp8_dp2_router_full.sh
```

```bash
  [   10.0s] sessions 0/500 | rounds 35/36550 | failed 0
  [   20.0s] sessions 0/500 | rounds 134/36550 | failed 0
  [   30.0s] sessions 0/500 | rounds 230/36550 | failed 0
  [   40.0s] sessions 0/500 | rounds 325/36550 | failed 0
  [   50.0s] sessions 0/500 | rounds 380/36550 | failed 0
  [   60.0s] sessions 0/500 | rounds 481/36550 | failed 0
  [   70.0s] sessions 0/500 | rounds 540/36550 | failed 0
  [   80.0s] sessions 0/500 | rounds 587/36550 | failed 0
  [   90.0s] sessions 0/500 | rounds 676/36550 | failed 0
  [  100.0s] sessions 0/500 | rounds 757/36550 | failed 0
  [  110.0s] sessions 0/500 | rounds 814/36550 | failed 0
  [  120.0s] sessions 0/500 | rounds 899/36550 | failed 0
  [  130.0s] sessions 0/500 | rounds 986/36550 | failed 0
  [  140.0s] sessions 0/500 | rounds 1052/36550 | failed 0
  [  150.0s] sessions 0/500 | rounds 1134/36550 | failed 0
  [  160.0s] sessions 0/500 | rounds 1205/36550 | failed 0
  [  170.0s] sessions 0/500 | rounds 1273/36550 | failed 0
  [  180.0s] sessions 0/500 | rounds 1339/36550 | failed 0
  [  190.0s] sessions 0/500 | rounds 1434/36550 | failed 0
  [  200.0s] sessions 0/500 | rounds 1527/36550 | failed 0
```

**<font style="color:#DF2A3F;">结论：dp2 tp8 不一定比tp8更好。</font>****<font style="color:#DF2A3F;">❌</font>**

**<font style="color:#DF2A3F;">因为上面开了投机解码，所以导致：默认最大运行请求数会改成 48：如果没显式设置 --max-running-requests。【</font>****<font style="color:#DF2A3F;">🆘</font>****<font style="color:#DF2A3F;">被坑惨了】</font>**

**<font style="color:#DF2A3F;"></font>**

### <font style="color:#DF2A3F;">关掉投机解码</font>
```bash
SPECULATIVE_ALGO=NONE bash serve_E_dpa_tp8_dp2.sh
bash router_E_dp2_cache_aware.sh

bash client_E_dpa_tp8_dp2_router_full.sh
```

```bash
  [   10.0s] sessions 0/500 | rounds 2/36550 | failed 0
  [   20.0s] sessions 0/500 | rounds 84/36550 | failed 0
  [   30.0s] sessions 0/500 | rounds 210/36550 | failed 0
  [   40.0s] sessions 0/500 | rounds 357/36550 | failed 0
  [   50.0s] sessions 0/500 | rounds 474/36550 | failed 0
  [   60.0s] sessions 0/500 | rounds 573/36550 | failed 0
  [   70.0s] sessions 0/500 | rounds 680/36550 | failed 0
  [   80.0s] sessions 0/500 | rounds 827/36550 | failed 0
  [   90.0s] sessions 0/500 | rounds 981/36550 | failed 0
  [  100.0s] sessions 0/500 | rounds 1101/36550 | failed 0
  [  110.0s] sessions 0/500 | rounds 1235/36550 | failed 0
  [  120.0s] sessions 0/500 | rounds 1333/36550 | failed 0
  [  130.0s] sessions 0/500 | rounds 1393/36550 | failed 0
  [  140.0s] sessions 0/500 | rounds 1490/36550 | failed 0
  [  150.0s] sessions 0/500 | rounds 1547/36550 | failed 0
  [  160.0s] sessions 0/500 | rounds 1625/36550 | failed 0
  [  170.0s] sessions 0/500 | rounds 1698/36550 | failed 0
  [  180.0s] sessions 0/500 | rounds 1768/36550 | failed 0
  [  190.0s] sessions 0/500 | rounds 1835/36550 | failed 0
  [  200.0s] sessions 0/500 | rounds 1897/36550 | failed 0
  [  210.0s] sessions 0/500 | rounds 1962/36550 | failed 0
  [  220.0s] sessions 0/500 | rounds 2031/36550 | failed 0
  [  230.0s] sessions 0/500 | rounds 2102/36550 | failed 0
  [  240.0s] sessions 0/500 | rounds 2160/36550 | failed 0
  [  250.0s] sessions 0/500 | rounds 2228/36550 | failed 0
  [  260.0s] sessions 0/500 | rounds 2289/36550 | failed 0
  [  270.0s] sessions 0/500 | rounds 2344/36550 | failed 0
  [  280.0s] sessions 0/500 | rounds 2408/36550 | failed 0
  [  290.0s] sessions 0/500 | rounds 2467/36550 | failed 0
  [  300.0s] sessions 0/500 | rounds 2521/36550 | failed 0
  [  310.0s] sessions 0/500 | rounds 2577/36550 | failed 0
  [  320.0s] sessions 0/500 | rounds 2624/36550 | failed 0
  [  330.0s] sessions 0/500 | rounds 2691/36550 | failed 0
  [  340.0s] sessions 0/500 | rounds 2741/36550 | failed 0
  [  350.0s] sessions 0/500 | rounds 2798/36550 | failed 0
  [  360.0s] sessions 0/500 | rounds 2857/36550 | failed 0
  [  370.0s] sessions 0/500 | rounds 2905/36550 | failed 0
  [  380.0s] sessions 0/500 | rounds 2942/36550 | failed 0
  [  390.0s] sessions 0/500 | rounds 3003/36550 | failed 0
  [  400.0s] sessions 0/500 | rounds 3053/36550 | failed 0
  [  410.0s] sessions 0/500 | rounds 3116/36550 | failed 0
  [  420.0s] sessions 0/500 | rounds 3166/36550 | failed 0
  [  430.0s] sessions 0/500 | rounds 3226/36550 | failed 0
  [  440.0s] sessions 0/500 | rounds 3276/36550 | failed 0
  [  450.0s] sessions 0/500 | rounds 3324/36550 | failed 0
  [  460.0s] sessions 0/500 | rounds 3369/36550 | failed 0
  [  470.0s] sessions 0/500 | rounds 3414/36550 | failed 0
  [  480.0s] sessions 0/500 | rounds 3470/36550 | failed 0
  [  490.0s] sessions 0/500 | rounds 3514/36550 | failed 0
  [  500.0s] sessions 0/500 | rounds 3562/36550 | failed 0
  [  510.0s] sessions 0/500 | rounds 3592/36550 | failed 0
  [  520.0s] sessions 0/500 | rounds 3636/36550 | failed 0
  [  530.0s] sessions 0/500 | rounds 3667/36550 | failed 0
  [  540.0s] sessions 0/500 | rounds 3699/36550 | failed 0
  [  550.0s] sessions 0/500 | rounds 3734/36550 | failed 0
  [  560.0s] sessions 0/500 | rounds 3762/36550 | failed 0
  [  570.0s] sessions 0/500 | rounds 3807/36550 | failed 0
  [  580.0s] sessions 0/500 | rounds 3835/36550 | failed 0
  [  590.0s] sessions 0/500 | rounds 3875/36550 | failed 0
  [  600.0s] sessions 0/500 | rounds 3915/36550 | failed 0
  [  610.0s] sessions 0/500 | rounds 3961/36550 | failed 0
  [  620.0s] sessions 0/500 | rounds 3998/36550 | failed 0
  [  630.0s] sessions 0/500 | rounds 4030/36550 | failed 0
  [  640.0s] sessions 0/500 | rounds 4069/36550 | failed 0
  [  650.0s] sessions 0/500 | rounds 4117/36550 | failed 0
  [  660.0s] sessions 0/500 | rounds 4145/36550 | failed 0
  [  670.0s] sessions 0/500 | rounds 4183/36550 | failed 0
  [  680.1s] sessions 0/500 | rounds 4218/36550 | failed 0
  [  690.1s] sessions 0/500 | rounds 4248/36550 | failed 0
  [  700.1s] sessions 0/500 | rounds 4296/36550 | failed 0
  [  710.1s] sessions 0/500 | rounds 4324/36550 | failed 0
  [  720.1s] sessions 0/500 | rounds 4363/36550 | failed 0
  [  730.1s] sessions 0/500 | rounds 4389/36550 | failed 0
  [  740.1s] sessions 0/500 | rounds 4425/36550 | failed 0
  [  750.1s] sessions 0/500 | rounds 4445/36550 | failed 0
  [  760.1s] sessions 0/500 | rounds 4481/36550 | failed 0

```

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779180852337-8b0a10e2-edb5-4abe-af8a-cf60ba0d3e7b.png" width="1108" title="" crop="0,0,1,1" id="ue8629a6f" class="ne-image">



```bash
SPECULATIVE_ALGO=NONE MAX_RUNNING_REQUESTS=640 bash serve_F_dpa_tp8_dp4.sh


# shell 2
bash router_F_dp4_cache_aware.sh

# shell 3: full dataset
SUMMARY_INTERVAL=60 bash client_F_dpa_tp8_dp4_router_full.sh
```

```bash
{"wall_time": 60.0, "successful_rounds": 513, "failed_rounds": 0, "total_sessions": 394, "avg_session_time": 25.3239, "total_input_tokens": 1104748, "total_prompt_tokens": 1325228, "total_cached_tokens": 720046, "total_uncached_prompt_tokens": 605182, "cache_hit_rate": 0.543337, "total_output_tokens": 75912, "output_throughput_tok_s": 1265.2, "request_throughput_req_s": 8.55, "ttft": {"avg": 1.1465, "p50": 0.4995, "p90": 3.2829, "p99": 10.8347}, "round_latency": {"avg": 19.4495, "p50": 17.0223, "p90": 32.3915, "p99": 48.2108}, "type": "interval_summary", "interval_index": 1, "interval_secs": 60.0, "elapsed_secs": 60.0, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.014036}
{"wall_time": 120.01, "successful_rounds": 1115, "failed_rounds": 0, "total_sessions": 478, "avg_session_time": 66.2302, "total_input_tokens": 2415978, "total_prompt_tokens": 4482607, "total_cached_tokens": 2220180, "total_uncached_prompt_tokens": 2262427, "cache_hit_rate": 0.495288, "total_output_tokens": 196720, "output_throughput_tok_s": 1639.2, "request_throughput_req_s": 9.29, "ttft": {"avg": 2.0752, "p50": 0.5967, "p90": 4.9926, "p99": 16.216}, "round_latency": {"avg": 28.3929, "p50": 23.1708, "p90": 54.1773, "p99": 84.1228}, "type": "interval_summary", "interval_index": 2, "interval_secs": 60.0, "elapsed_secs": 120.01, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.030506}
{"wall_time": 180.01, "successful_rounds": 1557, "failed_rounds": 0, "total_sessions": 486, "avg_session_time": 104.4542, "total_input_tokens": 3243953, "total_prompt_tokens": 7984441, "total_cached_tokens": 4334748, "total_uncached_prompt_tokens": 3649693, "cache_hit_rate": 0.542899, "total_output_tokens": 286752, "output_throughput_tok_s": 1593.0, "request_throughput_req_s": 8.65, "ttft": {"avg": 2.5121, "p50": 0.6573, "p90": 5.2532, "p99": 41.6803}, "round_latency": {"avg": 32.6042, "p50": 26.663, "p90": 61.4731, "p99": 111.6089}, "type": "interval_summary", "interval_index": 3, "interval_secs": 60.0, "elapsed_secs": 180.01, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.042599}
{"wall_time": 240.02, "successful_rounds": 1962, "failed_rounds": 0, "total_sessions": 491, "avg_session_time": 143.0429, "total_input_tokens": 3847685, "total_prompt_tokens": 11620178, "total_cached_tokens": 6592511, "total_uncached_prompt_tokens": 5027667, "cache_hit_rate": 0.567333, "total_output_tokens": 372148, "output_throughput_tok_s": 1550.5, "request_throughput_req_s": 8.17, "ttft": {"avg": 2.5772, "p50": 0.7199, "p90": 5.4322, "p99": 36.6883}, "round_latency": {"avg": 35.7972, "p50": 28.8988, "p90": 68.4886, "p99": 130.3889}, "type": "interval_summary", "interval_index": 4, "interval_secs": 60.0, "elapsed_secs": 240.02, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.05368}

```

```bash
SPECULATIVE_ALGO=NONE MAX_RUNNING_REQUESTS=640  bash serve_E_dpa_tp8_dp2.sh
bash router_E_dp2_cache_aware.sh

SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```

```bash
{"wall_time": 60.0, "successful_rounds": 553, "failed_rounds": 0, "total_sessions": 410, "avg_session_time": 24.0179, "total_input_tokens": 1152465, "total_prompt_tokens": 1402130, "total_cached_tokens": 776976, "total_uncached_prompt_tokens": 625154, "cache_hit_rate": 0.55414, "total_output_tokens": 81809, "output_throughput_tok_s": 1363.4, "request_throughput_req_s": 9.22, "ttft": {"avg": 0.8072, "p50": 0.3943, "p90": 2.3089, "p99": 5.9852}, "round_latency": {"avg": 17.8071, "p50": 15.4339, "p90": 30.3629, "p99": 45.5633}, "type": "interval_summary", "interval_index": 1, "interval_secs": 60.0, "elapsed_secs": 60.0, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.01513}
{"wall_time": 120.0, "successful_rounds": 1284, "failed_rounds": 0, "total_sessions": 492, "avg_session_time": 66.4805, "total_input_tokens": 2770820, "total_prompt_tokens": 5478763, "total_cached_tokens": 2951128, "total_uncached_prompt_tokens": 2527635, "cache_hit_rate": 0.538649, "total_output_tokens": 228446, "output_throughput_tok_s": 1903.6, "request_throughput_req_s": 10.7, "ttft": {"avg": 0.9891, "p50": 0.4696, "p90": 2.739, "p99": 7.138}, "round_latency": {"avg": 25.4738, "p50": 20.9691, "p90": 48.1289, "p99": 75.6557}, "type": "interval_summary", "interval_index": 2, "interval_secs": 60.0, "elapsed_secs": 120.0, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.03513}
{"wall_time": 180.01, "successful_rounds": 1774, "failed_rounds": 0, "total_sessions": 499, "avg_session_time": 104.6125, "total_input_tokens": 3517755, "total_prompt_tokens": 9459069, "total_cached_tokens": 5450822, "total_uncached_prompt_tokens": 4008247, "cache_hit_rate": 0.576254, "total_output_tokens": 330453, "output_throughput_tok_s": 1835.8, "request_throughput_req_s": 9.86, "ttft": {"avg": 1.0628, "p50": 0.5161, "p90": 2.8744, "p99": 7.2212}, "round_latency": {"avg": 29.426, "p50": 24.0643, "p90": 55.0572, "p99": 97.182}, "type": "interval_summary", "interval_index": 3, "interval_secs": 60.0, "elapsed_secs": 180.01, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.048536}
{"wall_time": 240.01, "successful_rounds": 2213, "failed_rounds": 0, "total_sessions": 500, "avg_session_time": 149.8172, "total_input_tokens": 4162298, "total_prompt_tokens": 13112008, "total_cached_tokens": 7543526, "total_uncached_prompt_tokens": 5568482, "cache_hit_rate": 0.575314, "total_output_tokens": 423948, "output_throughput_tok_s": 1766.4, "request_throughput_req_s": 9.22, "ttft": {"avg": 1.2256, "p50": 0.6035, "p90": 3.2122, "p99": 8.3236}, "round_latency": {"avg": 33.8493, "p50": 27.1626, "p90": 63.723, "p99": 124.844}, "type": "interval_summary", "interval_index": 4, "interval_secs": 60.0, "elapsed_secs": 240.01, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.060547}

```



出问题了！！！后期吞吐严重下降！

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779186295879-a18f3f9d-6d33-40ed-944a-280abe3b09c4.png" width="1076" title="" crop="0,0,1,1" id="u85c595fc" class="ne-image">

  你这组数据里，1500s 左右是明显拐点：ttft p99 从 153s 跳到 632s，同时 interval cache hit 从大约 50% 掉到 13%。后面 p90/p99 越来越差，就是这个问题继续累积。

```bash
  1. 看 1400s-1500s 附近 server log 有没有 KV cache pool is full. Retract requests、#retracted_reqs 上升。
  2. 按 60s 增量算 cache hit，不要只看累计值；你这里后段命中已经塌了。
  3. 确认同一个 session 是否稳定路由到同一个 DP rank。
  4. 拉 per-DP 的 input throughput、gen throughput、decode_sum_seq_lens、num_retracted_reqs，确认 dp1 是否真的卡住。
  5. 降低 loaded_sessions/并发，或限制 session history 长度；否则 500 个长会话的 working set 会把 prefix cache 顶爆。
  6. 如果显存允许，增大 KV 容量，比如调高 --mem-fraction-static；同时重新评估 chunked_prefill_size 和 max running/prefill 相关限制。

  一句话结论：后面不是“没请求所以 token usage 降”，而是请求都堆在队列里，活跃 KV 变少；真正瓶颈转成了长上下文 uncached prefill + cache
  eviction/DP 不均衡，导致 running batch 无法保持高 decode 利用率。
```

## 二、微调其他按钮
```bash
SPECULATIVE_ALGO=NONE MAX_RUNNING_REQUESTS=640 ENABLE_DP_LM_HEAD=1 bash serve_E_dpa_tp8_dp2.sh

bash router_E_dp2_cache_aware.sh

SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```

```bash
{"wall_time": 60.0, "successful_rounds": 569, "failed_rounds": 0, "total_sessions": 411, "avg_session_time": 23.567, "total_input_tokens": 1171775, "total_prompt_tokens": 1461229, "total_cached_tokens": 825133, "total_uncached_prompt_tokens": 636096, "cache_hit_rate": 0.564684, "total_output_tokens": 85817, "output_throughput_tok_s": 1430.2, "request_throughput_req_s": 9.48, "ttft": {"avg": 0.8523, "p50": 0.446, "p90": 2.1232, "p99": 6.0536}, "round_latency": {"avg": 17.0229, "p50": 15.0418, "p90": 28.7328, "p99": 43.5309}, "type": "interval_summary", "interval_index": 1, "interval_secs": 60.0, "elapsed_secs": 60.0, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.015568}
{"wall_time": 120.01, "successful_rounds": 1313, "failed_rounds": 0, "total_sessions": 495, "avg_session_time": 65.0483, "total_input_tokens": 2813857, "total_prompt_tokens": 5985410, "total_cached_tokens": 3596375, "total_uncached_prompt_tokens": 2389035, "cache_hit_rate": 0.600857, "total_output_tokens": 237448, "output_throughput_tok_s": 1978.6, "request_throughput_req_s": 10.94, "ttft": {"avg": 1.8177, "p50": 0.5228, "p90": 3.1803, "p99": 34.9999}, "round_latency": {"avg": 24.5231, "p50": 19.8479, "p90": 46.348, "p99": 76.9425}, "type": "interval_summary", "interval_index": 2, "interval_secs": 60.0, "elapsed_secs": 120.01, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.035923}

```

差不多 没啥变化



```bash
SPECULATIVE_ALGO=NONE MAX_RUNNING_REQUESTS=640 ENABLE_DP_LM_HEAD=1 MEM_FRACTION_STATIC=0.9 bash serve_E_dpa_tp8_dp2.sh

bash router_E_dp2_cache_aware.sh

SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```

```bash
{"wall_time": 60.0, "successful_rounds": 569, "failed_rounds": 0, "total_sessions": 411, "avg_session_time": 23.567, "total_input_tokens": 1171775, "total_prompt_tokens": 1461229, "total_cached_tokens": 825133, "total_uncached_prompt_tokens": 636096, "cache_hit_rate": 0.564684, "total_output_tokens": 85817, "output_throughput_tok_s": 1430.2, "request_throughput_req_s": 9.48, "ttft": {"avg": 0.8523, "p50": 0.446, "p90": 2.1232, "p99": 6.0536}, "round_latency": {"avg": 17.0229, "p50": 15.0418, "p90": 28.7328, "p99": 43.5309}, "type": "interval_summary", "interval_index": 1, "interval_secs": 60.0, "elapsed_secs": 60.0, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.015568}
{"wall_time": 120.01, "successful_rounds": 1313, "failed_rounds": 0, "total_sessions": 495, "avg_session_time": 65.0483, "total_input_tokens": 2813857, "total_prompt_tokens": 5985410, "total_cached_tokens": 3596375, "total_uncached_prompt_tokens": 2389035, "cache_hit_rate": 0.600857, "total_output_tokens": 237448, "output_throughput_tok_s": 1978.6, "request_throughput_req_s": 10.94, "ttft": {"avg": 1.8177, "p50": 0.5228, "p90": 3.1803, "p99": 34.9999}, "round_latency": {"avg": 24.5231, "p50": 19.8479, "p90": 46.348, "p99": 76.9425}, "type": "interval_summary", "interval_index": 2, "interval_secs": 60.0, "elapsed_secs": 120.01, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.035923}
{"wall_time": 180.01, "successful_rounds": 1790, "failed_rounds": 0, "total_sessions": 498, "avg_session_time": 102.5272, "total_input_tokens": 3529112, "total_prompt_tokens": 9815722, "total_cached_tokens": 5985114, "total_uncached_prompt_tokens": 3830608, "cache_hit_rate": 0.609748, "total_output_tokens": 336670, "output_throughput_tok_s": 1870.3, "request_throughput_req_s": 9.94, "ttft": {"avg": 1.8343, "p50": 0.5596, "p90": 3.615, "p99": 30.577}, "round_latency": {"avg": 28.5243, "p50": 23.1559, "p90": 54.346, "p99": 98.2842}, "type": "interval_summary", "interval_index": 3, "interval_secs": 60.0, "elapsed_secs": 180.01, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.048974}
{"wall_time": 240.02, "successful_rounds": 2190, "failed_rounds": 0, "total_sessions": 500, "avg_session_time": 146.3035, "total_input_tokens": 4249739, "total_prompt_tokens": 13293553, "total_cached_tokens": 7913386, "total_uncached_prompt_tokens": 5380167, "cache_hit_rate": 0.59528, "total_output_tokens": 421381, "output_throughput_tok_s": 1755.6, "request_throughput_req_s": 9.12, "ttft": {"avg": 2.2403, "p50": 0.6462, "p90": 4.6184, "p99": 32.6435}, "round_latency": {"avg": 33.4026, "p50": 26.5153, "p90": 63.9829, "p99": 125.1974}, "type": "interval_summary", "interval_index": 4, "interval_secs": 60.0, "elapsed_secs": 240.02, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.059918}
{"wall_time": 300.03, "successful_rounds": 2530, "failed_rounds": 0, "total_sessions": 500, "avg_session_time": 186.8738, "total_input_tokens": 5001722, "total_prompt_tokens": 17098424, "total_cached_tokens": 9358490, "total_uncached_prompt_tokens": 7739934, "cache_hit_rate": 0.547331, "total_output_tokens": 490270, "output_throughput_tok_s": 1634.1, "request_throughput_req_s": 8.43, "ttft": {"avg": 2.6326, "p50": 0.7505, "p90": 5.2405, "p99": 39.3178}, "round_latency": {"avg": 36.9316, "p50": 28.5127, "p90": 72.2124, "p99": 140.2325}, "type": "interval_summary", "interval_index": 5, "interval_secs": 60.0, "elapsed_secs": 300.03, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.06922}
{"wall_time": 360.03, "successful_rounds": 2872, "failed_rounds": 0, "total_sessions": 500, "avg_session_time": 235.4152, "total_input_tokens": 5943820, "total_prompt_tokens": 21690619, "total_cached_tokens": 11683497, "total_uncached_prompt_tokens": 10007122, "cache_hit_rate": 0.538643, "total_output_tokens": 570593, "output_throughput_tok_s": 1584.8, "request_throughput_req_s": 7.98, "ttft": {"avg": 2.8149, "p50": 0.8355, "p90": 5.6446, "p99": 40.0057}, "round_latency": {"avg": 40.9845, "p50": 30.9771, "p90": 82.4854, "p99": 174.0918}, "type": "interval_summary", "interval_index": 6, "interval_secs": 60.0, "elapsed_secs": 360.03, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.078577}
{"wall_time": 420.04, "successful_rounds": 3197, "failed_rounds": 0, "total_sessions": 500, "avg_session_time": 281.0372, "total_input_tokens": 6566133, "total_prompt_tokens": 26384050, "total_cached_tokens": 14053168, "total_uncached_prompt_tokens": 12330882, "cache_hit_rate": 0.532639, "total_output_tokens": 644435, "output_throughput_tok_s": 1534.2, "request_throughput_req_s": 7.61, "ttft": {"avg": 3.0864, "p50": 0.9008, "p90": 6.1654, "p99": 44.3218}, "round_latency": {"avg": 43.9533, "p50": 32.9629, "p90": 88.1896, "p99": 183.7812}, "type": "interval_summary", "interval_index": 7, "interval_secs": 60.0, "elapsed_secs": 420.04, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.087469}
{"wall_time": 480.35, "successful_rounds": 3495, "failed_rounds": 0, "total_sessions": 500, "avg_session_time": 326.7302, "total_input_tokens": 7326850, "total_prompt_tokens": 31115525, "total_cached_tokens": 16375570, "total_uncached_prompt_tokens": 14739955, "cache_hit_rate": 0.526283, "total_output_tokens": 713473, "output_throughput_tok_s": 1485.3, "request_throughput_req_s": 7.28, "ttft": {"avg": 3.3435, "p50": 0.9798, "p90": 6.407, "p99": 44.5469}, "round_latency": {"avg": 46.7425, "p50": 34.6479, "p90": 94.5167, "p99": 198.687}, "type": "interval_summary", "interval_index": 8, "interval_secs": 60.0, "elapsed_secs": 480.35, "loaded_sessions": 500, "completed_sessions": 0, "planned_rounds": 36550, "completion_ratio": 0.095622}

```



```bash
SPECULATIVE_ALGO=NONE MAX_RUNNING_REQUESTS=640 MEM_FRACTION_STATIC=0.95 ENABLE_DP_LM_HEAD=1 bash serve_E_dpa_tp8_dp2.sh

bash router_E_dp2_cache_aware.sh

SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```

mem0.95 OOM



```bash
SPECULATIVE_ALGO=NONE MAX_RUNNING_REQUESTS=640 MEM_FRACTION_STATIC=0.92 ENABLE_DP_LM_HEAD=1 bash serve_E_dpa_tp8_dp2.sh

bash router_E_dp2_cache_aware.sh

SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```

```bash

```

比mem0.9更好



hierarchical-cache

```bash
SPECULATIVE_ALGO=NONE MEM_FRACTION_STATIC=0.92 ENABLE_HICACHE=1 HICACHE_RATIO=2 HICACHE_IO_BACKEND=kernel HICACHE_STORAGE_PREFETCH_POLICY=wait_complete HICACHE_WRITE_POLICY=write_through bash serve_E_dpa_tp8_dp2.sh

bash router_E_dp2_cache_aware.sh
SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```

ok！✅

### sglang bugfix
commit：5f990f7e86221a89469a83f87c93be8acc5815d7 & 5645f139a2e47e05b2716e04608445442ca96dcf

```bash
这个 bug 的本质是 CUDA graph capture 的 batch size 被 SGLang 向上 padding 了，但 FlashInfer backend 的 metadata buffer 仍按未 padding
的 request pool size 分配。

你的失败日志里最关键的是：

Target sizes: [210]. Tensor sizes: [212]

这不是 OOM，而是 shape mismatch。

实际发生链路是：

1. 启动时 SGLang 先算实际可支持的 active request slots。
   你虽然设了：

   MAX_RUNNING_REQUESTS=512

   但 hybrid GDN/Mamba + KV/Mamba cache + HiCache 之后，实际 device request pool 不是 256 per DP，而是 210。

2. DP attention 下 CUDA graph batch size 要按 attention TP 对齐。
   你的 TP=8、DP=2，effective attention TP 是 4，所以：

   ceil(210 / 4) * 4 = 212

   因此 CUDA graph capture 选择了 bs=212。

3. FlashInfer backend 初始化 metadata buffer 时只看了实际 pool size。
   它的 kv_indptr 被分配成：

   210 + 1

   但 capture 时要写：

   kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)

   此时 bs=212，右侧是 212 个元素；左侧由于 buffer 只有 211 长度，实际切片只能容纳 210 个元素，于是报：

   Target sizes: [210]. Tensor sizes: [212]

之前 253 -> 256 也是同一类：

实际 pool = 253
对齐后 bs = 256
FlashInfer buffer = 253 + 1
=> crash

为什么不加 HiCache 可以跑？

因为不加 HiCache时，实际 request pool 很可能刚好是 320 或 256 这类 4 的倍数：

320 % 4 = 0
256 % 4 = 0

不需要 padding，FlashInfer buffer size 和 capture bs 一致，所以没暴露 bug。

为什么加 HiCache 后出问题？

HiCache 本身不是直接导致 shape mismatch。它改变了 cache stack 和内存分配结果，使实际 GPU active request slots 变成了 210。210 不是 4
的倍数，于是 CUDA graph 补到 212，FlashInfer 旧逻辑没处理这个补齐。

第一版修复为什么还报？

第一版只扩了：

self.kv_indptr
self.kv_last_page_len

但真正执行报错的 FlashInferIndicesUpdaterDecode 在初始化时已经保存了旧引用：

self.kv_indptr = attn_backend.kv_indptr

所以 backend 字段换成新 buffer 后，updater 还在用旧的 210 buffer。第二版修复把 updater 的引用也同步了，所以才生效。

最终修复逻辑是：

- 如果 CUDA graph max_bs 大于原始 request pool size，就扩容 metadata buffer。
- 同步 decode/prefill updater 中缓存的 buffer 引用。
- 只影响启动/capture 阶段，不改运行时 attention 计算。

一句话总结：

这是 SGLang FlashInfer backend 没有处理 CUDA graph padded batch size 的 buffer sizing bug；HiCache 只是让实际 request pool 变成非对齐
值，从而触发了它。
```



--enable-dp-lm-head

--mem-fraction-static

--kv-cache-dtype

显存

--enable-hierarchical-cache --hicache-ratio 4 --hicache-io-backend=kernel --hicache-storage-prefetch-policy=wait_complete --hicache-write-policy=write_through





--attention-backend 可以试试 默认是fa

--enable-torch-compile ✅

--cuda-graph-max-bs 默认打开

异步调度 ✅默认打开





## 三、准入控制
在实验过程中发现cache命中出现中间坍塌的情况，输出token的吞吐也一路下跌。这个应该是到中期的时候并发session太大，不断的刷新cache，导致无法cache命中，因此都在做prefill，所以导致输出token的吞吐也一路下跌。因此需要对session做准入控制。

<img src="https://cdn.nlark.com/yuque/0/2026/png/27134363/1779192927892-86cacf76-7c34-4ae7-82aa-af581c34dc5d.png" width="850" title="" crop="0,0,1,1" id="u2a369f4d" class="ne-image">

首先是做的最多让他跑64个session，效果明显，后面的cache命中保持很高，且输出token吞吐变化也很平坦，但是缺点是64硬编码不合适。

```bash
 SPECULATIVE_ALGO=NONE bash serve_E_dpa_tp8_dp2.sh

 bash router_E_dp2_cache_aware.sh

CONFIG=E_dpa_tp8_dp2_router_full_active64 MAX_ACTIVE_SESSIONS=64 SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```

 MAX_ACTIVE_SESSIONS=64 效果还可以

```bash
# server
SPECULATIVE_ALGO=NONE bash serve_E_dpa_tp8_dp2.sh

# router 改到 30001
ROUTER_PORT=30001 bash router_E_dp2_cache_aware.sh

# proxy 占 30000，做 in-flight=64
PROXY_PORT=30000 ROUTER_URL=http://127.0.0.1:30001 PROXY_POLICY=passthrough_router MAX_INFLIGHT_REQUESTS=64 bash proxy_length_aware.sh

# client 仍然打 30000
SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh

```





dynamic 准入控制：MAX_ACTIVE_SESSIONS=64 

```bash
# server
SPECULATIVE_ALGO=NONE bash serve_E_dpa_tp8_dp2.sh

# router
ROUTER_PORT=30001 bash router_E_dp2_cache_aware.sh

PROXY_PORT=30000 \
ROUTER_URL=http://127.0.0.1:30001 \
PROXY_POLICY=passthrough_router \
MAX_ACTIVE_SESSIONS=64 \
SESSION_IDLE_TIMEOUT=7200 \
PROXY_METRICS_PATH=../runs/proxy_E_active64_metrics.jsonl \
bash proxy_length_aware.sh

BASE_URL=http://127.0.0.1:30000 \
TIMEOUT=7200 \
SUMMARY_INTERVAL=60 \
bash client_E_dpa_tp8_dp2_router_full.sh

```



```bash
# server
SPECULATIVE_ALGO=NONE bash serve_E_dpa_tp8_dp2.sh

# router
ROUTER_PORT=30001 bash router_E_dp2_cache_aware.sh

PROXY_PORT=30000 ROUTER_URL=http://127.0.0.1:30001 PROXY_POLICY=passthrough_router MAX_ACTIVE_SESSIONS=64 SESSION_IDLE_TIMEOUT=7200 DYNAMIC_ADMISSION=1 DYNAMIC_MIN_INFLIGHT_REQUESTS=64 DYNAMIC_INITIAL_INFLIGHT_REQUESTS=128 DYNAMIC_MAX_INFLIGHT_REQUESTS=256 DYNAMIC_CONTROL_INTERVAL=60 PROXY_METRICS_PATH=../runs/proxy_E_active64_dynamic_inflight_metrics.jsonl bash proxy_length_aware.sh

BASE_URL=http://127.0.0.1:30000 SUMMARY_INTERVAL=60 bash client_E_dpa_tp8_dp2_router_full.sh
```







## 三、剪枝
参考的论文：REAP THE EXPERTS: WHY PRUNING PREVAILS FOR ONE-SHOT MOE COMPRESSION

**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">ICLR 2026</font>**会议论文，针对稀疏激活混合专家（**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">SMoE</font>**）大模型的**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">一次性压缩</font>**问题，通过理论与实验证明**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">专家剪枝</font>**在生成任务上显著优于专家合并；提出**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">REAP（Router-weighted Expert Activation Pruning）剪枝准则，综合路由器门值与专家激活范数最小化重构误差，在20B~1T 参数</font>**的 6 款 SMoE 模型上，**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">50% 压缩率</font>**下生成任务性能损失远低于合并方法，在**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">Qwen3-Coder-480B、Kimi-K2</font>**等代码模型上实现**<font style="color:rgb(28, 31, 35);background-color:rgba(0, 0, 0, 0);">近无损压缩</font>**，并开源代码与模型权重推动相关研究。

主要减少显存占用119->96G

```bash
PRUNE_MODE=reap_text_only bash serve_A_tp8.sh
```

```bash
bash serve_E_pruned_dpa_tp8_dp2.sh
bash router_E_dp2_cache_aware.sh

SUMMARY_INTERVAL=60 bash client_E_pruned_dpa_tp8_dp2_router_full.sh
```



晚上可以跑

1、不同量化的loss 速度【baseline tp8、kvfp8、剪枝模型、剪枝模型+kvfp8】

2、目前一些配置的速度【dp2tp8+hicache、dp8tp8+hicache、dp8并发控制，dp2并发控制】



1、dp2tp8+hicache

2、dp8tp8+hicache

3、tp+hicache

4、准入控制+dp2tp8+hicache

5、准入控制+dp8tp8+hicache

精度问题：  
6、kvfp8

7、剪枝模型 + hicache ratio可以试试4先测一下

8、剪枝模型+kvfp8



1、并发控制跑完【暂时看起来有效果】【正在做】

2、量化集成过来【】

3、准备晚上的脚本



```bash
SPECULATIVE_ALGO=NONE \
MEM_FRACTION_STATIC=0.92 \
ENABLE_HICACHE=1 \
HICACHE_RATIO=2 \
HICACHE_IO_BACKEND=kernel \
HICACHE_STORAGE_PREFETCH_POLICY=wait_complete \
HICACHE_WRITE_POLICY=write_through \
bash qiannan-test/scripts/serve_E_dpa_tp8_dp2.sh
```



【看看准入控制相关论文】







1、下午4点左右开始整理代码

2、
