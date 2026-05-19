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
bash client_A_tp8.sh T1
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

```plain
#启动 server：
bash qiannan-test/scripts/serve_C_dpa_tp8_dp8_no_router.sh

#C 的 client 默认打：http://localhost:8000
# 然后另一个 shell 跑 T1：
bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T1
#跑 T6：
bash qiannan-test/scripts/client_C_dpa_tp8_dp8_no_router.sh T6
```





### D：中等 DPA
**--tp-size 4 --dp-size 2 --enable-dp-attention**  
client -> router -> SGLang

目的：找 DP size 的甜点。

B 是比较激进的 DPA；D 是中间方案。DP rank 少一些，cache 被切分得没那么碎，但吞吐并行度也少一些。

D 用来回答：dp_size=8 是否太激进？dp_size=2/中等 DPA 是否更稳？

预期：

+ cache_hit_rate 可能高于 B
+ 吞吐可能低于 B
+ avg_session_time 可能介于 A 和 B 之间
+ 如果 B cache 抖动明显，D 可能反而更好







总结一下对比矩阵：

A vs B:  
  DPA 是否带来收益

B vs C:  
  router 是否必要

A vs C:  
  cache 被打烂的代价

B vs D:  
  dp_size=8 是否比中等 DPA 更好

T1:  
  看 cache/router 是否符合预期

T6:  
  看最终真实 workload 性能
