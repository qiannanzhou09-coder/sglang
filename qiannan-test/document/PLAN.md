# Qwen3.5-122B Serving 性能优化 —— 详细方案

> 目标：建立一套**可复用的"workload + observability + sweep"方法论**，并用它在单机 8×H200 上找到 SGLang 服务此 workload 的最佳配置组合。优化指标 `avg_session_time`，兼顾 TTFT、round_latency。

---

## 一、题目分析与核心洞察

### 1.1 workload 数据画像（`workload_data.jsonl`，500 session / 36550 轮）

| 维度 | 数据 |
|---|---|
| Session 数 | 500 |
| 每 session 轮数 | min 13 / avg 73 / max 109 |
| 总轮数 | 36,550 |
| `input` 字段（本轮新增 user 内容） | p50 ≈ 1.3k / p99 ≈ 2k / **max 130k**（216 轮 > 10k） |
| `output` | p50 = 166 / p99 = 848 / max 1024 |
| `wait`（轮间间隔） | p50 = 4.2s / p99 = 56.8s / max 60s |
| **system_prompt 种类** | **只有 5 种**，每种约 100 session 复用 |
| **每 session 最终累积送给 server 的真实 prompt** | min 91k / avg 148k / **max 150k tokens** |
| 真实 `prompt_tokens > 50k` 的轮数 | **22,404 / 36,550（61%）** |
| 真实 `prompt_tokens > 100k` 的轮数 | **11,937（33%）** |

### 1.2 核心洞察：这是一道"前缀缓存命中 + 长上下文调度"题，不是单纯吞吐基准

证据：`run_workload.py` 里 `messages.append(...)` **每轮拼上全部历史**。一个 session 跑到第 73 轮时，每轮真实送给 server 的 prompt 是 ~150k tokens 的 prefix + 一两千新增。

→ **前缀缓存命中决定性命脉**：

1. **session 内前缀复用**（第 N 轮 ≈ 第 N-1 轮的 prompt+output）→ RadixAttention KV 必须保住不被驱逐
2. **session 间 system_prompt 复用**（5 个 sysprompt × 100 session）→ 同 sysprompt 的 session 应共享 KV
3. **极少量超长 input（>10k，最高 130k）的离群轮**不能堵死 batch → chunked prefill + 调度策略
4. **session 间有 wait（中位数 4.2s，p99 56.8s）** → KV 在 wait 期间不能被驱逐
5. **MoE + 投机解码 accept rate** → NEXTN（SGLang）/ MTP（vLLM）的 accept 长度决定 decode 段加速

### 1.3 硬件 / 模型约束

- 单机 **8×H200 141GB**，HBM 总 1128GB
- Qwen3.5-122B-A10B-FP8：~122GB 权重，TP8 后剩余 ~1000GB 可作 KV pool
- 单卡 ~125GB 能存约 250 万 token KV（FP8 GQA 估算），TP8 统一池约 2000 万 token
- **500 session × 150k = 7500 万 token 是理论上限**，但实际不是同时活跃（session 间有 wait p50 4.2s / p99 56.8s，client 并发受 in-flight 限流）。**估算同时活跃 ~30-100 session** → 活跃 KV 100~150 万 token，**完全可全驻留**
- → **HiCache（CPU 卸载）在 H200 上预计收益小**，但**不能直接判死刑**：如果 session_aware 策略让活跃 session 数大于 ~130（KV pool 撑不住），HiCache 仍可能有价值。T5 fixture 必须认真跑

### 1.4 baseline 现状

`workload_metrics.jsonl` 是 5 个 session × 321 轮的 smoke test（早期版本），无 `avg_session_time` 字段，**不能作为参考值**。需要重新跑 baseline。

---

## 二、关键架构决策：分层与职责

### 2.1 完整数据流

```
client (run_workload.py，注入 trace_id)
  │
  ▼
our_proxy :8888                          ← 我们的准入/速率/反馈控制层
  │  职责：全局 in-flight 控制、token budget、TCP-cong-control 风格反馈、
  │       trace_id 贯穿、Observer 聚合
  ▼
sglang_router :30000  (--dp-aware --policy cache_aware)
  │  职责：cache-aware 选择目标 dp_rank（写入 data_parallel_rank 字段）
  │  这是 SGLang 官方实现，免费且生产级，我们不自己写
  ▼
SGLang :8000  (--tp-size 8 --dp-size 8 --enable-dp-attention)
  │  收到 data_parallel_rank=K → maybe_external_dp_rank_routing 跳过内置 round_robin
  │  → 直发到 workers[K]
  ▼
SGLang internal radix tree + scheduler（lpm 调度 + chunked prefill）
```

### 2.2 为什么不是其它架构

**实证依据**（已读 `/tmp/sglang/python/sglang/srt/managers/data_parallel_controller.py` 和 `/tmp/sglang/sgl-model-gateway/src/policies/cache_aware.rs`）：

| 选项 | 否决理由 |
|---|---|
| 仅 SGLang `--tp-size 8` | attention TP all-reduce 通信开销；DPA 吞吐增益拿不到 |
| 仅 SGLang `--tp-size 8 --dp-size 8 --enable-dp-attention`（无 router） | SGLang 内置 4 种 DP 分配策略 `ROUND_ROBIN/FOLLOW_BOOTSTRAP_ROOM/TOTAL_REQUESTS/TOTAL_TOKENS` **都不是 cache-aware**，默认 round_robin 会把同 session 的请求随机打到不同 dp_rank → cache 命中率崩溃 |
| 把调度逻辑 fork 进 SGLang 内部 | fork 维护成本高；要解决的是"入口准入"，本就不该塞进 engine |
| 直接用 sglang_router 作准入控制 | router 是 Rust 二进制，加新策略要改源码；它的设计点是 cache-aware 分发，不是流控；扩展性差 |
| **proxy + router 组合（我们的选择）** | proxy 解决 router 不管的事（准入、速率、全局 budget、trace 聚合）；router 解决官方已实现的事（cache-aware → dp_rank）；**职责正交**，两者独立演进，未来加新调度策略只改 proxy |

### 2.3 关键证据片段

SGLang 内部 DP 分配（无 cache-aware）：

```python
# /tmp/sglang/python/sglang/srt/managers/data_parallel_controller.py
class LoadBalanceMethod(Enum):
    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

def round_robin_scheduler(self, req: Req):
    if self.maybe_external_dp_rank_routing(req):  # ← router 的钩子
        return
    while True:
        if self.status[self.round_robin_counter]:
            self.workers[self.round_robin_counter].send_pyobj(req)
            ...

def maybe_external_dp_rank_routing(self, req: Req):
    if req.routed_dp_rank is not None:
        self.workers[req.routed_dp_rank].send_pyobj(req)
        return True
    return False
```

sglang_router 的 cache_aware policy（按 dp_rank 维护独立 tree）：

```rust
// /tmp/sglang/sgl-model-gateway/src/routers/http/router.rs
fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
    // worker URL 格式: http://host:port@dp_rank, 如 http://localhost:8000@5
    ...
}

// 转发时把 dp_rank 写进请求体
const DP_RANK_KEY: &str = "data_parallel_rank";
map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
```

---

## 三、服务端架构与参数旋钮全集（H200 单机视角）

这一章先列出 4 个核心架构配置（作为阶段 1 选型对象），再把所有 SGLang 启动参数（作为阶段 2 精调对象）系统列清楚——特别是**静默修改的坑**和**互斥约束**。所有数值都对照 `/tmp/sglang/python/sglang/srt/server_args.py` 源码验证过。

### 3.1 四个核心架构配置（阶段 1 选型用）

| 标签 | SGLang 启动 | 入口 | 实验目的 |
|---|---|---|---|
| **A. 纯 TP（对照基线）** | `--tp-size 8`（即当前 `sglang_serve.sh`） | 直接打 SGLang :8000 | 单进程统一 KV，cache 100% 自然命中；attention TP all-reduce 开销 |
| **B. DPA + router（生产推荐）** | `--tp-size 8 --dp-size 8 --enable-dp-attention` | sglang_router :30000 `--dp-aware --policy cache_aware` | DPA 吞吐 + router cache 命中兼得 |
| **C. DPA 裸跑（反面教材）** | 同 B 的 SGLang，**不挂 router** | 直接打 SGLang :8000 | 证明"光开 DPA 不挂 router"会把 cache 命中打烂——预期 avg_session_time 比 A 还差 |
| **D. 中等 DPA**（可选） | `--tp-size 4 --dp-size 2 --enable-dp-attention` + router | sglang_router :30000 | 找 dp_size 的甜点；时间紧张可砍 |

四组实验本身构成"为什么生产用方案 B"的完整论证：
- **A vs C** → cache 命中的价值
- **B vs A** → DPA 的吞吐增益
- **B vs C** → router 的价值
- **B vs D** → dp_size 甜点

> ⚠️ 配置 B/D 开 DPA 后两个参数会被**静默修改**，必须显式补偿才能跟 A 公平对比。详见 3.3。

### 3.2 H200 141GB 的官方默认值（源码 1439~1448 行 H20/H200 分支）

| 参数 | H200 默认 | 说明 |
|---|---|---|
| `--chunked-prefill-size` | **8192** | 单步 prefill 最多多少 token |
| `--cuda-graph-max-bs` | **512**（tp ≥ 4） | CUDA graph 最大 batch |
| `--mem-fraction-static` | **自动估算** `(gpu_mem - reserved) / gpu_mem` ≈ 0.88 | KV pool 占 HBM 比例（baseline 显式写 0.8） |
| `--page-size` | **1**（非 MLA backend） | KV cache 页大小 |
| `--schedule-conservativeness` | **1.0** | 越大越保守（少抢占多排队） |

### 3.3 ⚠️ DPA（`--enable-dp-attention`）的两个静默修改（关键坑）

源码 `server_args.py:3061-3066`：

```python
if self.enable_dp_attention:
    self.schedule_conservativeness = self.schedule_conservativeness * 0.3  # ← 没有 warning！
    assert self.tp_size % self.dp_size == 0
    self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size  # ← 你提到的坑
    logger.warning(
        f"DP attention is enabled. The chunked prefill size is adjusted to {self.chunked_prefill_size}..."
    )
```

含义：

| 你写 | 配置 A（TP8）实际生效 | 配置 B（TP8+DP8 DPA）实际生效 |
|---|---|---|
| `--chunked-prefill-size 8192` | **8192** | **8192 // 8 = 1024** |
| `--schedule-conservativeness 1.0`（默认） | **1.0** | **0.3** |

→ **配置 B 处理这两个修改的两种思路，都要测**：

```bash
# 思路 1：完全补偿（A vs B 干净对比 DPA 本身）
--chunked-prefill-size 65536       # 65536 // 8 = 8192，effective 等同 A
--schedule-conservativeness 3.33   # 3.33 × 0.3 ≈ 1.0，effective 等同 A

# 思路 2：尊重 SGLang 默认（SGLang 团队改成 0.3 是有理由的：DPA 下 batch 小，激进调度更优）
--chunked-prefill-size 8192        # effective 变 1024，可能更适合 DPA 的小 batch
# 不写 --schedule-conservativeness，让它自动变 0.3
```

阶段 1 同时跑 **B(补偿)** 和 **B(默认)** 两个变体，看哪个更优。**否则会犯"想公平比较反而劣化 DPA 配置"的错误**。改后矩阵从 3 个变 4 个 sglang_config（A、B补偿、B默认、C）。

### 3.4 主要调优旋钮（阶段 2 精调对象，按优先级排序）

| # | 参数 | 默认 / baseline | 候选值 | 影响维度 | 备注 |
|---|---|---|---|---|---|
| **P0（架构层）** ||||||
| 1 | `--enable-dp-attention` | off | on（要配 router） | 吞吐 | 必须 `tp_size % dp_size == 0`；开后两个参数被静默改 |
| 2 | `--dp-size` | 1 | 1 / 2 / 4 / 8 | 吞吐 + cache 切分 | 8 个 DP rank = 8 个独立 KV pool |
| 3 | `--enable-dp-lm-head` | off | on（仅当 DPA 开） | 进一步减少通信 | LM head 也走 DP，配合 DPA 用，避免 LM head 单独 TP all-reduce |
| 4 | `--schedule-policy` | `lpm` | `lpm` / `fcfs` / `lof` / `random` / `dfs-weight` | cache 命中 | lpm = 最长前缀匹配（最重要） |
| 5 | `--disaggregation-mode` | `null` | **固定 null，不扫**（PD 分离要起两组 server + bootstrap，不是单 sweep 旋钮） | PD 分离架构 | 详见 3.8。**对此 workload 不推荐** |
| **P1（cache 与内存）** ||||||
| 6 | `--mem-fraction-static` | 0.88 自动 / baseline 0.8 | 0.7 / 0.85 / **0.92 / 0.95**（找极限） | KV pool 大小 | 太高 OOM，太低 cache 装不下；**朋友提的"极限 mem frac"用 0.95 试到边界** |
| 7 | **`--kv-cache-dtype`** | `auto`（FP16） | `auto` / `fp8_e4m3` / `fp8_e5m2` | **KV 容量翻倍** | **关键遗漏**：FP8 KV cache 让前缀缓存能容纳 2× session 数，对此 150k 长 prompt workload 影响巨大；H200 原生支持 FP8 |
| 8 | `--disable-radix-cache` | off | on（反面教材） | cache 命中 | 关掉前缀缓存，预期慢 5~10× |
| 9 | `--enable-hierarchical-cache` | off | on | 长 wait 场景 | KV 卸载到 CPU；H200 上预期收益小 |
| 10 | `--hicache-ratio` | 2 | 2 / **4** / 8 | HiCache CPU 池大小倍数 | 朋友推荐 4 |
| 11 | `--hicache-io-backend` | `direct` | `direct` / **`kernel`** | HiCache 数据通路 | 朋友推荐 kernel |
| 12 | `--hicache-storage-prefetch-policy` | `best_effort` | `best_effort` / **`wait_complete`** | 预取策略 | 朋友推荐 wait_complete，保证命中率 |
| 13 | `--hicache-write-policy` | `write_back` | `write_back` / **`write_through`** | 写策略 | 朋友推荐 write_through，避免 KV 丢失 |
| 14 | `--page-size` | 1 | 1 / 16 / 64 / 128 | 内存碎片 vs 灵活性 | 受 attention backend 强约束（见 3.5） |
| **P2（调度与吞吐）** ||||||
| 15 | `--chunked-prefill-size` | 8192（H200） | 4096 / 8192 / 16384 / 32768 | 长 prefill 公平性 vs 吞吐 | **DPA 下自动除以 dp_size** |
| 16 | `--enable-mixed-chunk` | off / baseline on | on / off | 吞吐 | prefill 和 decode 同 batch 混合，提高吞吐 |
| 17 | `--schedule-conservativeness` | 1.0 | 0.3 / 1.0 / 3.0 | 抢占率 | 越高越少抢占（更保守地塞 batch） |
| 18 | `--max-running-requests` | 自动 | 显式 256 / 512 / 1024 | 并发上限 | 跟 KV pool 一起决定 batch 上限 |
| 19 | `--max-prefill-tokens` | 16384 | 8k / 16k / 32k | 单步 prefill 总量 | 跟 chunked-prefill-size 联动 |
| **P3（投机解码）** ||||||
| 20 | `--speculative-algo` | none / baseline `NEXTN` | none / NEXTN / EAGLE / EAGLE3 | decode 加速 | Qwen3.5 用 NEXTN |
| 21 | `--speculative-num-steps` | 3 | 1 / 3 / 5 / 7 | accept 长度 vs 草稿开销 | 越大并发受益越小（大 batch 时反而拖累） |
| 22 | `--speculative-eagle-topk` | 1 | 1 / 2 / 4 | 草稿多样性 | 大于 1 时 verify 开销显著 |
| 23 | `--speculative-num-draft-tokens` | 4 | 3 / 4 / 8 | 草稿池大小 | 通常 = num_steps + 1 |
| **P4（MoE / EP）** ||||||
| 24 | `--ep-size` | 1 | 1 / 4 / 8 | MoE expert 通信 | 配合 DPA 用 |
| 25 | `--moe-dp-size` | 1 | 1 / dp_size | MoE 的 DP 度 | 默认跟随 dp_size |
| 26 | `--enable-deepep-moe` | off | on | A2A 通信 | H 系列上有收益 |
| **P5（编译/后端）** ||||||
| 27 | **`--attention-backend`** | 自动选 | `flashinfer` / `fa3` / `triton` / `torch_native` | attention kernel | 默认 H200 用 flashinfer；H 系列上 fa3 也试一下，**朋友提到 SageAttention 低精度 attention 但 SGLang 主线还没支持** |
| 28 | **`--enable-torch-compile`** | off | on | decode 段加速 | torch.compile，**朋友提到要试**。注意会**强制关 piecewise CUDA graph**（1288 行） |
| 29 | `--cuda-graph-max-bs` | 512（H200, tp ≥ 4） | 256 / 512 / 1024 | decode 段延迟 | 启动时间长 + 显存 |
| **P6（其它）** ||||||
| 30 | `--mamba-scheduler-strategy` | baseline `extra_buffer` | `no_buffer` / `extra_buffer` | Qwen3.5 含 mamba 层专用 | 这是混合模型，必须有 |
| 31 | `--enable-cache-report` | off / baseline on | on | 可观测 | response 里返回 `cached_tokens`，**必开** |
| 32 | `--enable-metrics` | off | on | 可观测 | Prometheus `/metrics`，**必开** |

### 3.5 互斥约束（避免启动失败）

来自源码各处 assert：

| 约束 | 来源 |
|---|---|
| `chunked_prefill_size % page_size == 0` | 6911 行 |
| DPA 要求 `tp_size % dp_size == 0` | 3063 行 |
| `--enable-mis` 会强制关闭 `radix_cache` 和 `chunked_prefill` | 1356~1365 行 |
| `--enable-hierarchical-cache` 会强制关闭 piecewise cuda graph | 1322 行 |
| DPA 会强制关闭 piecewise cuda graph | 1285 行 |
| FlashMLA backend 强制 `page_size = 64`；Cutlass MLA 强制 128；TRT-LLM MLA 强制 64；FA4 强制 128 | 2694~2796 行 |
| `--mamba-scheduler-strategy no_buffer` 是 ModelScope 默认 | 1160 行 |

### 3.6 投机解码（NEXTN）值的取舍

Qwen3.5 baseline 用的是 `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`。规律：

- **batch 小 + decode 多**（如 fixture T4）→ num_steps 越大越好，可试 5/7
- **batch 大 + cache 命中高**（如 fixture T1）→ num_steps 大反而拖累（每 verify 步消耗 GPU 时间），可试 1/3
- **topk > 1 几乎从无收益**（draft 模型 verify 串行化）

### 3.7 配置 A / B 完整启动脚本（**KV cache 默认 FP16，FP8 KV 作为独立实验**）

```bash
# === 共用变量 ===
MODEL=/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-122B-A10B-FP8

# === 配置 A：纯 TP，baseline ===
# 注意：默认 KV 是 FP16（auto），FP8 KV 放阶段 2 单独跑精度+性能实验
SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
  --model-path $MODEL --port 8000 \
  --tp-size 8 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tool-call-parser qwen3_coder \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size 8192 \
  --enable-mixed-chunk \
  --schedule-policy lpm \
  --enable-cache-report \
  --enable-metrics

# === 配置 B-compensated：DPA + router，effective 等同 A（用来干净对比 DPA 本身）===
# 注意 1：DPA 会把 --chunked-prefill-size 自动除以 dp_size，所以写 65536 → effective 8192
# 注意 2：DPA 会把 --schedule-conservativeness 自动 × 0.3，所以写 3.33 → effective ≈ 1.0
# 注意 3：bash 续行 \ 后必须立即换行，不能跟空格或注释！
SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
  --model-path $MODEL --port 8000 \
  --tp-size 8 --dp-size 8 --enable-dp-attention --enable-dp-lm-head \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tool-call-parser qwen3_coder \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size 65536 \
  --schedule-conservativeness 3.33 \
  --enable-mixed-chunk \
  --schedule-policy lpm \
  --enable-cache-report \
  --enable-metrics

# === 配置 B-default：DPA + router，尊重 SGLang DPA 默认（看官方默认是否更优）===
# 跟 B-compensated 完全一样，去掉 --chunked-prefill-size 和 --schedule-conservativeness 两行
# effective chunked=1024, conservativeness=0.3

# router（B 配套，B-compensated 和 B-default 都用这个）
python -m sglang_router.launch_router \
  --worker-urls http://127.0.0.1:8000 \
  --dp-aware --policy cache_aware \
  --host 0.0.0.0 --port 30000

# === 配置 C：DPA 裸跑（不挂 router，反面教材，证明 router 价值）===
# 同 B-compensated 的 SGLang，但 client 直接打 :8000 不经过 router

# === FP8 KV 变体（阶段 2 单独跑，精度先过 RULER-16k 才合并到 winner）===
# 在 winner 配置上加：
#   --kv-cache-dtype fp8_e4m3        # 主推荐
# 或：
#   --kv-cache-dtype fp8_e5m2        # 备选（精度更松）

# === REAP-20-FP8 自制版本（阶段 3 P2，第二天上午自制 ckpt 后启）===
# 详见 3.9.4 路线 A —— 跳过下载 0xSero BF16 版本（200GB 浪费时间），直接在官方 FP8 上按 plan 剪
SGLANG_ENABLE_SPEC_V2=1 python -m sglang.launch_server \
  --model-path /path/to/Qwen3.5-122B-A10B-REAP-20-FP8-text-only \
  --tp-size 8 \
  --mem-fraction-static 0.8 \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --tool-call-parser qwen3_coder \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size 8192 \
  --enable-mixed-chunk \
  --schedule-policy lpm \
  --enable-cache-report --enable-metrics
# ✅ 权重 ~99GB（含我们顺手删 visual.* 节省 ~1GB），KV pool ~1030GB（比 FP8 baseline 多 30GB）
```

### 3.8 SGLang PD disaggregation 模式（朋友提到，但**本题不推荐**）

朋友说"可以看看 prefill 和 decode 分离的方案，sglang 自带"——SGLang 0.4+ 确实有 `--disaggregation-mode prefill/decode`，把 prefill 和 decode 跑在两组独立的 worker 上，靠 KV 传输衔接。

**为什么本题不推荐做主架构**：

| 维度 | 影响 |
|---|---|
| 硬件 | 单机 8 GPU 切分成 2 组（如 4 prefill + 4 decode）→ **每组只有 4 卡的 KV 池**，单 session 150k 长 prompt 可能装不下 |
| Cache | PD 分离后，**前缀缓存被劈成两边**，session 的 KV 在 prefill 和 decode 之间要传输，长 prompt 场景反而劣化 |
| 适用场景 | 真正受益的是"超大并发 + 短上下文 + decode 主导"workload（如 chat 应用）；我们这里是"高 cache 复用 + 长上下文" |
| 复杂度 | PD 分离要起 bootstrap server、配置 transfer backend（mooncake/nixl/ascend），调通耗时 |

**记录但不进入实验矩阵**。等阶段 1/2 跑完，如果还有时间可以做一组对比验证我们的判断。

### 3.9 模型层加速：量化与剪枝综合调研（含 HF 现成版本梳理）

> 数据来源：朋友给的两个 dr.miromind.ai 链接 + Qwen 官方 HF + SGLang 官方文档 + 朋友筛出的 6 个 HF 候选版本。所有 SGLang 兼容性结论都对照过 SGLang quantization 文档。

#### 3.9.1 通用方案对比矩阵

| 方案 | 类型 | Qwen3.5-122B 现成版本 | SGLang 兼容性 | 精度（vs FP16） | 速度增益 | 风险 |
|---|---|---|---|---|---|---|
| **官方 block-wise FP8**（**当前 baseline 就在用**） | W8A8，per-block 128 | ✅ `Qwen/Qwen3.5-122B-A10B-FP8` | ✅ 原生支持 | ~无损 (<1%) | weight 体积减半 | 无 |
| **`--kv-cache-dtype fp8_e4m3`** | **KV cache 量化**（独立于权重量化） | 任意 FP16/FP8 权重均可 | ✅ 原生 | KV 量化对长 context 精度有小损失（需测） | **KV 容量翻倍** → cache 命中率提升 → **本题潜在最大收益** | long-context 精度需回归，**不默认开** |
| **`--kv-cache-dtype fp8_e5m2`** | KV 量化（另一格式） | 同上 | ✅ 原生 | 比 e4m3 更松，长 context 精度更差 | 同 e4m3 | 同上 |
| **NVFP4 / MXFP4 (W4A4)** | W4A4 微缩放 | ✅ 英伟达版（234G→75G） | ⚠️ SGLang 支持中；**Hopper 上无硬件加速**，H200 跑可能反而慢 | ~near-lossless (<1%) | 仅 Blackwell 才有真加速 | **H200 不一定快**，需测，**放阶段 3 最后**做 |
| **REAP MoE 剪枝 + FP8（自制）** | 复用 0xSero pruning plan 在官方 FP8 ckpt 上剪 51 个 expert/层 | ⚠️ HF 现成只有 BF16 版（显存悖论）；**我们自制 FP8 版**（第二天 1~2h 搞定） | ✅ 改完是普通 FP8 ckpt，SGLang 原生 | 平均保留 97.9% capability（0xSero plan）+ 自测 RULER-16k 守门 | **weight ~99GB（vs FP8 baseline 120GB），KV pool +30GB** | 详见 3.9.3 + 3.9.4 |
| **Unsloth Dynamic GGUF / APEX / ParoQuant / REAP-GGUF / MLX** | 各种 mixed-bit / GGUF / MLX | ✅ 多个版本 | ❌ SGLang 对 Qwen3.5 MoE GGUF 不支持 + mixed-bit limitation；MLX 只在 Apple Silicon | -- | -- | **全部放弃**（详见 3.9.2） |

#### 3.9.2 HF 上 6 个 Qwen3.5-122B 候选版本逐一定性

朋友筛出来的 6 个，**只有 1 个能用**。

| # | 模型 | 类型/大小 | SGLang 能跑？ | 推理速度判断 | 入选我们的实验？ |
|---|---|---|---|---|---|
| 1 | [**0xSero/Qwen3.5-122B-A10B-REAP-20**](https://huggingface.co/0xSero/Qwen3.5-122B-A10B-REAP-20) | **REAP 20% + BF16 safetensors**，205/256 experts，99B params (~200GB BF16) | ✅ 能（README 自带 SGLang 命令；ckpt 自带 visual.*，SGLang 直接加载会带 ViT；**SGLang 没有 vLLM 的 `--language-model-only` 等价 flag**，详见 3.9.4b） | ⚠️ 是 BF16 不是 FP8！比 FP8 baseline 占显存多 80GB；显存悖论使其在 long-prompt 场景预期更慢 | ❌ **不下载**：只用其 `targeted_refusal_analysis.json` plan，自制 FP8 版本（详见 3.9.4 + 3.9.8） |
| 2 | [mradermacher/Qwen3.5-122B-A10B-REAP-30-GGUF](https://huggingface.co/mradermacher/Qwen3.5-122B-A10B-REAP-30-GGUF) | REAP 30% + 静态 GGUF | ❌ SGLang 对 Qwen3.5 MoE GGUF 不完整 | -- | ❌ 不用 |
| 3 | [mradermacher/Qwen3.5-122B-A10B-REAP-30-i1-GGUF](https://huggingface.co/mradermacher/Qwen3.5-122B-A10B-REAP-30-i1-GGUF) | REAP 30% + imatrix GGUF | ❌ 同上 | -- | ❌ 不用 |
| 4 | [0xdfi/Qwen3.5-122B-A10B-abliterated-REAP20-oQ6-MLX](https://huggingface.co/0xdfi/Qwen3.5-122B-A10B-abliterated-REAP20-oQ6-MLX) | abliterated + REAP-20 + MLX 6bit | ❌ **MLX 只能 Apple Silicon** | -- | ❌ 完全用不了 |
| 5 | [unsloth/Qwen3.5-122B-A10B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF) | Unsloth Dynamic 2.0 + 含 MTP 头 + GGUF（多档 1bit~BF16） | ❌ 同 GGUF 限制 | **而且 MTP 头我们 baseline 已经用了**（`--speculative-algo NEXTN`） | ❌ 不用 |
| 6 | [unsloth/Qwen3.5-122B-A10B-GGUF](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-GGUF) | Unsloth Dynamic 2.0 + GGUF（无 MTP） | ❌ 同上 | -- | ❌ 不用 |

**结论**：HF 上能在 SGLang 用的剪枝/量化版本，只有 0xSero/REAP-20 BF16 一个，但因显存悖论我们**不下载**——只取它的 pruning plan（用户已上传 `targeted_refusal_analysis.json`），在官方 FP8 ckpt 上自制 REAP-20-FP8。其余 5 个全部因 GGUF / MLX 格式被淘汰。

#### 3.9.3 REAP-20 BF16 的"显存悖论"（**重要**）

| | 当前 baseline（FP8） | REAP-20（BF16） | **REAP-20 + FP8（不存在但理论值）** |
|---|---|---|---|
| 参数量 | 122B (100% experts) | 99B (80% experts) | 99B (80% experts) |
| 权重精度 | FP8 (1 byte/param) | BF16 (2 byte/param) | FP8 |
| 权重显存 | ~120 GB | ~200 GB | **~99 GB** |
| 剩余给 KV pool | ~1000GB | ~920GB（少了 80GB） | **~1030GB（多了 30GB）** |

REAP-20 BF16 的现成版本，KV pool 反而缩水 80GB。我们 workload 是 long-prompt（p99>100k）+ 高 cache 复用，**受益于大 KV pool**：
- REAP 砍 20% expert → prefill/decode 单步 FLOPs 减少 ~15%
- 但 KV pool 缩水 ~8% → cache miss 增加
- **两个效应方向相反**，BF16 现成版本极可能反而拖累 → **决定跳过下载/实测 BF16，直接做 FP8 自制**（用户提议）

**真要拿 REAP 的好处**，需要 "REAP-20 + 再 FP8 量化"。0xSero 没出这个版本，需要自己做（plan 已就绪，1~2h 能搞定）。详见 3.9.4。

#### 3.9.4 REAP 算法本质（**完全 zero-shot，无训练**）—— 可自己做 "REAP + FP8"

REAP 论文 [arxiv 2510.13999](https://arxiv.org/abs/2510.13999) 第 5 节明确："All models are evaluated in the one-shot setting, **with no additional fine-tuning after compression**."

**算法核心**（公式 9）：

```
S_j = (1/|X_j|) * Σ_{x ∈ X_j} g_j(x) * ||f_j(x)||_2
```

- `g_j(x)`：router gate 给 expert j 的权重
- `||f_j(x)||_2`：expert j 在 token x 上的输出 activation norm
- `X_j`：**只在 expert j 被激活的 token 上做平均**（不是全 token；避免低频 expert 被误剪）

**完整流程**（一次性，无训练）：
1. 准备校准数据：1024 samples × 2048 token（≈200 万 token），或更激进 24k samples × 16k token
2. 在原始 ckpt 上跑 forward，打 hook 收集每个 expert 的 `g_j` 和 `||f_j||_2`
3. 计算 `S_j`，按 saliency 排序
4. 剪掉最小的 25%/50% expert：
   - 删 FFN 权重（up/gate/down proj 三个矩阵）
   - 删 router weight 对应行（或让对应 logit 永远 -inf）
   - 修改 config（`num_experts: 256 → 205`）
5. 重新打包 safetensors

**对我们的可行性**：

| 路线 | 工作量 | 关键风险 | 优先级 |
|---|---|---|---|
| **A：复用 0xSero pruning plan + FP8 权重** | **1~2h**（plan 已拿到，详见下）：写脚本按 expert_id 删 FP8 weights、改 router、打包 safetensors | BF16 选的 expert 在 FP8 上未必绝对最优；但相对顺序大概率保留，几乎无额外精度损失 | **🌟 第二天上午第一件事** |
| B：在官方 FP8 ckpt 上自己重新跑 REAP | 4~6h：用 0xSero 公开的 [reap-calibration-data-v1](https://huggingface.co/datasets/0xSero/reap-calibration-data-v1)（23k samples）在 8×H200 跑 forward 收集 stats，然后剪 | FP8 量化误差对 saliency 排序影响小但要测；要实现 hook | 第二天下午（如果 A 翻车） |
| C：自己写 REAP + 自定义校准数据 | 1~2 天 | 重复造轮子 | 不做 |

**🟢 路线 A 的 pruning plan 已确认拿到**：

- 文件：`/Users/kausal/Downloads/targeted_refusal_analysis.json`（75936 行，48 layer × 51 expert/层）
- 字段：每层 `pruned_indices` 数组直接给出要删的 expert_id
- 元信息确认就是 0xSero/REAP-20 用的那份：`model_name: Qwen3.5-122B-A10B` + `metric: reap` + `compression_ratio: 0.2` + `n_experts_to_prune: 51` + `preserve_threshold: 0.8` —— **完全对得上** model card 的 "REAP with targeted refusal preservation, Preserve Threshold 80%"

**⚠️ 这是 "targeted_refusal" 变体（不是纯 REAP）**：
- `target_category: "refusal_ablation"` + 每层保留 110~150 个 refusal-dominant expert
- 意思是：剪枝**保护了对"拒绝回答有害问题"重要的 expert**，避免剪后模型对齐崩坏
- **对我们 serving 性能完全无影响**（refusal 是对齐特性，不影响 prefill/decode 速度）
- 反而是好事：保持模型对齐行为正常，不会变成 abliterated 那种乱答

**路线 A 的具体步骤**（第二天上午执行，**冯老师已确认 OK，无前提阻塞**）：

```
脚本 build_pruned_ckpt.py 一次性输出 3 个 ckpt：
  1. baseline-FP8-text-only     ← 只删 visual.*，保留 256 expert（用于 KV FP8 实验的更优 baseline）
  2. REAP-20-FP8                ← 只剪 expert，保留 visual.*（risk-free 备份）
  3. REAP-20-FP8-text-only      ← 剪 expert + 删 visual.*（最终主推）
```

详细步骤：
1. 解析 `targeted_refusal_analysis.json` → 字典 `{layer_id: [51 个 expert_id]}`（**注意：plan 里的 expert_id 都是 routed expert 0~255，不涉及 shared_expert**）
2. 从 `Qwen/Qwen3.5-122B-A10B-FP8` 加载 ckpt（safetensors，~120GB）
3. **共用函数 `strip_visual(state_dict)`**：删所有 `visual.*` 权重 + 删 config 的 vision 字段（vision_config、image_token_id 等）；**不动 `architectures` 字段**（详见 3.9.4b 步骤 2）
4. **共用函数 `prune_experts(state_dict, plan)`**：
   - 对每层 MoE，按 plan 删 `mlp.experts.{id}.{gate,up,down}_proj.weight` + 对应 `_scale_inv` 张量
   - **不要动 `mlp.shared_expert.*`**（Qwen3.5 是 "8 Routed + 1 Shared"，shared 是公用的）
   - 修改 router weight `mlp.gate.weight`：删 51 行，剩余 reindex 为 0~204
   - 改 `config.json` 的 `num_experts: 256 → 205`
5. 组合调用：
   - `ckpt_1 = strip_visual(load(FP8))` → 保存 `baseline-FP8-text-only`
   - `ckpt_2 = prune_experts(load(FP8), plan)` → 保存 `REAP-20-FP8`
   - `ckpt_3 = strip_visual(prune_experts(load(FP8), plan))` → 保存 `REAP-20-FP8-text-only`
6. 重新打包 safetensors（保持 `model.safetensors.index.json` 一致；可以共享 unmodified shard）

**🔴 路线 A 的最大技术风险：router weight reindex**

```
原始 (256 experts):
  mlp.gate.weight       shape = [256, hidden]    行 i 对应 mlp.experts.{i}
  mlp.experts.{0..255}.{gate,up,down}_proj.weight

剪 51 行后 (205 experts):
  mlp.gate.weight       shape = [205, hidden]    行 k 必须对应 mlp.experts.{k}
  mlp.experts.{0..204}.{gate,up,down}_proj.weight
                        ↑ 这里的 0..204 是**新 id**，
                          值 = 原 ckpt 中保留下来的第 k 个 expert
```

**风险**：SGLang `models/qwen3_5.py` 的 MoE 实现里如果有任何"按绝对 expert_id 索引"的隐含假设（比如某些 cuda kernel 直接读 expert_id），reindex 后模型行为会完全错乱。**必须 unit test**：
- 跑 forward 一次，对比原 ckpt（256 expert）和新 ckpt（205 expert）在某个固定 prompt 上的输出 logits
- 期望：模型输出有合理偏差（因为剪了 expert），但**不应该出现 nan / inf / 完全乱码**

**风险隔离的启动顺序**（每步过了再下一步）：
- 步骤 7a：SGLang 启 `REAP-20-FP8`（保留 visual，**先验证 reindex 正确**） → 跑 gsm8k → 跑 RULER-16k → 跑 T1（smoke）
- 步骤 7b：SGLang 启 `baseline-FP8-text-only`（不剪 expert，**只验证删 visual 是否破坏 SGLang**） → 跑 gsm8k
- 步骤 7c：SGLang 启 `REAP-20-FP8-text-only`（两个都做） → 完整阶段 3 实验

**回退规则**：
- 7a 翻车（reindex 出错）→ 整个 REAP 路线放弃，损失也不大（baseline 还在）
- 7b 翻车（删 visual 破坏 SGLang）→ 7c 也不做，最终用 7a 的 `REAP-20-FP8` 跑（损失 ~1GB KV pool 可接受）

#### 3.9.4b SGLang 跳过 multimodal vision encoder 的现状

**用户提问：能不能让 SGLang 加载时不实例化 visual head 来省显存？**

| 方案 | 可行？ | 说明 |
|---|---|---|
| SGLang `--language-only` flag | ❌ 单机不可用 | 这个 flag 是给 **EPD（Encoder-Prefill-Decode）分布式部署**用的，必须配 `--encoder-urls`。看 [server_args.py:3727](https://github.com/sgl-project/sglang) `if self.language_only and len(self.encoder_urls) == 0: raise ValueError` |
| SGLang `--enable-multimodal` 默认 None | ⚠️ 不能 disable | 文档说 "If the model being served is not multimodal, nothing will happen"。用来主动启用功能，不是关闭 |
| SGLang Qwen3.5 model.py 已有 `if "visual" in name: continue` | ⚠️ 只跳过部分权重，`self.visual` 还会实例化 | line 1179/1814 跳过 visual 权重加载，但 `self.visual = ...` 还是会创建（line 1467 调用 `self.visual.deepstack_visual_indexes`） |
| 改 SGLang 源码 / fork | ❌ 不做 | 维护成本高（每次升级要 rebase），收益 < 1.5GB 不值 |
| **在 ckpt 自己删 visual.\* + 改 config 移除 vision 字段**（路线 A 同步做） | ✅ 推荐方案 | 改 ckpt 是 self-contained 的；SGLang 完全 unmodified；详见 3.9.4 路线 A |

**收益估算**：
- Qwen3.5-122B-A10B 的 ViT 参考 Qwen2-VL-72B 是 ~0.7B 参数 ≈ ~1.4 GB BF16 / ~0.7 GB FP8
- 占总显存 < 1%；**对 1000GB KV pool 影响 ~0.1%**
- 对推理性能影响 = 0（client 不发图片，vision 不被调用）
- 单独投入精力**不值**；在 REAP-FP8 自制脚本里**顺手做 + 风险隔离**（路线 A 步骤 7a/7b/7c）

**关键执行细节**（避免踩坑）：

1. **删 visual.\* 之后 SGLang 能否加载？** 需要实测：
   - Qwen3.5 model.py 的 `__init__` 里 `self.visual = ...` 是否依赖 config 字段（如果是，删 vision_config 后会报错）
   - 如果会，可能要在 model.py 里加 monkey patch（runtime 注入），不是改源码 fork
2. **必须改 `config.json` 的哪些字段**？至少：
   - 删 `vision_config`
   - 删 `image_token_id` / `video_token_id`（如果存在）
   - **不要**改 `architectures`（改了 SGLang 找不到 model class）
3. **回退路径**：步骤 7b 翻车就用 7a 的"保留 visual"版本（损失 1GB 是可接受的）

#### 3.9.5 SGLang 的关键限制（官方文档明确）

> Mixed‑bit Quantization Limitations: Mixed‑bit quantization is not fully supported. Because of vLLM's layer fusion (e.g., QKV fusion), you cannot safely apply different bit‑widths to different components.

**含义**：
- 同一 attention 层里 Q/K/V 不能不同 bit
- 同一 linear 层不能切块用不同 bit
- 真正 runtime 级别的"每层不同 bit"，SGLang **不保证正确性**

**结论**：APEX / Unsloth Dynamic / ParoQuant 这类"内部混合"路线在 SGLang 上风险高，**全部放弃**。

#### 3.9.6 long-context 精度风险（**链接 2 重要警告**）

> Qwen3.5 attention 张量对量化特别敏感，attn_* 最好保持高精度；某些 quant 对 general chat 几乎无损，**但对 long-context coding / agent 场景下降明显**。

**对我们的意义**：本 workload 真实 prompt p50 ≈ 50k，p99 > 100k，**正是 long-context**。所以：
- `--kv-cache-dtype fp8_e4m3` 必须做 **long-context benchmark 回归**
- KV INT8 也要测（更激进，精度可能更差）
- REAP-20 也要测 long-context 精度（虽然官方 benchmark 表里它 LongBench v2 = 60.2 略掉，但不知道是不是有水分）

#### 3.9.7 精度实验矩阵（**用户明确要求做这块**）

| # | 实验项 | 短任务（gsm8k） | 长任务快速（RULER-16k）| 长任务认证（LongBench v2 子集） |
|---|---|---|---|---|
| 0 | baseline FP8 + FP16 KV | ✅ 跑 1 次（基准） | ✅ 跑 1 次（基准） | ✅ 跑 1 次（最终对照） |
| 1 | baseline FP8 + **FP8 KV (e4m3)** | ✅ | ✅ | ✅ if 短/快通过 |
| 2 | baseline FP8 + **FP8 KV (e5m2)**（备选，仅当 e4m3 超阈值） | ✅ | ✅ | 仅当 e4m3 超阈值才测 |
| 3 | **REAP-20 + FP8 自制**（详见 3.9.4 路线 A，**跳过 BF16 中间步骤**） | ✅ | ✅ | ✅ |
| 4 | NVFP4 W4A4（如能起得来） | ✅ | ⚠️ 仅当 NVFP4 在 H200 不慢才跑 | -- |

> ⚠️ **SGLang 不支持 INT8 KV cache**：`--kv-cache-dtype` 只接受 `auto / fp8_e4m3 / fp8_e5m2`（vLLM 有 INT8 KV 但 SGLang 没有）。所以 KV 量化备选只有 e4m3 和 e5m2 两个 FP8 变体。

**benchmark 选型**：
- **RULER-16k 单 length**（合成 needle-in-haystack）：在 122B 上估计 ~30-45 min，作为**快速守门**
- **LongBench v2 子集**（5~6 个长度敏感任务）：~1h，作为最终 winner **认证**
- Qwen 官方在 README 自己公布的 LongBench v2 = 60.2，我们可以**直接对照**

#### 3.9.8 我们的最终选型（按用户要求重排优先级）

| 优先级 | 方案 | 阶段 | 行动 |
|---|---|---|---|
| **P0** | 权重保持官方 FP8 + **FP16 KV**（baseline 不默认开 FP8 KV） | 阶段 1 | 第一天 baseline 跑通，gsm8k+RULER-16k 基准建立 |
| **P0** | KV FP8 (e4m3) **独立精度+性能实验** | 阶段 3 | 跑短任务+RULER-16k，过线则纳入 winner 配置 |
| **P1** | KV FP8 (e5m2) / KV INT8 | 阶段 3 | 仅当 e4m3 翻车才测 |
| **P2** | **REAP-20 + FP8 自制**（详见 3.9.4 路线 A） | 第二天 | **跳过下载 200GB BF16 中间步骤，直接产 FP8 自制版本**；1~2h 改 ckpt，1.5h 跑实验 |
| **P3** | **NVFP4** | 阶段 3 末 | 试 1 次能不能起；起得来再看速度；放最后做 |
| **P4** | Unsloth GGUF / APEX / ParoQuant / MLX 版 | 不做 | SGLang 不兼容 |
| **不做** | 0xSero/REAP-20 BF16 现成版本 | -- | **下载 200GB 浪费时间，跑出来大概率显存悖论拖累；可信结论已在 0xSero model card（97.9% capability 保留）+ 我们 P2 的 gsm8k/RULER 自测中得到** |
| **不做** | 自己跑 REAP 剪枝（路线 B：在 FP8 ckpt 上重新跑） | -- | plan 已就绪（路线 A），不需要路线 B |

**冯老师已 OK 模型层修改**，第二天上午直接执行 P2。

---

## 四、可观测性设计（分四层 + trace_id 对账）

### 4.1 分层架构

| 层 | 看到什么 | 实现 |
|---|---|---|
| **L0 客户端** `run_workload.py` | TTFT、round_latency、avg_session_time，**按维度分桶**：首轮 vs 后续轮、按 input token 桶、按 sysprompt id、按 session id | 改造 `run_workload.py` 内 Observer 接口；每 request 注入 `x-request-id` |
| **L1 our_proxy** | 入队/出队时间、in-flight 数演化、被限流的 request、按 trace_id 的 4 个时间戳（recv / admit / forward / first-token / done） | proxy 自带 logger，落 `proxy_metrics.jsonl` |
| **L2 sglang_router** | 转发统计（被路由到各 dp_rank 的次数、prefix match 率） | router 启动 `--prometheus-port` 拉取（如果当前版本支持） |
| **L3 SGLang server** | **cache_hit_rate、token_usage、num_running_reqs、num_queue_reqs、gen_throughput、TTFT/inter-token 直方图** | `--enable-metrics`，proxy 内置 scraper 每 1s 拉 `/metrics` |

### 4.2 SGLang `--enable-metrics` 提供的指标（已查官方文档确认）

| 维度 | 你需要的 | 官方指标名 |
|---|---|---|
| output token | 当前速率 | `sglang:gen_throughput` (gauge) |
| | 累积 | `sglang:generation_tokens_total` (counter) |
| cache | 命中率 | `sglang:cache_hit_rate` (gauge, 0~1) |
| | 命中 token 累积 | `sglang:cached_tokens_total` (counter) |
| 利用率 | KV 占比 | `sglang:token_usage` (gauge, 0~1) |
| | 已用绝对值 | `sglang:num_used_tokens` / `max_total_num_tokens` |
| 调度 | batch 大小 | `sglang:num_running_reqs` |
| | 排队 | `sglang:num_queue_reqs` |
| 延迟 | 直方图 | `sglang:time_to_first_token_seconds` / `inter_token_latency_seconds` / `e2e_request_latency_seconds` |

**结论：宏观维度官方完全够用**，先不 patch SGLang 加自定义 metric。

### 4.3 trace_id 对账机制（核心可观测设计）

```python
# L0 生成
trace_id = f"{session_id:04d}_{round_idx:03d}_{uuid8}"

# 三处落地：
# 1. HTTP header `x-request-id: <trace_id>` → SGLang 0.4+ 透传到日志
# 2. client jsonl 每条记录都带 trace_id
# 3. proxy 落盘的 jsonl 每条也带 trace_id 和 4 个时间戳
```

**对账输出**（`analysis/correlate.py`）：

```text
trace_id ─┬─ client: TTFT=4.2s, total=8.1s, ttft_pct_of_total=52%
          ├─ proxy:  recv=t0, admit=t0+0.3s (waited!), forward=t0+0.3s, first_token=t0+4.5s, done=t0+8.4s
          ├─ server: cached_tokens=143820/150000 (95.9% hit), dp_rank=5
          └─ server@t0..done window: cache_hit_rate_avg=0.91, token_usage_peak=0.78,
                                     num_running_reqs_avg=12.4, num_queue_reqs_peak=8

→ 归因：admit 等了 0.3s（proxy 限流），命中率 95.9% 良好，慢点在 prefill 阶段
```

这套对账机制对未来你们要在 SGLang 上加的"调度控制逻辑"是**同一套接口**——新调度层只是多注入一组时间戳。

---

## 五、our_proxy 设计

### 5.1 定位

- **不做路由**（路由给 sglang_router 干）
- **做准入/速率/反馈控制**：决定每个 request 现在能不能进系统、要不要排队
- **做可观测聚合**：trace_id、4 时间戳、SGLang metrics scraper

### 5.2 5 个 policy（可 `--policy` 切换）

| policy | 描述 | 控制旋钮 | 类比 / 来源 |
|---|---|---|---|
| `passthrough` | 不控制，直接转发 | 无 | 无 cong control（实验对照） |
| `simple_window` | 全局 in-flight **请求数**封顶 | `--max-inflight-reqs N` | TCP 固定 cwnd |
| `token_budget` | 全局 in-flight **prompt tokens** 封顶 | `--max-inflight-tokens M` | 加权 cwnd（避免 1 个 130k 长请求堵死） |
| **`length_aware`** | **短请求（< 阈值）走 passthrough；长请求才走 cache-aware 调度（送 router）** | `--length-threshold 2048` | **朋友提的核心策略**：因为 system_prompt 都一样，2k 以内没必要走 cache 调度；2k 以上前缀差异才显出来 |
| `session_aware` | 同 session 串行 + 跨 session 共享 token budget | `--max-inflight-tokens M --serialize-session` | 让 session 内相邻轮天然时间局部性最大化 cache 命中，避免抢占 |

可选第二阶段（如时间够）：

| policy | 描述 |
|---|---|
| `prom_feedback` | 拉 SGLang `/metrics` 的 `token_usage` / `num_running_reqs`，超阈值降 in-flight | 真正的 AIMD 拥塞控制 |

**`length_aware` 的实现备注**：

- 阈值 2k 是朋友给的初始值，可扫 1k / 2k / 4k / 8k
- 实现上：short request → 直接 POST 到 `:8000`（绕开 router）；long request → POST 到 `:30000`（走 router cache_aware）
- 设计动机：避免 5 个 system prompt 完全相同的短请求被 router 反复 hash 到不同 dp_rank，污染各 rank 的 radix tree
- **⚠️ 阈值看的是哪个 length？**
  - **不能看本轮新增 `input` 字段**（永远 1k~2k，绝大多数会落到 short 通道）
  - **要看累积 prompt 长度** = `sys + history + new_input`，proxy 端用 trace_id 关联同 session 历史即可重建
  - 朋友原意"system prompt 很多一样，要后面才看得出区别" → 第 1~3 轮（累积 < 阈值）走 passthrough 反正都命中 system prompt 的 cache；第 4 轮起累积 > 阈值，前缀差异化才出现，此时送 router 做 cache-aware 路由有意义
- 实现近似：client 在 header 里塞 `x-prompt-tokens-estimate: <int>`，proxy 直接读这个值，避免在 proxy 端再做 tokenizer（重复开销）

### 5.3 接口契约

```python
# observers/base.py
class Observer:
    def on_recv(self, trace_id: str, payload: dict): ...
    def on_admit(self, trace_id: str): ...
    def on_forward(self, trace_id: str, target_url: str): ...
    def on_first_token(self, trace_id: str): ...
    def on_done(self, trace_id: str, success: bool, usage: dict): ...
    def on_tick(self, t: float): ...   # 定期采样（如拉 SGLang /metrics）

# policies/base.py
class Policy:
    async def admit(self, trace_id: str, payload: dict) -> None:
        """阻塞直到可以发送（信号量/budget 同步）"""
    def release(self, trace_id: str, usage: dict) -> None:
        """完成后归还 budget"""
```

未来你们要加新调度逻辑时，新写一个 `Policy` 子类即可。

### 5.4 与 sglang_router 的边界

```
proxy 负责：
  ✓ trace_id 注入与时间戳
  ✓ 全局 in-flight req/token 计数与封顶
  ✓ per-session 串行队列
  ✓ SGLang /metrics scraper
  ✓ 落 proxy_metrics.jsonl

router 负责：
  ✓ cache_aware → 选 dp_rank
  ✓ 把 data_parallel_rank 写进请求体

两者职责正交，任何一层都可以单独换或者短路（实验时方便对照）
```

---

## 六、测试用例设计（6 个手写 fixture）

> 都按 `workload_data.jsonl` 同样 schema 写，每个 1~10 分钟跑完，每个为了"分离一个变量"设计。

| 文件 | sysprompt | input | output | wait | 轮×session | 区分什么 |
|---|---|---|---|---|---|---|
| **T1 sysprompt_pure** | 1 种 | 全 1k | 全 128 | 0 | 100 × 20 | 测纯 sysprompt+session 双前缀缓存上限。跑不快 = radix/lpm/router 坏了 |
| **T2 unique_sysprompt** | 50 种全不重复 | 1k | 128 | 0 | **20 × 50** | cache miss baseline。前缀缓存完全没用，用来作减法。**修正：单 session 至少 20 轮，才能让 cache miss 累积效应稳定** |
| **T3 burst_long_input** | 1 种 | 全 10k~16k | 128 | 0 | 5 × 20 | 测 chunked-prefill-size + 调度公平性，长 prefill 不能堵死短请求 |
| **T4 decode_heavy** | 5 种 | 1k | 全 1024 | 0 | 10 × 20 | decode 占比高 → spec on/off 在这里差异最大 |
| **T5 long_wait** | 5 种 | 1k | 256 | 全 60s | 20 × 20 | 测 KV 驻留 vs 驱逐 / HiCache 价值（H200 上预期差异小，留作对照） |
| **T6 official_50** | **按 sysprompt 分层抽**（每种 10 session × 5 种 = 50） | 原值 | 原值 | 原值 | 原值 | 对齐官方分布，最终对外结果出这个数。**修正：分层抽样而非随机抽，否则 sysprompt 比例可能严重失衡（5 种分布 23/21/20/19/16 抽 50 个，随机抽极端情况某种可能 0 个）** |

T1/T2 是"上限/下限"，T6 是"对外数字"，T3/T4/T5 是"分离变量"。

---

## 七、实验矩阵（分两阶段：架构选型 → 参数精调）

> 参数太多，必须分阶段。先选出最优的"架构 + proxy 组合"作为底盘，再在它上面做单参数扫描，否则 4 × 4 × 5 × 3 × 3 × ... 是组合爆炸。

### 7.1 阶段 1：架构选型（明天主跑，**约 6~8h**——重新估算后）

目的：回答"DPA 要不要、router 要不要、补偿要不要、proxy 要不要、spec 要不要"五个 yes/no 问题。

```
fixture       : T1, T6                              (2)
sglang_config : A, B-compensated, B-default, C      (4)  ← 比之前 +1
proxy_policy  : passthrough, session_aware          (2)
spec_decode   : on, off                             (2)
其它参数（所有 run 共用，作为默认）：
  KV cache: FP16 (auto)                            ← 改：不默认开 FP8 KV，留到阶段 2 单独跑精度+性能
  --mem-fraction-static 0.8
  --schedule-policy lpm
  --enable-mixed-chunk
  NEXTN steps=3 topk=1 draft=4
─────────────────────────────────────────────────
共: 2 × 4 × 2 × 2 = 32 个 run
```

**driver 复用策略（关键省时优化）**：同一 `sglang_config` 下连跑所有 `proxy_policy × fixture × spec` 组合，**server 不重启**，proxy 进程切换即可（proxy 切换 < 5s）。
- 每个 sglang_config 重启 1 次，4 次 server 重启 × 8min = 32min
- 32 个 run，每个 client 跑 T1 ~2min 或 T6 ~10min（avg ~6min）= 192min
- **总耗时 ≈ 32 + 192 + 30（其它开销） ≈ 4h** ← 跟之前估算回到接近

**关键约束**：所有 run 的 KV cache、mem-frac、mixed-chunk、policy 等参数一致；只切 sglang_config 的架构开关。

输出：选出 winner 架构 X*（架构 + proxy_policy + spec on/off 三元组）。

### 7.2 阶段 2：参数精调（基于阶段 1 winner，约 4h）

在阶段 1 winner 架构 X* 上做**单参数扫描**（OFAT，one-factor-at-a-time），每个 sweep 锁定 fixture 集合：

| sweep 名 | 参数 | 候选值 | fixture | run 数 |
|---|---|---|---|---|
| **S1 chunked** | `--chunked-prefill-size` (effective) | 4096 / 8192 / 16384 / 32768 | T3（burst long input）+ T6 | 8 |
| **S2 mem** | `--mem-fraction-static` | 0.7 / 0.8 / 0.85 / 0.9 | T1（cache 上限）+ T6 | 8 |
| **S3 spec_steps** | `--speculative-num-steps` | 1 / 3 / 5 / 7 | T4（decode heavy）+ T6 | 8 |
| **S4 schedule** | `--schedule-policy` | lpm / fcfs / lof | T1 + T6 | 6 |
| **S5 conservativeness** | `--schedule-conservativeness` (补偿后) | 0.5 / 1.0 / 2.0 / 3.0 | T6 | 4 |
| **S6 mixed_chunk** | `--enable-mixed-chunk` | on / off | T1 + T3 + T6 | 6 |
| **S7 hicache** | `--enable-hierarchical-cache` | on / off | T5（long wait）+ T6 | 4 |
| **S8 proxy** | proxy `--policy` | 全 4 个 + max_inflight 各 2 档 | T6 | 8 |

总计 52 个 run，平均 5min/run → ~4.5h。

### 7.3 阶段 3：精度+性能联合实验 + 补充论证（约 4~5h，按优先级排序）

> **重点**：用户明确要求所有"降精度"方案（KV FP8、剪枝、量化）都要先过精度回归。下面按"性价比 × 时间投入"排优先级。

| 优先级 | 实验 | 目的 | 时长估算 |
|---|---|---|---|
| **P0** | **KV FP8 e4m3 精度实验**：winner 配置基础上加 `--kv-cache-dtype fp8_e4m3` × gsm8k + **RULER-16k** | 决定能不能开 FP8 KV（**用户最关心**，开了 cache 容量翻倍） | gsm8k 25min + RULER 40min = ~1h |
| **P0** | **KV FP8 e4m3 性能实验**：winner + FP8 KV × T1 + T6（**第二天后用 `baseline-FP8-text-only` 替代 stock FP8 多 0.7GB KV pool**） | 看真实 workload 上 cache 命中率提升和 session_time 收益 | T1 2min + T6 10min = ~15min |
| **P1** | `--disable-radix-cache` × T1 | 证明前缀缓存量级贡献（应慢 5~10×），写报告用 | ~5min |
| **P1** | 配置 D（`--tp-size 4 --dp-size 2`）× T1 + T6 | dp_size 甜点（如果阶段 1 winner 是 B） | ~20min |
| **P2（第二天）** | **REAP-20 + FP8 自制 + 性能实验**：复用 `targeted_refusal_analysis.json` plan，在官方 FP8 ckpt 上按同样 expert_id 剪（+ 顺手删 visual.*），然后 gsm8k + RULER-16k 守门、过了进 T1+T6 workload 实验 | 真正拿到剪枝收益：weight ~99GB（vs FP8 baseline 120GB），KV pool 反而 +30GB；**跳过下载 200GB BF16 的中间步骤，直接出最终结果**；详见 3.9.4 路线 A | **1~2h 改 ckpt + 1.5h 跑实验 = ~3h** |
| **P3** | KV FP8 e5m2（仅当 e4m3 精度翻车才做；SGLang 不支持 INT8 KV） | 备选 KV 量化方案 | -- |
| **P3** | **NVFP4** 起不起得来 + 单 fixture 速度 | 用户说"放最后"。链接 2 警告 H200 没硬件加速，可能反而慢 | 起不起来 30min，跑速度 30min |
| **P4** | vLLM baseline × T6 | 对外汇报对比基线 | 30min（含 vLLM 启动） |
| **P4** | **LongBench v2 子集**（最终 winner 用 baseline FP16 KV vs FP8 KV vs REAP 三组对比） | 最终对外报告的 long-context 精度认证（跟 Qwen 官方 60.2 对照） | 跑 3 配置 × 1h = 3h |
| **P5（收尾，时间充裕）** | **SGLang 侵入式 patch 实测 vision 真实占用**：本地改 `server_args.py` + `qwen3_5.py` 让 `--language-only` 单机生效，stock FP8 ckpt 启服务对比 nvidia-smi | (a) 实测 vision encoder 真实占用是否 > 估算的 1.5GB；(b) 验证改 ckpt 方案是否在路上漏掉了什么；(c) 给 SGLang 主线留 PR 草稿 | ~2h（改+测+对比） |

**精度回归 benchmark 的选择**（用户提问）：

| benchmark | 速度（122B 上） | 任务性质 | 用法 |
|---|---|---|---|
| **gsm8k** | ~20-30 min（短任务，5 shot） | 数学短任务，不长 | 守门（短） |
| **RULER-16k 单 length** | ~30-45 min（合成 needle-in-haystack） | 合成长上下文 | **守门（长）首选**：单次最便宜的 long-context 测试 |
| **RULER 全套** (4k/8k/16k/32k/64k/128k) | 2-3 h | 长上下文全谱 | 阶段 3 末做 1 次（最终 winner 的全谱报告） |
| **LongBench v2 全套**（21 任务） | 3-4 h | 真实长文档 | 不做（太重） |
| **LongBench v2 子集**（5~6 个 reading/qa 任务） | ~1 h | 真实长文档 | 最终 winner 认证 + 跟 Qwen 官方 60.2 数字对比 |

**结论**：**RULER-16k 比 LongBench 子集快约 1.5×**（30-45min vs 60min），且单 length 已经能暴露 KV FP8 / REAP 的明显精度问题。**用 RULER-16k 做守门，LongBench v2 子集做最终认证**。

### 7.4 总实验预算

| 阶段 | run 数 | 预计耗时 |
|---|---|---|
| 阶段 1（架构选型） | **32** | ~4h（复用 server 后） |
| 阶段 2（参数精调） | 52 | ~4.5h |
| 阶段 3（精度+性能联合） | ~12 | ~5h（含 REAP-FP8 自制 ~3h + KV FP8 精度+性能 ~1.5h + 其它 ~0.5h） |
| **总计** | **~96** | **~13.5h**（≈ 第 1 天主跑阶段 1 ~4h + 第 2 天前半阶段 2 ~4.5h + 第 2 天后半阶段 3 ~5h） |

明天目标：跑完阶段 1。如果阶段 1 winner 很清晰，提前开始阶段 2 的 S1/S2/S3。

### 7.5 每个 run 的落盘格式

```
runs/<run_id>/
  ├─ config.yaml           ← {fixture, sglang_args, proxy_policy, ...}
  ├─ sglang_stdout.log     ← server 启动日志（含 spec accept 等）
  ├─ client_metrics.jsonl  ← 每 round 一行 + 末尾 summary
  ├─ proxy_metrics.jsonl   ← 每 request 一行（4 时间戳）+ 每秒 tick
  ├─ server_metrics.jsonl  ← 每秒一行 prometheus snapshot
  └─ summary.json          ← {run_id, avg_session_time, ttft_p50/90/99, ...}
```

---

## 八、自动化评估方法论（如何客观判定配置优劣）

> 你的核心提问。"扫一堆配置"不难，难在**机器自动得出"配置 X 比 Y 好"的可信结论**。这一节定义评估函数、失败判定、winner 决策规则、报告自动生成。

### 8.1 三类指标（按重要性排序）

| 类 | 指标 | 来源 | 用途 |
|---|---|---|---|
| **主指标（决定输赢）** | `avg_session_time` (秒) | client | 题目优化目标，**winner 由它决定** |
| **守门指标（不达标直接 disqualify）** | failed_rounds（失败率） | client | > 1% 视为不合格 |
| | **gsm8k loss（短任务）** | lm_eval | < 2% vs baseline FP8 + FP16 KV，**结果作废** |
| | **RULER-16k single length（长任务快速）** | lm_eval（用 [`OpenLab/RULER`](https://github.com/NVIDIA/RULER)） | < 3% vs baseline，**专防 KV 量化/REAP 在 long-context 翻车** |
| | **LongBench v2 子集（最终 winner 认证）** | 手跑 5~6 个 task | 跟 Qwen 官方 README 60.2 对照，差距 < 3% | 只对最终 winner 配置跑，不参与 sweep 决策 |
| | ttft_p99 | client | 超过 30s 视为用户体验崩，淘汰 |
| | OOM / server crash | driver | 直接淘汰 |
| **诊断指标（用来归因，不参与排名）** | TTFT p50/p90/p99 | client | 排队/prefill 慢 |
| | round_latency p50/p90/p99 | client | 端到端 |
| | output_throughput_tok_s | client | 系统总吞吐 |
| | request_throughput_req_s | client | 系统请求吞吐 |
| | cache_hit_rate（avg / p10） | server prom | cache 是否被打烂 |
| | token_usage（avg / peak） | server prom | KV pool 利用率 |
| | num_running_reqs（avg / peak） | server prom | 实际 batch 大小 |
| | num_queue_reqs（avg / peak） | server prom | 排队深度 |
| | spec_accept_length | server log | 投机解码 accept 长度 |
| | preempt_count（间接） | derived | cache_hit_rate 急降时刻的次数 |
| | proxy_wait_p99 | proxy | 准入控制等了多久 |
| | dp_rank 分布熵 | proxy | router 是否把请求均匀打散（low 熵 = 大量集中到几个 rank） |

### 8.2 配置失败的自动判定（driver 收完结果后自动 mark fail）

```python
GSM8K_REL_LOSS_MAX = 0.02     # 跟 baseline FP8+FP16 KV 比，最多劣化 2%
RULER_REL_LOSS_MAX = 0.03     # 跟 baseline 比，最多劣化 3%（专防长 context 翻车）

def is_failed(summary: dict, baseline: dict) -> tuple[bool, str]:
    if summary.get("error") == "OOM":
        return True, "OOM"
    if summary.get("error") == "server_crash":
        return True, "server_crash"
    if summary["failed_rounds"] / max(summary["successful_rounds"] + summary["failed_rounds"], 1) > 0.01:
        return True, f"fail_rate>1%"
    if summary["ttft"]["p99"] > 30:
        return True, f"ttft_p99={summary['ttft']['p99']:.1f}s"
    # 精度回归（仅对开启了降精度优化的配置必查）
    if (g := summary.get("gsm8k_acc")) is not None:
        if (baseline["gsm8k_acc"] - g) / baseline["gsm8k_acc"] > GSM8K_REL_LOSS_MAX:
            return True, f"gsm8k_drop={g:.3f}<base"
    if (r := summary.get("ruler16k_acc")) is not None:
        if (baseline["ruler16k_acc"] - r) / baseline["ruler16k_acc"] > RULER_REL_LOSS_MAX:
            return True, f"ruler16k_drop={r:.3f}<base"
    return False, ""
```

**精度回归的执行时机**（不是每个 run 都跑，太贵）：

| 配置类别 | gsm8k | RULER-16k | LongBench v2 |
|---|---|---|---|
| 阶段 1/2 的标准 SGLang 参数调优（不改权重/不改 KV） | ❌ 跳过（精度跟 baseline 等价） | ❌ 跳过 | ❌ 跳过 |
| 任何加了 `--kv-cache-dtype fp8_*` 或 `int8` 的配置 | ✅ 必跑 | ✅ 必跑 | 仅 winner |
| 任何换权重（REAP-20、NVFP4 等） | ✅ 必跑 | ✅ 必跑 | 仅 winner |
| 最终对外报告的 final winner | ✅ | ✅ 全谱 4k/16k/64k | ✅ 5~6 个 task |

### 8.3 winner 决策规则（自动选最佳配置）

```python
def rank_configs(runs: list[dict], fixture: str) -> list[dict]:
    """对同一 fixture 下的所有 run 排序，返回 winner 列表"""
    # 1. 守门：先剔除失败的
    qualified = [r for r in runs if r["fixture"] == fixture and not r["failed"]]
    
    # 2. 主指标：avg_session_time 升序（越小越好）
    qualified.sort(key=lambda r: r["avg_session_time"])
    
    # 3. 标记 winner：与最优值相差 < 3% 视为"并列 winner"（噪声范围）
    if not qualified:
        return []
    best = qualified[0]["avg_session_time"]
    winners = [r for r in qualified if (r["avg_session_time"] - best) / best < 0.03]
    return winners
```

**关于 3% 阈值**：单次跑的 noise 大约 1~3%（来自 GPU 时钟抖动、batch 切片随机性、网络抖动）。第二天如果有时间，会对前 3 个 winner 各重复 3 次取中位数，把 noise 降到 < 1%。

### 8.4 阶段间的传递规则

| 阶段 | 输入 | winner 判定 | 输出给下一阶段 |
|---|---|---|---|
| 阶段 1（架构选型） | 24 个 run | 对每个 fixture 排名，**T6 是最重要的**（对齐官方） | 选 T6 的 winner 架构 X*（sglang_config + proxy_policy + spec on/off） |
| 阶段 2（参数精调） | 在 X* 上 OFAT 扫每个参数 | 对每个 sweep S_i，T6 上排名第一的参数值 v_i* | 把所有 v_i* 拼成 X** |
| 阶段 3（补充） | X** vs 一些"理论上限/下限" | 对照实验，不参与 winner 选择 | 最终对外报告的"我们的配置 X**" |

### 8.5 自动报告生成

每次 sweep 跑完，`analysis/report.py` 自动产出：

```
report/<sweep_id>/
  ├─ summary.csv              ← 所有 run 的扁平表（fixture × config × 主指标 × 诊断指标）
  ├─ ranking.md               ← winner 排名（每 fixture 一段），含 disqualified 原因
  ├─ scatter_avg_vs_ttft.png  ← 散点图：x=avg_session_time, y=ttft_p99, 每点一个 config
  ├─ trace_id_slowest.csv     ← 最慢的 20 个 request 的 trace_id 对账（client × proxy × server 三方）
  └─ winner_diff.md           ← winner 与第二名的"诊断指标 diff"，告诉你为什么 winner 赢了
```

`ranking.md` 示例：

```markdown
## fixture: T6 (official_50)

| rank | config | avg_session_time | TTFT p99 | cache_hit_rate | status |
|------|--------|------------------|----------|----------------|--------|
| 1 (winner) | B+session_aware+spec=on | 142.3s | 4.1s | 0.91 | ok |
| 2          | B+passthrough+spec=on   | 145.8s | 6.3s | 0.88 | ok (within 3%) |
| 3          | A+session_aware+spec=on | 168.5s | 4.5s | 0.97 | ok |
| ...
| disqualified | A+passthrough+spec=on   | -      | -        | -              | ttft_p99=42s |
```

### 8.6 显著性 / 重复实验

| 场景 | 重复策略 |
|---|---|
| 阶段 1 主矩阵（24 run） | **跑 1 次**，3% 阈值容忍噪声 |
| 阶段 1 winner candidate 们（top 3） | **再重复 3 次取中位数**，确认 winner |
| 阶段 2 OFAT 扫描 | **跑 1 次**，看趋势 |
| 阶段 3 对外汇报数字 | **重复 5 次取中位数 + 报告 std** |

### 8.7 这一切如何串起来（driver/sweep.py 伪代码）

```python
for run_cfg in matrix.expand():
    # 1. 启动 server（A/B/C/D）
    server = start_sglang(run_cfg.sglang_args)
    wait_healthy(server, timeout=180)
    
    # 2. 启动 router（如果 cfg 要求）
    router = start_router(run_cfg) if run_cfg.use_router else None
    
    # 3. 启动 proxy
    proxy = start_proxy(run_cfg.proxy_policy, target=router or server)
    
    # 4. 跑 client
    client_result = run_client(run_cfg.fixture, target=proxy)
    
    # 5. 收齐三方 metrics → summary.json
    summary = build_summary(client_result, proxy.metrics, server.scraped_metrics)
    
    # 6. 守门 + 落盘
    summary["failed"], summary["fail_reason"] = is_failed(summary)
    save(run_cfg.run_id, summary)
    
    # 7. cleanup
    kill_all([proxy, router, server])
    wait_gpu_clean(timeout=30)

# 8. 跑完整批后，自动 ranking + 生成报告
runs = load_all_runs()
for fx in unique_fixtures(runs):
    winners = rank_configs(runs, fx)
    write_report(fx, winners)
```

**关键设计点**：driver 不需要懂 winner 是什么、为什么——它只负责忠实执行 + 落盘 + 调用 `rank_configs`。**ranking 逻辑写在一个独立函数里**，未来要改决策规则（比如加权指标）只改它一处。

---

## 九、文件结构

```
workload-suite/
├── PLAN.md                              ← 本文档
├── README.md                            ← 题目说明（已存在）
│
├── client/                              ← L0
│   ├── runner.py                        ← 改造自 run_workload.py
│   ├── trace.py                         ← trace_id 生成
│   ├── metrics.py                       ← 按维度分桶
│   └── observers/
│       ├── base.py
│       ├── client_only.py
│       └── cached_tokens.py             ← 从 response 读 cached_tokens
│
├── proxy/                               ← L1（our_proxy）
│   ├── main.py                          ← aiohttp server
│   ├── observers/
│   │   ├── base.py
│   │   ├── trace_logger.py
│   │   └── prom_scraper.py              ← 拉 SGLang /metrics
│   └── policies/
│       ├── base.py
│       ├── passthrough.py
│       ├── simple_window.py
│       ├── token_budget.py
│       ├── length_aware.py              ← 朋友提的核心策略
│       └── session_aware.py
│
├── serving/                             ← 服务端配置
│   ├── sglang/
│   │   ├── A_tp8.sh                     ← baseline
│   │   ├── B_dpa.sh                     ← DPA（配合 router）
│   │   ├── D_tp4_dp2.sh                 ← 可选
│   │   └── _common.sh                   ← 共用参数
│   ├── router/
│   │   └── start_router.sh              ← sglang_router 启动
│   └── vllm/                            ← 对比基线
│       └── ...
│
├── fixtures/                            ← 测试用例（手写）
│   ├── T1_sysprompt_pure.jsonl
│   ├── T2_unique_sysprompt.jsonl
│   ├── T3_burst_long_input.jsonl
│   ├── T4_decode_heavy.jsonl
│   ├── T5_long_wait.jsonl
│   └── T6_official_50.jsonl
│
├── driver/                              ← L4 扫参 driver
│   ├── sweep.py                         ← 串起 serving + proxy + client
│   ├── matrices/
│   │   ├── main.yaml                    ← 7.1 主矩阵
│   │   └── sub_*.yaml                   ← 7.2 子矩阵
│   └── lifecycle.py                     ← start/health-check/kill 服务
│
├── analysis/                            ← L1 对账与出图
│   ├── correlate.py                     ← trace_id join client × proxy × server
│   ├── report.py                        ← 出表格 / 图
│   └── notebooks/
│
└── runs/                                ← 实验结果（git ignore）
    └── <run_id>/
        ├── config.yaml
        ├── client_metrics.jsonl
        ├── proxy_metrics.jsonl
        ├── server_metrics.jsonl
        ├── sglang_stdout.log
        └── summary.json
```

---

## 十、时间表（明天 8h 工作日）

明天目标：**完成阶段 1（架构选型 32 个 run，复用 server 后实际约 4h）**，时间够则启动阶段 2 的 S1~S3。

> **driver 复用 server 是关键**：sglang_config 作为最外层循环（4 次启动 × 8min = 32min server overhead），proxy_policy/fixture/spec 在 server 不变的情况下切换（proxy 重启 < 5s）。**这一条把原本不可行的时间表救回来。**

| 时段 | 任务 | 产物 | 阻塞关系 |
|---|---|---|---|
| 09:00-09:30 | 用**现成 `sglang_serve.sh`**（已有的，本质就是配置 A）起 baseline 挂着 + 后台跑 `lm_eval gsm8k`（122B 跑一次约 20-30min） | SGLang 可用，基线 loss 写入 `baseline_loss.json` | 后台跑 |
| 09:00-09:30 | **同时**让 codex 后台搜：NVFP4 在 H200 起 server 的姿势（**REAP-20 我们已选定不下载 BF16，不用搜**） | `model_search.md` | 后台 |
| 09:30-12:00 | **改造 client**：抽 Observer 接口、加 trace_id、加 `x-prompt-tokens-estimate` header（给 length_aware 用）、metric 按维度分桶 | `client/runner.py` | 关键路径 |
| 同时 09:30-12:00 | **写 6 个 fixture**（注意 T2 改 20 轮，T6 分层抽样） | `fixtures/T1~T6.jsonl` | 并行（独立） |
| 同时 09:30-10:30 | **写 4 个 SGLang 启动脚本**（A、B-compensated、B-default、C），**KV cache 用默认 FP16**（FP8 KV 放阶段 3 单独跑精度回归） | `serving/sglang/*.sh` | 并行（独立） |
| 12:00-14:00 | **写 our_proxy**（~300 行 aiohttp + 5 policy 含 length_aware + prom scraper） | `proxy/main.py` | 关键路径 |
| 14:00-15:00 | 写 sglang_router 启动脚本 + 写 driver（**最外层是 sglang_config 循环，复用 server**） | `serving/router/`、`driver/sweep.py` | |
| 15:00-19:00 | **跑阶段 1 主矩阵 32 个 run**（先 4 个 smoke 验证 pipeline，再跑全量） | `runs/<id>/*` | |
| 同时 15:00-17:00 | 写 `analysis/correlate.py`（trace_id 三方 join、出对账表） | `analysis/` | 并行 |
| 19:00-19:30 | 出第一张"sglang_config × policy × fixture × avg_session_time"对比表 + 选出阶段 1 winner X* | `report/round1.md` | |
| **缓冲** | 启动 RULER-16k 长任务回归（夜跑 1h+）；启动阶段 2 的 S1（chunked）做夜间长跑 | | |

**关键路径**：client 改造（trace_id）→ proxy（依赖 trace_id）→ driver（依赖 proxy）→ sweep（依赖 driver）

**并行**：fixture 编写、serving 配置脚本、analysis 都不依赖客户端代码，可让另一人同时写。

**预算的不确定性与对策**：
- H200 加载 122B-FP8 实际可能 5~10min，按 8min 估
- 4 个 sglang_config 重启 = 32min；32 个 run × 平均 client 时间 6min = 192min；buffer 30min → **~4h，可控**
- gsm8k 一次 ~25min，作为**后台**跑不阻塞主流程
- 如果实际超时，**优先砍 spec=off 那组**（spec on 是默认推荐），把 32 个 run 砍到 16 个

---

## 十一、明天我们都不解决、但要记着的事

1. **vLLM 对比**：本方案完全围绕 SGLang，vLLM 留到第二天对比。vLLM 没有 sglang_router 那种 worker 级 cache-aware 路由（vLLM 自己有 prefix caching，但策略不同），proxy 那一层照样能挂上去做控制；metric 名不一样，proxy 需要适配。
2. **lm_eval 正确性回归**：题目说"阈值稍后发布"，先跑所有 winner 配置存 loss，等阈值出来回查。**回归执行时机详见 8.2 表格**：阶段 1/2 标准 SGLang 参数调优**不跑**精度（因为精度跟 baseline 等价）；只有加了 KV FP8 / 换权重（REAP/NVFP4）的配置才必须跑 gsm8k + RULER-16k。链接 2 提示量化对 long-context 下降明显，本 workload 正是 long-context，所以 RULER-16k 必跑。
3. **spec accept rate**：若官方 `/metrics` 没有，从 SGLang stdout grep `spec_accept_length`，或从 `usage.completion_tokens / num_decode_iters` 反推。
4. **抢占（preempt）次数**：用来证明"加调度后抢占减少"。若官方 metric 没单独 export，用 `cache_hit_rate` 的下降作为抢占的间接证据。
5. **未来你们要在 SGLang 上加的自定义调度逻辑**：放在 `proxy/policies/` 下新写一个 `Policy` 类即可，可观测、trace_id、对账都已经在框架里。
6. **PD disaggregation**：详见 3.8 节，记录但不进入主矩阵；如果阶段 1/2 跑完还有时间，可做 1 组对比验证我们不推荐的判断。
7. **REAP MoE 剪枝（性价比最高的潜在优化路线）**：第二天上午做 "REAP-20 + FP8 自制"——复用 0xSero pruning plan（用户已上传 `targeted_refusal_analysis.json`）在官方 FP8 ckpt 上同样剪掉 51 个 expert，1~2h 能搞定，不需要训练，**不需要下载 BF16 现成版本**。详见 3.9.4 路线 A。✅ 冯老师已确认 OK。
8. **SageAttention / 低精度 attention**：朋友提到，但 SGLang 主线还没支持。等版本支持后再说。
9. **通用数据集回归（不止 gsm8k）**：朋友提到"通用数据集试一下"。如果阶段 3 有空，加 mmlu / humaneval。**RULER/LongBench 优先级更高**（对应我们的 long-context 场景）。
10. **NVFP4（W4A4）**：链接 2 提到英伟达做了 Qwen3.5-122B NVFP4 版（234G → 75G），但**仅 Blackwell 有硬件加速，H200（Hopper）可能跑不动或更慢**。明天试一下能不能起来，能起就测速度，慢就放弃。
11. **Unsloth Dynamic / APEX GGUF**：4bit 高质量量化，但 **SGLang 对 Qwen3.5 MoE GGUF 支持不完整 + mixed-bit 限制**（链接 2 + SGLang 文档），先放弃，等 SGLang 后续版本。
12. **延迟 vs 吞吐之争**：README 明确"优化目标主要为 `avg_session_time`"（延迟），朋友聊天里说"看吞吐还是延迟，应该是吞吐"。我们的处理是：**主指标按 README 用 avg_session_time，但实现路径是提高 effective throughput（throughput × cache_hit_rate）**，因为此 workload 是高并发，吞吐高 = 排队短 = session_time 短，两者方向一致。
13. **🔧 SGLang 侵入式修改（收尾阶段，时间充裕才做）**：所有主线实验跑完后，如果还有 ≥ 2h，可尝试**改 SGLang 源码彻底跳过 vision encoder 实例化**，对照实测 vision module 在 H200 上的**真实**显存占用（我们之前的估算是 ~0.7-1.4GB，但 SGLang `models/qwen3_5.py` 的 `visual` module 可能有 buffer/cache 让实际占用更大）。
    - **改动点 1**：`server_args.py:3727` 取消 `--language-only` 单机模式的限制（允许 `encoder_urls 为空 + disaggregation_mode=null` 当作"本机只加载 LM"）
    - **改动点 2**：`models/qwen3_5.py:1467/1620` 在 `language_only=True` 时跳过 `self.visual = ...` 实例化，并 stub `self.deepstack_visual_indexes = []` 等下游引用
    - **验证步骤**：用 stock `Qwen/Qwen3.5-122B-A10B-FP8` 启 SGLang，对比 `--language-only` 开/关时的 `nvidia-smi` 显存差，跟我们改 ckpt 方案对比
    - **价值**：(a) 给 SGLang 主线提 PR；(b) 实测 vision 在 H200 真实占用（如果显著 > 1.5GB，说明改 ckpt 方案被低估了）；(c) 提供"不改 ckpt 也能省显存"的方案给后续工程
    - **不在 PR 周期内做**：用本地 patch 跑，不指望合入主线；纯粹是"额外验证 + 给未来留路"
14. **改 ckpt 方案的"未触达的优化"列表**（同步记录，避免遗忘）：
    - Qwen3.5 还有 MTP head 是否能保留 + 用 SGLang `--speculative-algo NEXTN`（应该已经支持，但验证一下）
    - 删 visual 后能否进一步把 `embed_tokens` / `lm_head` 的 padded vocab 256-pad 缩回（影响极小，不做）

---

## 十二、决策记录（这次讨论已经定下来的）

| # | 决策 | 理由 |
|---|---|---|
| 1 | 单机 8×H200 baseline 是配置 A 而不是方案 B | 错。已修正：方案 B（DPA+router）才是生产推荐，单机也成立 |
| 2 | 调度控制层放外置 proxy，不动 SGLang 内部 | 入口准入不该塞进 engine；fork 维护成本高 |
| 3 | 调度控制层**不**复用 sglang_router | router 是 Rust 二进制做 cache-aware 分发；我们的需求是流控，职责正交 |
| 4 | 用 SGLang 官方 `--enable-metrics`，不 patch | 官方覆盖了 output/cache/利用率三大维度；不够再说 |
| 5 | 测试用例先手写 6 个 fixture，不上 generator | 用户决策。后续可演化为参数化 generator |
| 6 | sweep 不带 radix off 作为主矩阵元素 | 用户决策。最后写报告时补 1 组证明性实验 |
| 7 | 加 in-flight 限流到 client | 用户提出：避免 cache 严重抢占；类比 TCP 流量控制 |
| 8 | 实验分两阶段：架构选型（阶段 1）→ 参数精调（阶段 2） | 参数太多组合爆炸，OFAT 单参数扫描更可解释 |
| 9 | **配置 B 必须显式补偿 DPA 静默修改的两个参数**（chunked × dp_size，conservativeness × 1/0.3） | 否则 A vs B 对比有隐藏变量。用户指出 chunked 这个坑后顺藤发现 conservativeness 同样被改 |
| 10 | 投机解码改成阶段 2 单参数扫描，主矩阵只对比 on/off | num_steps/topk 影响大但维度独立，OFAT 更干净 |
| 11 | H200 上 `--chunked-prefill-size` 默认 8192、`--cuda-graph-max-bs` 默认 512、`--mem-fraction-static` 自动 ~0.88，baseline 显式写 0.8 略保守 | 来自 server_args.py:1439-1448 H20/H200 分支 |
| 12 | 加 `--kv-cache-dtype fp8_e4m3` 作为**独立精度+性能实验**（不默认开） | 朋友提到，被遗漏。FP8 KV 让 cache 容量翻倍，但必须先过 long-context 精度回归（RULER-16k），过了才合并到 winner |
| 13 | 加 `length_aware` 作为第 5 个 proxy policy | 朋友提的核心策略："2k 以内走 passthrough，2k 以上才走 router cache_aware"，避免短请求污染 radix tree |
| 14 | 新增第八章"自动化评估方法论"，定义守门指标 / winner 决策规则 / 自动报告 | 用户提问。"扫一堆配置"不难，难在机器能自动出"X 比 Y 好"的可信结论 |
| 15 | **不**做 PD disaggregation 作为主架构（虽朋友提到） | 单机 8 GPU 切两组导致 KV 池腰斩；本 workload 是"高 cache 复用 + 长上下文"，PD 分离反受其害；记录到 3.8 节，备选 |
| 16 | HF 上能在 SGLang 跑的剪枝版本**只有** [`0xSero/Qwen3.5-122B-A10B-REAP-20`](https://huggingface.co/0xSero/Qwen3.5-122B-A10B-REAP-20) BF16；其余 5 个（REAP-30 GGUF、abliterated-MLX、Unsloth GGUF、MTP-GGUF、unsloth-GGUF）都因 SGLang 不支持 MoE GGUF / MLX 而淘汰 | 用户提供 6 个 HF 链接，逐一定性后筛选 |
| 16b | REAP-20 BF16 的**显存悖论**：BF16 权重 200GB > FP8 baseline 120GB，可能反而拖累 long-prompt workload | 详见 3.9.3 节计算；阶段 3 实测 |
| 16c | **REAP 是 zero-shot 无训练**（论文 arxiv 2510.13999 第 5 节确认），算法本质就是收集 `gate × activation_norm`、按 saliency 排序剪枝 | 用户提问 REAP 做法；论文确认无 fine-tuning |
| 16d | **REAP pruning plan 已拿到**（用户上传 `/Users/kausal/Downloads/targeted_refusal_analysis.json`，48 layer × 51 expert_id/层），元信息确认就是 0xSero/REAP-20 用的那份 | 详见 3.9.4 路线 A 步骤 |
| 16e | plan 是 **"targeted_refusal" 变体**（保护 refusal-dominant expert），不是纯 REAP。对 serving 性能完全无影响；反而保留模型对齐行为正常 | 详见 3.9.4 |
| 16f | **冯老师已确认 OK 允许模型层修改** | 用户在最新一轮告知 |
| 16g | **第二天上午一次性产 3 个自制 ckpt**：`baseline-FP8-text-only`、`REAP-20-FP8`（保留 visual 兜底）、`REAP-20-FP8-text-only`（最终主推），用 risk-isolated 启动顺序 7a→7b→7c | 详见 3.9.4 步骤；7b 翻车就退回 7a |
| 16h | **不 fork SGLang**（不改源码加 flag）；想省 ViT 显存通过改 ckpt 实现，self-contained | 详见 3.9.4b；fork 维护成本不值 ~1GB 收益 |
| 16i | **收尾阶段（时间充裕才做）**：尝试 SGLang 侵入式修改实测 vision 真实占用，对照改 ckpt 方案 | 用户提议；详见第十一章第 13 条。本地 patch 不合主线，纯验证 + 给未来留路 |
| 16j | **跳过 0xSero/REAP-20 BF16 现成版本下载和实测**，直接做 FP8 自制 | 用户洞察：200GB 下载浪费时间；显存悖论预判 BF16 在 long-prompt 场景会拖累，跑出来无新信息；plan 效果已在 0xSero model card 公布（97.9% capability 保留），我们 P2 跑 gsm8k/RULER 自测足以替代 sanity check |
| 17 | 主指标用 avg_session_time（按 README），不用吞吐 | 朋友说"应该是吞吐"，但 README 明确"优化目标主要为 avg_session_time"。两者方向一致：吞吐高 → 排队短 → session_time 短 |
| 18 | **配置 B 用两个变体**（B-compensated 抹平 DPA 静默修改 vs B-default 尊重 SGLang 默认） | SGLang 团队改 0.3 是有理由的（DPA 下 batch 小，激进调度更优）。原计划只跑 B-compensated 会得出错误结论 |
| 19 | **fixture T2 改 20 轮**（原 5 轮）、**T6 分层抽样**（原随机抽 50） | T2 5 轮不够 cache miss 累积；T6 随机抽容易让 sysprompt 比例失衡 |
| 20 | **`length_aware` 阈值看累积 prompt 长度而非本轮 input** | 看本轮 input 永远 1k 落 short 通道，length_aware 失效；要 client 在 header 传 `x-prompt-tokens-estimate` |
| 21 | **driver 最外层循环是 sglang_config**（复用 server） | 否则 4 个 config × 8 个变体 = 32 次启动，server 启动开销吃光时间预算 |
| 22 | **守门指标加 long-context 回归**（RULER-16k 或 LongBench） | 链接 2 警告：量化对 long-context 下降明显，而我们 workload 是 long-context（p50 ~50k）；只跑 gsm8k 不够 |
| 23 | **NVFP4 在 H200 上要实测**而不是直接采用，且**放阶段 3 最后做** | 用户明确要求"NVFP4 放最后，可能不支持"；链接 2 也说仅 Blackwell 有硬件加速 |
| 24 | **Unsloth GGUF / APEX GGUF / ParoQuant 全部不进主矩阵** | SGLang 对 mixed-bit / Qwen3.5 MoE GGUF 支持不完整（链接 2 + SGLang 文档），不值得花时间 |
| 25 | **KV FP8 / 剪枝 / 低精度 KV 都要做独立精度实验**，不能直接默认开 | 用户明确要求；阶段 3 用 RULER-16k 做守门，过线才合并到 winner |
| 26 | **long-context 守门用 RULER-16k**（30-45min），最终认证用 LongBench v2 子集（~1h） | 用户问哪个快，RULER 单 length 比 LongBench 子集快 1.5×；LongBench v2 可对照 Qwen 官方公布的 60.2 数字 |

---

## 十三、还需要你/朋友最终确认的几件事

1. **client 是否始终打 proxy（:8888）**？我建议是：proxy 再分发给 router(:30000) 或 SGLang(:8000) 或 vLLM(:8000)，这样切换实验只换 proxy 配置不动 client。
2. **配置 D（`--tp-size 4 --dp-size 2`）要不要进主矩阵**？主矩阵只 A/B-compensated/B-default/C 四个能控制时间，D 留作阶段 3 子矩阵。
3. ✅ **KV FP8 / 剪枝 / 低精度 KV 都做独立精度实验**（用户已决策）：详见 3.9.6 表格 + 7.3 阶段 3 + 8.2 守门规则。
4. **`gsm8k loss 阈值` 题目说"稍后发布"**——明天先记录所有配置的 loss，等阈值出来回查。我们守门默认用"相对 baseline 劣化 2%"（gsm8k）和 3%（RULER-16k），等阈值出来回查。是否 OK？
5. ✅ **long-context 守门用 RULER-16k**（用户已决策），最终 winner 跑 LongBench v2 子集对照 Qwen 官方 60.2。
6. ✅ **冯老师已确认 OK 允许模型层修改**。第二天上午执行路线 A，一次性产 3 个 ckpt 详见 3.9.4。
7. **冯老师那边对"优化目标"是否还有别的口径**（比如同时看 throughput SLA）？还是就 README 写的 `avg_session_time` 单指标？
8. ✅ **NVFP4 放阶段 3 最后**（用户已决策）。明天上午让 codex 后台搜 H200 上 NVFP4 起 server 的姿势，能起就单 fixture 测速度，慢就放弃。
9. ✅ **REAP-20 BF16 阶段 3 跑（用户确认），且第二天追加 REAP-20 + FP8 自制**。可行性详见 3.9.4 路线 A：
   - 上午先看 0xSero 仓库是否带 `pruning-plan.json`；没有的话对比 router weight 反推
   - 写 ~50 行 python 在官方 FP8 ckpt 上按同样 expert_id 删 weights 和 router 行
   - 导出新 safetensors，SGLang 加载，跑 RULER-16k 守门
   - **如果过线**：weight 99GB（vs FP8 baseline 120GB），KV pool 反而 +30GB，**性价比最高的优化**
