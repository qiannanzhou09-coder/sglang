# 第三章 Serving 参数测试优先级说明

> 本文档整理 `PLAN.md` 第三章：哪些服务端架构和参数需要测、为什么要测、按什么优先级测。

核心原则：

1. 先测 **架构是否正确**：DPA 是否有收益，router 是否保住 cache 命中。
2. 再测 **KV cache / 长 prefill / 调度参数**：这是本 workload 的主要矛盾。
3. 最后测 **会带来精度风险的模型层优化**：KV FP8、REAP、NVFP4 等必须先过精度守门。

本 workload 的关键特征是：每轮请求都会带上完整历史，后期 prompt 可达 100k+ tokens。因此，服务端优化的第一目标不是单纯提高单轮吞吐，而是让长上下文前缀缓存尽可能命中，同时避免超长 prefill 阻塞 decode。

---

## P0：架构选型，必须最先测

这一组决定生产底盘。没有先跑清楚 A/B/C，后续参数扫描没有意义。

| 优先级 | 要测什么 | 怎么测 | 为什么测 | 主要指标 |
|---|---|---|---|---|
| 1 | A：纯 TP8 baseline | `--tp-size 8`，client 直接打 SGLang | 建立基准线。纯 TP 共享一个 KV pool，cache 命中天然稳定，但 TP all-reduce 开销较大。 | `avg_session_time`、TTFT、cache hit |
| 2 | B-default：DPA + router，尊重 SGLang 默认 | `--tp-size 8 --dp-size 8 --enable-dp-attention`，入口走 `sglang_router --dp-aware --policy cache_aware`，不补偿 DPA 静默修改 | 这是最接近 SGLang 官方默认意图的 DPA 方案。DPA 下 effective chunked 会变小，调度会更激进，可能更适合小 batch。 | `avg_session_time`、cache hit、running reqs |
| 3 | B-compensated：DPA + router，补偿静默修改 | `--chunked-prefill-size 65536`，`--schedule-conservativeness 3.33`，使 DPA 后 effective 值接近 A | 用于公平比较 DPA 本身收益。如果不补偿，A 和 B 实际参数不同，结论不干净。 | 同上 |
| 4 | C：DPA 裸跑，不挂 router | SGLang 同 B，但 client 直接打 `:8000` | 证明 router 的价值。预期 C 会把同 session 请求分散到不同 dp rank，前缀 cache 命中率下降。 | cache hit、avg_session_time |
| 5 | B vs C 的 router 价值 | 对比 `cache_hit_rate`、`cached_tokens`、`avg_session_time` | 本 workload 的核心是前缀缓存。如果 router 没有提升 cache 命中，架构假设需要重审。 | cache hit 差异 |

推荐 fixture：

| fixture | 用途 |
|---|---|
| T1 `sysprompt_pure` | 测 prefix cache 上限，验证 radix/lpm/router 是否工作正常。 |
| T6 `official_50` | 对齐真实 workload 分布，作为阶段 1 主要决策依据。 |

阶段 1 要回答三个问题：

1. DPA + router 是否优于纯 TP8？
2. B-default 和 B-compensated 哪个更好？
3. DPA 裸跑是否因为 cache 命中下降而明显变差？

---

## P1：KV cache、长 prefill、调度参数

架构 winner 选出来之后，优先调这组。原因是本 workload 的主要瓶颈来自长 prompt、KV 驻留、前缀复用和调度公平性。

| 优先级 | 要测什么 | 候选值 | 为什么测 | 推荐 fixture |
|---|---|---|---|---|
| 6 | `--mem-fraction-static` | `0.8 / 0.88 / 0.92 / 0.95` | KV pool 越大，长 session 前缀越不容易被驱逐；但太高可能 OOM。 | T1 + T6 |
| 7 | `--chunked-prefill-size` | effective `4096 / 8192 / 16384 / 32768` | 有少量 10k+ 到 130k 的长 input。chunk 太大可能堵 decode，太小会降低 prefill 吞吐。 | T3 + T6 |
| 8 | `--schedule-conservativeness` | effective `0.3 / 1.0 / 2.0 / 3.0` | 控制调度激进程度。太激进可能带来抢占和 cache 抖动，太保守可能排队变长。 | T6 |
| 9 | `--enable-mixed-chunk` | on / off | prefill 和 decode 是否混合执行会影响长 prompt 下的整体吞吐和 TTFT。 | T1 + T3 + T6 |
| 10 | `--schedule-policy` | `lpm / fcfs / lof` | `lpm` 理论上最适合 prefix cache，但需要用实测证明。 | T1 + T6 |

这组要回答的问题：

1. KV pool 需要开到多大才足够？
2. 长 prefill 的 chunk 大小是否会影响 TTFT 和 session time？
3. DPA 下激进调度和保守调度谁更适合？
4. `lpm` 是否确实优于普通 FCFS？

---

## P2：投机解码

投机解码重要，但不应该排在架构和 cache 参数之前。原因是这个 workload 的第一矛盾是长上下文 cache 命中，不是纯 decode。

| 优先级 | 要测什么 | 候选值 | 为什么测 | 推荐 fixture |
|---|---|---|---|---|
| 11 | spec on/off | NEXTN on / off | 先确认投机解码对本 workload 是否真有收益。 | T4 + T6 |
| 12 | `--speculative-num-steps` | `1 / 3 / 5 / 7` | decode-heavy 时 steps 大可能更快；高并发时 verify 开销可能抵消收益。 | T4 + T6 |
| 13 | `--speculative-num-draft-tokens` | 通常设为 `steps + 1` | draft 太少收益小，太多浪费。 | T4 |
| 14 | `--speculative-eagle-topk` | 先固定 `1`，必要时测 `2 / 4` | `topk > 1` 通常 verify 开销更大，优先级低。 | T4 |

这组要回答的问题：

1. NEXTN 在 T6 上是否降低 `avg_session_time`？
2. decode-heavy 场景下 `num_steps=3` 是否是最佳值？
3. 大 batch 或 cache 命中很高时，spec 是否反而拖慢？

---

## P3：KV FP8，潜在收益大，但必须先做精度守门

KV FP8 是第三章里最值得认真测的低精度优化。它可能让 KV cache 容量近似翻倍，从而减少长 session 的 cache eviction。但它直接影响 long-context 精度，不能默认开启。

| 优先级 | 要测什么 | 怎么测 | 为什么测 |
|---|---|---|---|
| 15 | `--kv-cache-dtype fp8_e4m3` 精度 | 先跑 gsm8k + RULER-16k | e4m3 是主推格式，但 long-context 精度可能下降。 |
| 16 | FP8 KV 性能 | 精度过线后，在 winner 配置上跑 T1 + T6 | 验证 KV 容量变大是否真的降低 `avg_session_time`。 |
| 17 | `--kv-cache-dtype fp8_e5m2` | 仅当 e4m3 精度不理想时测 | 备选格式，不应该优先测。 |

守门规则：

| benchmark | 用途 |
|---|---|
| gsm8k | 短任务精度守门，确保基本能力不掉。 |
| RULER-16k | 长上下文快速守门，专门防止 KV 量化在长 prompt 场景翻车。 |
| LongBench v2 子集 | 只给最终 winner 做认证，不参与每个 sweep。 |

结论规则：

- 如果 RULER-16k 明显下降，FP8 KV 不能进最终配置。
- 如果精度过线，再看 T1/T6 上是否提升 `avg_session_time`。
- 不允许只凭 T1/T6 性能收益直接打开 FP8 KV。

---

## P4：HiCache，长 wait 场景验证

HiCache 不是主线优化，但需要用 T5 验证。原因是 workload 里 session 之间有 wait，p99 接近 60s，KV 在等待期间可能被驱逐。

| 优先级 | 要测什么 | 候选值 | 为什么测 | 推荐 fixture |
|---|---|---|---|---|
| 18 | `--enable-hierarchical-cache` | on / off | 验证 CPU KV cache 对长 wait 是否有收益。 | T5 + T6 |
| 19 | `--hicache-ratio` | `2 / 4 / 8` | 如果 HiCache on 有收益，再调 CPU cache 容量。 | T5 |
| 20 | HiCache backend / policy | `kernel/direct`、`wait_complete/best_effort`、`write_through/write_back` | 只有 HiCache 明显有效时才继续测这些细项。 | T5 |

预期判断：

- H200 显存较大，活跃 session 数不高时，HiCache 可能收益小。
- 但如果 session_aware 或真实并发导致 KV pool 压力变大，HiCache 仍可能有价值。

---

## P5：MoE / EP / DeepEP

这组属于通信和 MoE kernel 层优化，有潜在收益，但应该放在 cache/router/chunked 之后。

| 优先级 | 要测什么 | 为什么测 |
|---|---|---|
| 21 | `--enable-dp-lm-head` | DPA 下减少 LM head 通信，成本低，建议和 B 一起测。 |
| 22 | `--ep-size` / `--moe-dp-size` | Qwen3.5 是 MoE，expert 并行度可能影响吞吐和通信。 |
| 23 | `--enable-deepep-moe` | 可能优化 MoE all-to-all，但复杂度更高。 |

这组不进入第一轮主矩阵，原因是解释性弱：如果 cache 和 router 没有先定下来，MoE 参数的效果很难归因。

---

## P6：attention backend、torch compile、CUDA graph

这组是 winner 配置上的性能补充，不参与早期架构判断。

| 优先级 | 要测什么 | 候选值 | 为什么测 |
|---|---|---|---|
| 24 | `--attention-backend` | `flashinfer / fa3` | H200 默认可能是 flashinfer，但 fa3 值得单独试一次。 |
| 25 | `--enable-torch-compile` | on / off | 可能提升 decode，但会影响 piecewise CUDA graph。 |
| 26 | `--cuda-graph-max-bs` | `256 / 512 / 1024` | 影响 decode latency 和显存占用，放后面测。 |

这组的原则是：只在最终架构比较稳定后做，不要让它污染阶段 1 的结论。

---

## P7：模型层优化，必须和精度实验绑定

这些会改变模型权重或数值精度，不能只看 serving 性能。必须先过精度回归，再跑 workload。

| 优先级 | 要测什么 | 为什么测 | 必须先跑 |
|---|---|---|---|
| 27 | `baseline-FP8-text-only` | 删除 visual 权重，验证 SGLang 能否正常加载，顺手省一点显存。 | gsm8k |
| 28 | `REAP-20-FP8` 自制 | 剪掉部分 MoE expert，理论上减少权重和 FLOPs，同时比 BF16 现成版更省显存。 | gsm8k + RULER-16k |
| 29 | `REAP-20-FP8-text-only` | 剪 expert + 删除 visual，是最终模型层候选。 | gsm8k + RULER-16k |
| 30 | NVFP4 | 放最后试能不能起。H200 没有 Blackwell FP4 原生加速，可能不快。 | gsm8k，必要时 RULER-16k |

不建议测：

| 项目 | 原因 |
|---|---|
| 0xSero REAP-20 BF16 现成版 | BF16 权重比当前官方 FP8 baseline 更占显存，长上下文场景可能因为 KV pool 变小而拖累。 |
| GGUF / MLX / Unsloth Dynamic | SGLang 对 Qwen3.5 MoE GGUF / mixed-bit 支持不可靠。 |
| APEX / ParoQuant mixed-bit | SGLang mixed-bit quantization 限制较多，风险高。 |

---

## 反面教材和补充验证

这些不进主矩阵，但建议至少做一次，用于报告和归因。

| 要测什么 | 怎么测 | 目的 |
|---|---|---|
| `--disable-radix-cache` | winner 配置上关 radix cache，跑 T1 | 证明前缀缓存对本 workload 的量级贡献。 |
| PD disaggregation | 有时间再跑一组 | 验证为什么本题不推荐 PD。单机 8 GPU 切 prefill/decode 会切小 KV pool。 |
| vLLM baseline | 最终阶段跑 T6 | 对外报告时提供另一个 serving 系统基线。 |

---

## 推荐执行顺序

按下面顺序跑，最容易快速得到可靠结论。

1. A：纯 TP8 baseline。
2. B-default：DPA + router，尊重 SGLang 默认。
3. B-compensated：DPA + router，补偿 DPA 静默修改。
4. C：DPA 裸跑，不挂 router。
5. 用 T1 + T6 选出阶段 1 架构 winner。
6. 在 winner 上扫 `mem-fraction-static`。
7. 扫 `chunked-prefill-size`。
8. 扫 `schedule-conservativeness`。
9. 扫 `--enable-mixed-chunk`。
10. 扫 `schedule-policy`。
11. 测 spec on/off，再扫 `speculative-num-steps`。
12. FP8 KV：先 gsm8k + RULER-16k，过线后跑 T1 + T6。
13. HiCache：用 T5 + T6 验证是否值得继续调。
14. REAP-20-FP8：先精度，后 T1 + T6 性能。
15. NVFP4：最后试能否启动和单 fixture 速度。

最重要的前三个结论：

1. **DPA + router 是否优于纯 TP8？**
2. **B-default 和 B-compensated 哪个更适合这个 workload？**
3. **FP8 KV 是否能在 long-context 精度不过线的前提下带来性能收益？**

---

## 每类实验重点看什么指标

| 实验类型 | 主指标 | 诊断指标 |
|---|---|---|
| 架构选型 | `avg_session_time` | cache hit rate、TTFT p99、num running reqs、token usage |
| cache / memory | `avg_session_time` | cache hit rate、token usage peak、num used tokens |
| chunked prefill | TTFT、`avg_session_time` | prefill 阶段耗时、queue depth、long request tail latency |
| 调度策略 | `avg_session_time`、TTFT p99 | cache hit rate、proxy wait、queue depth |
| spec decode | round latency、output throughput | accept length、decode throughput |
| FP8 KV / REAP / NVFP4 | 先看精度，再看 `avg_session_time` | gsm8k、RULER-16k、LongBench v2 子集 |

最终 winner 选择规则：

1. 先剔除失败配置：OOM、server crash、失败率超过 1%、TTFT p99 超过阈值、精度回归不过线。
2. 剩余配置按 `avg_session_time` 排序。
3. 与最优值差距小于 3% 的配置视为并列 winner，需要重复跑取中位数。
