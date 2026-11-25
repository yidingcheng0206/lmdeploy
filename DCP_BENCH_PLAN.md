# DCP 优化分阶段收益实验方案

> 配套文档：`DCP_PEERPULL_DESIGN.md`（Phase 2 设计）
> 硬件：单机 8×H100/H800/H200（Hopper + NVSwitch），同一节点、同一驱动
> 目标：把 ag_rs → a2a → q复制 → symm-mem ag → PeerPull 每一步的收益单独、
> 可信地测出来，为 auto 默认策略和 q 复制的去留提供数据。

---

## 0. 总原则

1. **一个二进制，旗标回放所有阶段**。所有优化均为运行时开关，基准只换旗标不换构建：

| 实验编号 | 配置（旗标） | 对应阶段 |
|---|---|---|
| C0 | `--dcp 1`（纯 TP） | 参照系 |
| C1 | `--dcp 4 --dcp-comm-backend ag_rs` | main 现状 |
| C2 | `--dcp 4 --dcp-comm-backend a2a` | 本分支现状 |
| C3 | C2 + `--dcp-replicate-q-proj` | q 复制 |
| C4 | `--dcp 4 --dcp-comm-backend ag_rs` + symm-mem ag 生效 | Phase 1（#4995） |
| C5 | C4 语义 + `--dcp-comm-backend pull` | Phase 2（PeerPull） |
| C6 | C5 + sparse 直写 arena | Phase 2.5（可选） |

   未实现的阶段（C4/C5/C6）行位预留，落地后补测——harness 现在就按最终旗标面构建。
2. **三层测量隔离归因**：微基准（纯通信段）→ 单层算子（nsys 分段）→ 端到端
   （TPOT/吞吐）。任何端到端异常都要能下钻到层内段落。
3. **生产形态测量**：decode 全程 CUDA Graph 开启，测 replay 而非 eager；
   前缀缓存关闭（控制方差）；EOS 关闭、定长生成。

## 1. E0：环境与噪声地板（半天，最先做）

- 锁频：`nvidia-smi -lgc <boost-clock>`（所有卡），记录 `nvidia-smi -q -d CLOCK`
- 固定环境变量：`NCCL_DEBUG=INFO` 只在冒烟时开，正式测量关掉；同一 `CUDA_VISIBLE_DEVICES`
- 噪声地板：C1 配置下连跑 3 组 × (20 warmup + 100 测量) step，组间重启进程。
  计算 step 时间的组间漂移 σ——**后续所有 < 2σ（经验上 <2%）的差值一律视为无差异**
- 记录环境块（驱动/NCCL/torch/CUDA 版本、卡型号、频宽）进报告头

## 2. E1：通信微基准（归因核心，1-2 天搭）

**目的**：单独测 merge 段与 q 侧通信的成本，不跑 attention。

**Harness**：复用 `tests/pytorch/test_dcp_communication.py` 的 mp.spawn 框架，
合成张量（真实形状），CUDA events 计时，graph 捕获后测 replay：

```
每配置: warmup 20 → capture graph → replay 100 次取 p50/p99
计时范围 = merge 段（或 q gather 段）单层成本
```

**矩阵**（核心，~30 组，每组 <1 分钟）：

| 维度 | 取值 |
|---|---|
| backend | ag_rs / a2a / pull（+ 各自的 q 侧：NCCL ag / symm-mem ag / 复制=无通信） |
| T（query 行数） | 1, 8, 32, 128, 512（对应 batch × 每 token 数；MTP 时 T=B×q） |
| dcp | 2, 4（可选 8：`--tp 8 --dcp 8` 纯 DCP，对齐 RTP-LLM 文章口径） |
| dtype | bf16（fp8 KV 不影响 wire 格式，跳过） |

**预期曲线形态**（用于校验实现正确性）：
- NCCL 类：小 T 平坦（固定延迟主导 ~15-30μs），大 T 线性上升
- pull：小 T ~5μs 级，大 T 斜率 ≈ 裸 NVLink 带宽（先单测裸 pull 带宽做上限参照）
- **理论校验点**：merge 段成本与 KV 长度无关（文章推论 2）——扫一组
  KV∈{8K, 128K} 确认平坦，不平坦说明实现有额外读写
- 顺带测：裸 TMA pull 带宽上限（同字节量，无归并）→ pull 的"归并开销"应 <2μs

**产出表**：`backend × T × dcp → p50/p99 μs`，后续所有讨论的数据底座。

## 3. E2：单层算子基准（含 attention，nsys 分段）

**目的**：真实 kernel 链路下验证 E1 的归因，并测 q 复制的计算侧成本。

- 模型：DeepSeek-V3（dense MLA，bf16 KV）主；V3.2（sparse）次
- 链路：`q_proj(+gather/复制) → attention → merge 段 → o_proj`
- nsys + NVTX 标注各段，取 decode step 内 61 层的平均段时长（最慢 rank 口径）
- **关键对比**：
  - C2 vs C4：q gather 段 NCCL vs symm-mem（验证 ~20→~5μs）
  - C2 vs C5：merge 段三 kernel vs 单 kernel + staging copy
  - C3 vs C2：q 侧段（gather 消失）vs Q GEMM 段（×P 增大）——**拿到 q 复制的
    盈亏平衡 batch**：小 batch 省通信、大 batch GEMM 反超，找交叉点
- KV 长度：{8K, 32K, 128K}（attention 段随之变化，merge 段应不变——再次验证）

## 4. E3：端到端基准（用户可感知收益）

**配置**：lmdeploy 离线推理，定长输入/输出，prefix cache off，CUDA graph on，
batch 由客户端并发控制。

**矩阵**（核心 12 组 + 可选扩展）：

| 维度 | 取值 |
|---|---|
| 配置 | C0, C1, C2, C3, C4, C5（+C6 若做） |
| batch | 8（延迟敏感）, 64, 256（吞吐） |
| 输入/输出 | 8K/512 与 128K/512 两个长度点（输出固定 512，EOS off） |
| 模型 | V3；扩展项：V3.2 sparse、GLM DSA、fp8 KV |

**指标**：
- 主指标：**TPOT**（p50/p99）、吞吐（token/s = ΣB/ΣTPOT）
- 校验指标：TTFT（应不受 decode 优化影响，若变说明有意外耦合）
- 报告口径：每配置 3 次独立运行取均值，标注 E0 的噪声地板

**预期幅度核对**（来自 E1/E2 估算，用于 sanity）：C5 vs C2 在 B=8/8K 应省
~1.5-3.5ms/step（61 层 × 每层省 25-60μs）；若端到端测不出对应差值，
回 E2 找被掩盖的段落（常见嫌疑：入口等最慢 rank、graph 边界）。

## 5. E4：q 复制专项决策实验

E2 已给出计算侧交叉点，E4 补齐决策的另外两块数据：

1. **内存 → 容量换算**：实测 replica GB（torch.cuda.memory_snapshot 前后差），
   按目标模型 KV 字节/token 换算成"牺牲的 KV token 数"，除以 rank 总 KV 预算得百分比
2. **fallback 价值验证**：人为禁用 symm-mem（环境变量/模拟跨机）使 auto 落回
   NCCL a2a，重测 C3 vs C2——确认"回退场景下 q 复制仍省 ~20μs/层"的论断
3. **产出**：一张三行决策表（小 batch 延迟 / 大 batch 吞吐 / fallback 场景），
   直接回答"q 复制保留为什么定位、文档怎么写"

## 6. E5：数值门禁（每次配置变更重跑）

1. **后端等价性**：8 卡 harness 内 C1/C2/C5 三种 backend 对同一合成输入的输出
   逐元素对比（复用 `test_dcp_a2a.py` 的 NaN/±inf/空行矩阵；pull vs a2a 容差
   与 a2a vs reference 同档）
2. **GSM8K**：C2 与 C5 各一整跑（同 seed、同解码参数），对齐 ±0.5 点验收线
3. C3 单独一跑（GEMM 路径变化）

## 7. E6：内存核算

| 项 | 方法 | 预期 |
|---|---|---|
| merge workspace | C1/C2/C5 各测 `torch.cuda.max_memory_allocated` 差 + NVML 进程峰值 | C5 ≤ C2（send/recv buffer 消失，arena 常驻）；验证 arena 跨层共享后不再 ×层数 |
| q 复制权重 | memory_snapshot 差 | V3 ~4GB/rank 量级核对 |
| KV 容量损益 | (可用 KV 预算变化) ÷ 字节/token | DCP 增益 vs 复制损失净额 |

## 8. E7：CUDA Graph 行为专项

- graph 捕获 → 连续 replay 500 步（数据每步变化），输出正确性抽查
- 容量增长触发（先小 batch 再大 batch bucket）后旧 graph 重放仍正确
  （arena 地址保活、epoch 单调推进）
- pull 与 fallback 切换发生在两次 capture 之间，不跨 graph

## 9. 验收门（Gate）

| Gate | 条件 | 决策 |
|---|---|---|
| G1 | E1：pull 在 T≤32 时 ≤ a2a 的 50% | 否则查 kernel 实现，不切默认 |
| G2 | E3：C5 在 B=8/8K 的 TPOT 改善 ≥5% 或 ≥0.5ms/step | auto 默认是否切 pull |
| G3 | E5：等价性 + GSM8K ±0.5 点全过 | 任何合并/发布的前置 |
| G4 | E6：workspace 超预算或 ×层数未消除 | 回修 arena manager |
| G5 | E4：大 batch 处 C3 比 C2 慢（GEMM 反超）确认 | q 复制文档降级为小 batch/fallback 选项 |

## 10. 执行顺序与工期

| 周 | 内容 |
|---|---|
| W1 | E0 + E1 harness 搭建；先测已有配置（C1/C2/C3 的通信段）→ **立即得到本分支 a2a+q复制的真实收益** |
| W1-2 | E2/E3 跑通已有配置（C0-C3）；E5 数值门禁随行 |
| W2+ | Phase 1 落地 → 补 C4 行；Phase 2 落地 → 补 C5 行 + E7 + G1/G2 决策 |
| 随行 | E4 在 C5 数据齐后出 q 复制决策表 |

**先跑哪组最有价值**：E1 的 C1/C2/C3 三行——不动任何新代码就能回答
"我的 a2a 和 q 复制现在到底值多少 μs"，并为后续所有阶段的对比定锚点。

## 11. 报告模板（每实验一节）

```
## E1 通信微基准
环境块（E0 输出）
表：backend × T × dcp → p50/p99 (μs)，附噪声地板
图：T 扫描曲线（每 backend 一条），标注固定延迟区/带宽区
异常与归因备注
```

## 12. 2026-11-19：新增精度验证目标与 2A 接入约束

用户要求后续使用 **Qwen3-235B-A22B** 验证下列三组配置。先记录原始目标；**尚未执行，当前存在模型及 KV 分组前置缺口**。本条是新增要求，不是已完成的评测结果。

| 用户指定配置 | 当前解析后的拓扑 | 执行前置条件 |
|---|---|---|
| TP8 + DCP2 | world_size=8，attn_tp=8，DP1/EP1 | 需要先实现 GQA DCP；该模型原始 KV heads=4，TP8 时每个 KV head 有两个副本，适合在副本组内做 DCP2 |
| TP8 + DCP2 + DP2 + EP8 | world_size=8，attn_tp=4，mlp_tp=8，moe_tp=1 | 除 GQA DCP 外，还需解决 KV 分组：attn_tp=4 时四个 KV heads 不再复制，不能直接沿用“在同一 KV head 副本内分片”的 DCP2 路径；需要另行设计 KV/head 布局或调整拓扑 |
| TP8 + DCP2 + DP2 + EP8 + MTP | target 拓扑同上一行；draft 参数待明确 | 上述前置条件之外，当前没有 Qwen3-235B-A22B 的原生 MTP 接入；需先明确可用 draft 权重、方法与支持实现 |

兼容性依据：

- [官方模型配置](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json)为 `Qwen3MoeForCausalLM`，64 个 Q heads、4 个 KV heads，属于 GQA。当前 `lmdeploy/pytorch/config.py` 的模型校验仍要求 `use_flash_mla`；第一阶段只吸收 PR #4995 的基础架构，没有实现其 GQA DCP 路径。
- `lmdeploy/pytorch/configurations/builder.py` 按解析后的 attention TP 计算 KV 副本数，不能只检查 `attn_tp % dcp == 0` 就认定 GQA 的模型布局成立。
- 当前 Q 投影复制只对显式适配的 BF16 MLA 模型开放，MTP draft 未适配；MTP 验证必须关闭 `dcp_replicate_q_proj`。支持的 MLA 模型可以验证 DCP + MTP，不能据此推导 Qwen3-235B-A22B 也支持。

后续精度口径：沿用完整 GSM8K test 集（1,319 题），固定模型 revision、提示词、thinking 模式、解码参数及评分规则；分别保留同拓扑 DCP1、DCP2 原生通信和优化通信的对照。MTP 开关作为独立变量，记录 draft 配置和实际启用状态。通信输入/输出逐位检查与 GSM8K 分开记录，不能仅凭准确率接近判定通信正确。模型或拓扑调整需单独明确，不把替换后的实验当作这三组原始要求已经完成。

2A 接入说明：

- 当前 `--dcp-comm-backend {ag_rs,a2a}` / `PytorchEngineConfig.dcp_comm_backend` 选择输出合并算法，默认 `ag_rs`；DCP 的 collective 目前仍是原生 PyTorch/NCCL。
- 2A 是可独立接入的 **all-gather 传输实现**，接入 `DCPManager` / `create_all_gather_workspace`，不改变 attention 或 FP32 合并算法。需要按 DCP 组协商能力、在 capture 前准备缓冲区，并提供原生通信回退。
- 建议使用独立 gather 选项区分 `nccl` / `symm_mem` / `auto`；具体参数名待实现，**当前没有可用的 DCP symmetric-memory 开关**。已有 TP all-reduce / LM-head symmetric-memory 环境变量不控制 DCP gather。
- 对 attention 主链路：`ag_rs` 可优化 Q/LSE gather；`a2a` 可优化 Q gather；`a2a + Q复制` 已没有这两项 gather。Sparse indexer、prefix KV gather 的复用收益需单独测量，不能混入 decode 主链路收益。
