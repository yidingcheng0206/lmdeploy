# DCP Phase 2 设计文档：PeerPullMerge（纯通信层优化）

> 状态：设计稿（未实施）
> 适用硬件：Hopper SM90 + NVSwitch（单机 dcp 组）
> 前置依赖：Phase 0（与 PR #4995 合流）、Phase 1（symmetric-memory 基建）
> 参考：知乎《乱谈推理之在RTPLLM中为K3 MLA实现DCP及通算融合》（RTP-LLM MegaA2A/PeerPullMerge）；
> InternLM/lmdeploy PR #4952（MLA DCP 基础，已合并）、PR #4995（symm-mem all-gather 基建）

---

## 0. 摘要

用单个自研 kernel（PeerPullMerge）替换 DCP decode 输出归并的
`pack → NCCL all_to_all → combine` 三段链路。**不修改任何 MLA attention kernel**，
数值语义与现有 combine kernel 逐位同路（零数值风险），预期把每层 merge 段
从 ~60-100μs 压到 ~25-40μs（RTP-LLM 同构实现的实测口径），NCCL a2a 路径保留为
fallback。

## 1. 背景与动机

当前 a2a backend 每层 decode 的输出归并链路：

| 步骤 | 内容 | 存在原因 | 实测量级（RTP-LLM 同构，8卡，8K/B32） |
|---|---|---|---|
| pack | O/LSE 重排进 `[dest][token][local_head][word]` 连续分块 | **NCCL collective 要求给每个目标的数据连续**，而 attention 输出是 `[token][head][dim]` 交错布局 | ~8μs |
| all_to_all_single | NCCL 集合通信 | 跨 rank 交换 | ~49μs（小消息固定延迟占主导） |
| combine | FP32 logsumexp 加权合并 | 最终归并 | ~4μs |

三段合计 ~61μs，其中 pack 是为 collective 的布局要求服务的"纯税"，NCCL 的固定
延迟在小消息下占主导。

PeerPullMerge 的两个消除手段：

1. **kernel 读远端内存不在乎布局**——目的卡按 (t, my_heads) 的自然偏移直接从每个
   peer 读，pack 整个消失；
2. **pull（目的端拉取）替代 collective**——TMA bulk copy 由硬件异步执行，LSE 权重
   计算与拉取重叠，实测归并开销仅 ~1μs（26.0μs vs 裸 NVLink pull 25.3μs @435GB/s）。

## 2. 设计原则：纯通信层，MLA kernel 零改动

```
FlashMLA / tilelang sparse kernel      ← 完全不动
        ↓ O [T, H_all, D] bf16, LSE [T, H_all] fp32
┌─ 接缝：输出落到对称内存 arena（见 §4）────────────────────┐
│                                                            │
│  PeerPullMerge（新 kernel，单次 launch）                    │
│    ① 入口 epoch 同步（~3.4μs，8 卡）                       │
│    ② 每 warp 对 P 个 peer 发 TMA bulk pull 进 SMEM slot     │
│    ③ LSE 权重在拉取在途时并行算好（重叠）                    │
│    ④ mbarrier 等数据到齐 → SM 侧 FP32 加权归并 → 写出        │
│                                                            │
└────────────────────────────────────────────────────────────┘
        ↓ 本 rank 的 H_local 头切片
o_proj / 后续全部              ← 不动（且靠 o_proj all-reduce 做排序，见 §5.2）
```

**改动文件清单**：

| 文件 | 改动 |
|---|---|
| `lmdeploy/pytorch/backends/cuda/attention/cp.py` | `merge_dcp_attention` 新增 pull 分支；a2a 分支保留为 fallback |
| `lmdeploy/pytorch/kernels/cuda/dcp.py`（或新文件） | 新增 `peer_pull_merge` kernel；pack/combine 保留（fallback 用） |
| `lmdeploy/pytorch/backends/cuda/attention/mla.py` / `sparse_mla.py` | 仅调用点：传 backend 枚举与 arena 句柄，attention 调用本身不动 |
| 新增 arena manager | 挂在 #4995 的 DCPManager 模式上，跨层共享一份 arena |
| CLI / config | `--dcp-comm-backend` 增加 `pull`（或 `auto`） |

**不碰**：`_forward_prefill`、attention kernel、KV cache、scheduler、paging、o_proj 链路。

## 3. 数据布局：相同布局消除 pack

每个 rank 在对称内存中持有：

```
O_arena   : [max_tokens, H_all, D]   bf16    （H_all = 本 attn TP 组全部头）
LSE_arena : [max_tokens, H_all]      fp32
epoch 槽  : 每 rank P 个入槽（peer 写）+ 1 个出槽（本 rank 写），单调 int64
```

关键性质：**各 rank 布局完全相同**。rank r 要合并头 h∈[r·H_local, (r+1)·H_local)，
则它在每个 peer 的 arena 里读的行偏移 = `t·(H_all·D) + h·D`，对所有 peer 一致
——不需要任何 per-destination 重排，这就是 pack 消失的原因。

D = v_head_dim（如 512），每行 O 为 1KB（bf16），LSE 每 (t,h) 一个 fp32 标量。

## 4. 接缝：attention 输出的落点（唯一与 MLA kernel 相关的决策）

PeerPull 要求 O/LSE 位于固定地址的对称 arena；vendor kernel 每次调用自行分配输出。

| 方案 | 条件 | 额外成本 | 决策 |
|---|---|---|---|
| A. kernel 直写 arena | wrapper 支持 `out=` | 0 | **已排除**：已查实 `flash_mla_with_kvcache` 不支持 out=（`mla.py:351` 直接接收返回值） |
| B. staging copy | 无条件可用 | 一次 `arena.copy_(attn_output)`，成本 ≈ 现有 pack（读 O + 写 buffer 各一遍） | **dense 路径采用** |
| C. tilelang 直写 | 自家 kernel，可改 epilogue | 0 | **sparse 路径 Phase 2.5 后置采用** |

注意：方案 B 下 Phase 2 的净收益 = 省掉 NCCL a2a（~49μs→TMA pull）+ combine 融合
进 pull kernel + receive buffer 内存；pack 等价成本以 staging copy 形式保留。
FlashMLA 的 `num_splits` 机制确认其内部已完成 split-KV 归并，输出永远是"已归并的
本地 O"——因此**不做 epilogue 融合**（RTP-LLM 实测 S>1 时融合会让通信量 ×S×2，
而 vendor kernel 场景下我们始终等效于"S>1 → 不融合 → pull"分支）。

## 5. 同步协议

### 5.1 就绪同步（epoch 机制）

- 每 rank kernel 尾部：所有 CTA 完成后，用 `atom.acq_rel.gpu` 计数，最后到达的
  CTA 以 `st.release.sys` 向 P 个 peer 的入槽写 `epoch`（单调递增 int64，**从不清零**
  ——CUDA Graph 反复 replay 安全）
- PeerPullMerge 入口：等本 rank 的 P 个入槽全部达到当前 epoch 再开始拉取
- 实测参考：8 卡一次 ~3.4μs（其中 ~2μs 是 system-scope release），比独立 barrier
  kernel 便宜 ~1μs

### 5.2 "读完"屏障：删除（用模型自身的 collective 排序）

下一层写 arena 前需保证所有 peer 已读完本层数据。论证链：

```
rank B 的 o_proj RS(i) 完成 ⟹ rank A 已进入 RS(i)
                            ⟹ A 的 merge(i) 已结束（同 stream 顺序）
⟹ B 进入第 i+1 层写 arena 时，A 必然已读完 B 的第 i 层数据
```

即 o_proj 的 attention-TP all-reduce 已经充当了 arena 复用的 ordering barrier，
**无需任何额外同步**。前提条件必须写进代码注释：
- attention 段不得启用 PDL（programmatic dependent launch）让远端写越过 collective；
- 没有 o_proj collective 的调用方（单测/benchmark）需自行补一次 all-reduce。

（这与 #4995 的 `reuse_sync=False` 是同一思想，可互相印证。）

## 6. 数值语义（零风险论证）

- 权重计算与加权求和全部在 SM 侧 FP32 完成，**与现有 `combine_dcp_attention`
  kernel 的数学路径逐位同路**：`w = softmax(LSE)`，空行（所有 LSE = −∞ 或
  valid_count = 0）→ 权重和为 0 → 输出 0（kernel 内 `denominator > 0` 守护，
  镜像现有分支）
- 无预乘、无 bf16 中间舍入点、无 multicast 依赖——相比 multicast `ld_reduce` 方案
  （需 exp 预乘，有 bf16 量化风险或 fp32 双倍流量），PeerPull 在 fabric 字节量相同
  的前提下数值零改动
- CI 中以"PeerPull 输出 vs 现有 a2a combine 输出"逐元素断言等价（复用
  `test_dcp_a2a.py` 的 NaN/±inf/空行/非连续 stride 对抗矩阵）

## 7. Kernel 设计要点（资源与实现配方）

参考 RTP-LLM 实测调优结论（Hopper 均适用：TMA bulk copy / mbarrier /
`st.release.sys`/`ld.acquire.sys` 均为 SM90 特性；不可用的仅 tcgen05，本设计不需要）：

- **分工**：一个 warp 负责一行输出（一个 token 的一个本 rank 头）
  - elect lane（1 线程）：对 P 个 peer 各发一条 1KB 的 `cp.async.bulk` TMA pull，
    写进本 warp 的 SMEM 槽，mbarrier 按字节数等齐
  - lane 0..P-1：各读一个 peer 的 LSE 标量
  - 全 warp：shuffle 求 LSE max / 权重 / 分母（与 TMA 在途重叠）
  - 等 mbarrier → 每 lane 从 SMEM 读 8 份中自己负责的特征段，FP32 归并，
    向量化写出
- **资源**：每 CTA 16 warp × 8KB SMEM 槽（P 个 peer × 1KB）≈ 128KB SMEM/CTA，
  每 SM 至多 1 CTA；grid = min(SM 数, ⌈行数/16⌉)，行多则 warp 按 grid 步长循环
- **入口在途字节**：所有行第一波同时发出，在途量 ≈ 本层全部待拉数据（如
  128×12 行 × 8KB ≈ 12MB），天然限流
- **实测敏感度**：grid/warp 数/单双缓冲/拷贝粒度的调整只在 1-3μs 内波动，
  不必过度调参；归并开销 ~1μs

**实现语言建议**：同步原语（`ld.acquire.sys`/`st.release.sys`/mbarrier 字节计数）
在 CUDA C++ 里写最稳（Triton 的内存序支持别扭）；可做成薄扩展，kernel 主体仍可
考虑 Triton + 指针重解释（现有 pack kernel 已用 `Packed.to(tl.pointer_type(...))`
的先例），或整体 CUDA C++。二选一在实施时定。

## 8. 配置与 Fallback

```
--dcp-comm-backend auto | pull | a2a | ag_rs     （默认 auto）
```

auto 的 eligibility 判定（所有 rank 一致决策，复用 #4995 的协商模式）：

- dcp 组 node-local 且 NVSwitch 互联（dcp 组=连续秩，单机部署天然满足）
- 对称内存分配成功（`_symmetric_memory` 或 IPC 映射可用）
- SM90+

任一不满足 → 静默回退 a2a（pack/all_to_all/combine 路径与 kernel 全部保留）。
跨机 dcp 组（未来 attn_tp>8 场景）自动落 a2a。

**风险隔离**：本优化的爆炸半径被限制在 merge 段加速；attention 数值行为完全不变；
fallback 路径即现有生产路径。

## 9. CUDA Graph 兼容

- arena 地址固定（初始化时一次分配，跨层共享，容量按 graph bucket 预留）
- epoch 单调递增、从不清零 → replay 安全
- 增长策略：沿用 #4995 的 `graph_ready_shapes` 模式，bucket 首次出现时在捕获外
  确认容量；已捕获 graph 引用的旧地址由 arena manager 持有保活

## 10. 预期收益与不节省项

| 项 | 现状（a2a/NCCL） | PeerPull 后 |
|---|---|---|
| 每层 merge 段（小-中 batch） | ~60-100μs（pack ~8 + NCCL ~49 + combine ~4 + 同步） | **~25-40μs**（staging copy + pull ~26μs + 入口等待） |
| 每层 kernel launch 数 | 3 + NCCL 内部 | 2（staging copy + PeerPullMerge） |
| receive buffer 内存 | send+recv 双份 wire buffer | 无（arena 复用） |
| 数值 | FP32 合并 | **不变** |

不节省项（诚实账）：staging copy（≈pack 等价成本，方案 B 固有）；入口处等最慢
rank attention 完成的时间（任何同步 merge 方案固有，实测 ~10μs 级）。

61 层 × 每层省 ~20-60μs ≈ 每 decode step 省 ~1.5-3.5ms。

## 11. 实施步骤

1. **Phase 0**：与 #4995 作者对齐边界并合流（DCPManager / SymmetricMemoryAllGather
   基建归公共，a2a 保留为 fallback，dp/ep 放开为本分支贡献）
2. **Phase 1**：q/LSE gather 换 symm-mem all-gather（#4995 现成代码）
3. **Phase 2**：
   a. arena manager（对称分配 + epoch 槽 + peer 指针交换，signal pad 可用 #4995 的）
   b. `peer_pull_merge` kernel（先 CUDA C++ 同步原语 + 主体，2 卡功能对齐）
   c. `merge_dcp_attention` 接 pull 分支 + auto eligibility 协商
   d. staging copy 接入 dense 路径；sparse 路径 Phase 2.5 改 tilelang 直写 arena
   e. 8 卡集成测试 + graph 捕获/重放 + arena 复用竞态测试
   f. benchmark 定版：B∈{1,8,64,256,512} × dcp∈{2,4}，a2a vs pull 曲线
4. 验收后考虑把 auto 的默认权重切到 pull

工期量级：kernel + 接线 + 测试 ≈ 数周（基建已被 Phase 0/1 摊销）。

## 12. 测试与验收

| 类别 | 内容 |
|---|---|
| 单元 | PeerPull vs a2a combine 逐元素等价（NaN/±inf/空行/stride 对抗矩阵，复用 `test_dcp_a2a.py`） |
| 集成 | 2 卡（CI 可跑）+ 8 卡全拓扑；graph 捕获/重放后 epoch 正确推进；arena 复用无竞态 |
| 精度 | GSM8K 对齐 ±0.5 点验收线（pull vs a2a 双跑） |
| 性能 | 上表指标；小 batch 延迟与大 batch 带宽双口径 |
| fallback | eligibility 不满足时静默回退且结果一致 |

## 13. 风险与已知坑

| 风险 | 缓解 |
|---|---|
| `torch.distributed._symmetric_memory` 为私有 API | 锁 PyTorch 版本 + 运行时特性探测 + 回退 |
| 轮询循环中 `%globaltimer` 编译到 `ld.acquire.sys` 之前会在部分配置必然 MMU fault（RTP-LLM 在 sm_103 实测） | 先 load 一次再开始计时；多配置跑 MMU fault 冒烟 |
| TMA 相关内存序 | `fence.proxy.async` 的正确使用：发布 epoch 前先 `wait_group`（不带 `.read`）排空 |
| 跨机 / 非 NVSwitch dcp 组 | auto 探测 → a2a fallback |
| PDL 与 §5.2 排序论证冲突 | 代码注释写死前提；单测/无 collective 调用方自行补 all-reduce |

## 14. 相关资料

- RTP-LLM DCP + MegaA2A 实测数据与设计（本方案的主要蓝本）：
  https://zhuanlan.zhihu.com/p/2091626447597261642
- InternLM/lmdeploy #4952：MLA DCP 基础（已合并，cp.py 基线）
- InternLM/lmdeploy #4995：GQA DCP + symmetric-memory all-gather 基建
- 本仓库相关文件：`backends/cuda/attention/cp.py`、`kernels/cuda/dcp.py`、
  `backends/cuda/attention/mla.py`（`flash_mla_with_kvcache` 调用，无 `out=`）、
  `tests/pytorch/kernel/test_dcp_a2a.py`、`tests/pytorch/test_dcp_communication.py`
