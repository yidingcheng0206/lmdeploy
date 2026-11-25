# 序列并行

LMDeploy 的两种推理后端提供不同的上下文并行接口：TurboMind 使用 `--cp`，PyTorch 的 Decode Context Parallelism（DCP）使用 `--dcp`。

## TurboMind 序列并行

在单卡显存不足以部署模型的时候，通常会以 `TP` 的方式进行部署，而这一般要求 `num_key_value_heads` 被 `TP` 整除。如果要以 `TP > num_key_value_heads` 的方式进行部署，需要创建 kv-heads 的副本，以满足整除需求。但是这样会有两个缺点：

1. 可用的 kvcache 数量减半，进而减少请求最大推理长度
2. 降低推理的最大 batch 数量，减少吞吐量。

为了解决这个问题，TurboMind 推理后端支持设置 `attn_dp_size`，避免了创建 kv-heads 的副本，但是这会引入数据的不均衡性。为了消除数据的不均衡，TurboMind 支持了序列并行，支持将 kv_cache 交错存储到不同的 cp_rank 上，例如

```
cp_rank=2, prompt_len=5, generation_len=4
kv_cache stored on cp_rank0: 0, 2, 4, 6, 8
kv_cache stored on cp_rank1: 1, 3, 5, 7
```

在序列并行模式下，`cache_block_seq_len` 仍表示每个 rank 的一个 k/v cache block 实际存放的 token 数。调度器中对应的逻辑块覆盖 `cache_block_seq_len * cp` 个全局 token。因此每个 rank 上 k/v block 的显存大小不变，而整块前缀复用和只读 cache 边界按更大的全局跨度计算。

### 使用说明

以 `Intern-S1` / `Qwen3-235B-A22B` 为例，他们的 `num_key_value_heads` 为 4，若要用 `TP=8` 的方式部署，并避免 kv_cache 的拷贝，可以用如下的方式部署

```
lmdeploy serve api_server internlm/Intern-S1 --tp 8 --cp 2

lmdeploy serve api_server Qwen/Qwen3-235B-A22B --tp 8 --cp 2
```

## PyTorch Decode Context Parallelism

PyTorch DCP 在已有的 attention TP 组内分摊 MLA KV cache，无需增加 GPU 即可提高有效缓存容量，支持基于 FlashMLA 的 dense MLA 和 sparse DSA 模型。

```bash
lmdeploy serve api_server <mla-model> --backend pytorch --tp 4 --dcp 2
```

Python 中使用 `PytorchEngineConfig(tp=4, dcp=2)`；默认 `dcp=1` 表示关闭 DCP。

### 使用条件与支持范围

- 需要 NVIDIA Hopper/SM90 GPU、FlashMLA 和 BF16 激活。Sparse DSA 还需要兼容的 DeepGEMM、TileLang top-k 内核，支持 top-k 512 或 2048。
- `dcp` 必须整除解析后的 `attn_tp` 和复制后的 KV head 数。DCP 组位于 attention TP 组内，可以与 DP/EP 组合。例如 `--tp 8 --dp 2 --ep 8 --dcp 4` 使用 8 张卡，包含两个各 4 张卡的 attention TP 组，各自包含一个 DCP 组。
- 支持兼容 MLA 模型的 prefix caching 和 `deepseek_mtp`，包括 DeepSeek V3/V3.1、DeepSeek V3.2、GLM DSA。
- 支持 BF16 KV cache；sparse MLA 也支持通过 `--quant-policy fp8` 使用 FP8 KV cache。
- 不支持 sliding-window attention、MemDecode、PD 分离、外部 KV cache connector 和 TileLang attention 后端。

### Decode 通信选项

默认的 `--dcp-comm-backend ag_rs` 先 all-gather Q，再 all-gather 各分片的 log-sum-exp（LSE），最后对修正后的 attention 输出执行 reduce-scatter。`--dcp-comm-backend a2a` 将输出和 FP32 LSE 打包到一次 all-to-all 中，接收端合并结果。最终输出仍只包含本 rank 的 attention heads。两种路径均支持 CUDA Graph。

对于支持的 BF16 MLA Q 投影，`--dcp-replicate-q-proj` 在权重加载后，于 DCP 组内一次性收集 Q 投影和吸收后的 K 投影权重。Decode 在本地计算整个 DCP 组对应的 Q heads，从而省去 Q all-gather；prefill 继续使用原有的本地 head 切片。该选项要求 `dcp > 1`，增加权重显存和投影计算量，不改变 attention TP、MLP TP 或 EP 分组。

```bash
lmdeploy serve api_server <bf16-mla-model> --backend pytorch \
  --tp 8 --dp 2 --ep 8 --dcp 4 \
  --dcp-comm-backend a2a --dcp-replicate-q-proj
```

Python 配置对应为：

```python
PytorchEngineConfig(tp=8, dp=2, ep=8, dcp=4,
                    dcp_comm_backend='a2a', dcp_replicate_q_proj=True)
```

Q 复制需显式启用，目前支持 DeepSeek V2/V3、DeepSeek V3.2 和 GLM DSA 实现中的未量化 BF16 Q 投影。量化 Q 投影、LoRA、未显式适配的模型实现（包括 MTP draft 模型）会被拒绝。A2A 不依赖 Q 复制，可以用于现有的量化 MLA 路径。本实现不包含 GQA DCP。

长上下文 FP8 sparse MLA 冒烟测试中，A2A 引入前的基线（main 加 DP/EP 配置修改）和 A2A 都曾出现 retrieval 错误。根因尚未定位，不能据此判定为 A2A 引入的回归，也不能认为 FP8 精度已验证通过。完整的端到端 FP8 精度验证仍待完成。

Attention 的 DCP 通信链中，AG/RS、A2A、A2A 加 Q 复制分别需要 3、2、1 次 collective；这里不计输出投影、MoE 和 sparse indexer 通信。通信次数减少不保证端到端延迟降低，应根据实际负载测试。

### 通信资源管理

每个分布式上下文持有一个绑定本 DCP 组的 `DCPManager`，统一承接 attention 通信、sparse indexer 候选收集和缓存前缀 KV 收集。不同 DP 组的 manager 相互独立，可以处理不同的 batch；同一 DCP 组内的 collective shape 必须一致。

同一 CUDA 执行流上的各层共享 A2A 收发缓冲区，不同流使用独立缓冲区。缓冲区按几何级数扩容，旧分配会保留以供已捕获的 Graph 使用，warmup 必须在 capture 前分配足够空间。共享缓冲区的 Graph 必须在 capture 时的执行流上串行 replay，Graph runner 会检查这一条件。重置时先释放 target 和 draft 的 Graph，再释放共享缓冲区。KV cache 容量估算会在 prefill/indexer 临时空间之外预留这部分持久显存。

目前组级 communicator 使用原生 PyTorch/NCCL collective；workspace 接口为后续优化 gather provider 预留扩展点。本次重构未启用 symmetric-memory 通信，也未改变 attention 合并的数值算法。
