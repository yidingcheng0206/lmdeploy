# Pipeline 与 Serving Benchmark 流程

本文记录如何从 smoke 数据集开始，跑通 LMDeploy 的 pipeline benchmark、serving benchmark，以及 `benchmark_serving.py` 自动编排流程。本文的目标不是得到正式性能结论，而是先理解 benchmark 的概念、脚本关系、输入输出、配置文件和常见问题。

## 1. Benchmark 是什么

Benchmark 是在固定条件下做性能测试。对于 LLM 推理系统，benchmark 关注的是推理系统的性能，而不是模型回答质量。

它回答的问题包括：

- 一个系统每秒能完成多少请求
- 每秒能生成多少 token
- 用户多久能看到第一个 token
- 后续 token 生成速度是否稳定
- 请求失败率是多少
- 不同后端在相同条件下谁更快

常见指标：

| 指标 | 含义 |
|---|---|
| Request throughput | 每秒完成请求数 |
| Input token throughput | 每秒处理输入 token 数 |
| Output token throughput | 每秒生成输出 token 数 |
| E2E latency | 请求从发出到完成的端到端耗时 |
| TTFT | Time To First Token，首 token 延迟 |
| ITL | Inter-token Latency，相邻输出 token 的间隔 |
| TPOT | Time Per Output Token，平均每个输出 token 耗时 |
| Successful requests | 成功完成的请求数 |

普通用例验证的是：

```text
接口能不能跑通，模型能不能返回文本。
```

Benchmark 验证的是：

```text
在一批请求和固定压力下，系统跑得多快、多稳。
```

## 2. 本次使用的模型和数据

### 2.1 模型

先用小模型打通流程：

```text
Qwen2.5-0.5B-Instruct
```

真实模型路径：

```text
/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

如果以后换模型，可以用：

```bash
find /path/to/models--xxx--yyy -name config.json
```

`config.json` 所在目录就是通常应传给 LMDeploy 的模型目录。

### 2.2 Smoke 数据集

本次新增了小数据集：

```text
benchmark/smoke_sharegpt.json
```

它是 ShareGPT 格式：

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Explain machine learning."},
      {"from": "gpt", "value": "Machine learning is ..."}
    ]
  }
]
```

Benchmark 脚本会取：

- 第一轮 `human.value` 作为输入 prompt
- 第一轮 `gpt.value` 的 token 长度作为默认输出长度

如果指定：

```bash
--sharegpt-output-len 32
```

则忽略原始回答长度，每个请求目标输出 32 tokens。

Smoke 数据集只用于验证流程，不用于正式性能结论。

## 3. 三个 Benchmark 脚本的关系

本阶段涉及三个脚本：

```text
benchmark/profile_pipeline_api.py
benchmark/profile_restful_api.py
benchmark/benchmark_serving.py
```

### 3.1 `profile_pipeline_api.py`

作用：

```text
直接调用本地 Python pipeline API 做 benchmark。
```

它会：

1. 加载模型并创建 `pipeline`
2. 读取数据集
3. 构造 `GenerationConfig`
4. 调用 `pipe.stream_infer(...)` 或 `pipe(...)`
5. 用 `Profiler` 统计吞吐和延迟
6. 打印表格并可保存 CSV

它测的是：

```text
Python pipeline API + engine
```

不经过 HTTP 服务。

### 3.2 `profile_restful_api.py`

作用：

```text
压测一个已经启动的 HTTP serving 服务。
```

它不会启动服务。你必须先启动：

```bash
lmdeploy serve api_server ...
```

然后它会：

1. 访问 `/v1/models` 获取模型 id，或使用显式传入的 `--model`
2. 读取数据集
3. 异步发送 HTTP 请求
4. 支持 `/v1/completions` 和 `/v1/chat/completions`
5. 统计 TTFT、ITL、TPOT、throughput
6. 打印结果

它测的是：

```text
HTTP serving + OpenAI 协议开销 + engine
```

### 3.3 `benchmark_serving.py`

作用：

```text
自动编排 serving benchmark。
```

它本身不是新的压测逻辑，而是把手动步骤自动化：

```text
读取 YAML 配置
  -> 启动 lmdeploy/vLLM/SGLang 服务
  -> 等待 /v1/models ready
  -> 调用 profile_restful_api.py
  -> 保存 CSV
  -> 关闭服务
```

关系可以理解为：

```text
profile_restful_api.py = 真正发请求和统计指标的压测客户端
benchmark_serving.py = 启服务、等服务、调压测客户端、关服务的编排器
```

## 4. Pipeline Benchmark 手动流程

### 4.1 依赖检查

在 `(lmdeploy)` 环境中执行：

```bash
python3 - <<'PY'
import numpy
import aiohttp
import tqdm
import transformers
print("benchmark deps ok")
PY
```

如果缺包，可安装：

```bash
pip install numpy aiohttp tqdm requests pybase64 pillow
```

### 4.2 选择空闲 GPU

查看 GPU：

```bash
nvidia-smi
```

本次 4-7 卡被占用，2 卡空闲，因此使用：

```bash
CUDA_VISIBLE_DEVICES=2
```

Smoke 模型是 0.5B，单卡足够。流程验证阶段不建议先上双卡，因为 `tp=2` 会引入多卡通信变量。

### 4.3 执行 pipeline benchmark

```bash
cd /mnt/shared-storage-user/llmrazor-share/yidingcheng/lmdeploy

CUDA_VISIBLE_DEVICES=2 python3 benchmark/profile_pipeline_api.py \
  benchmark/smoke_sharegpt.json \
  /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --backend pytorch \
  --num-prompts 8 \
  --concurrency 2 \
  --session-len 4096 \
  --cache-max-entry-count 0.5 \
  --sharegpt-output-len 32 \
  --stream-output \
  --csv benchmark_pipeline_smoke.csv
```

参数解释：

| 参数 | 含义 |
|---|---|
| `benchmark/smoke_sharegpt.json` | 数据集路径 |
| `model_path` | 模型目录 |
| `--backend pytorch` | 使用 PyTorch backend |
| `--num-prompts 8` | 总共跑 8 条请求 |
| `--concurrency 2` | pipeline benchmark 并发为 2 |
| `--session-len 4096` | 上下文长度 |
| `--cache-max-entry-count 0.5` | KV cache 使用空闲显存比例 |
| `--sharegpt-output-len 32` | 每条目标输出 32 tokens |
| `--stream-output` | 使用流式输出并统计 TTFT/ITL |
| `--csv` | 保存结果 CSV |

### 4.4 成功输出

示例结果：

```text
Total requests                                   8
Successful requests                              8
Total input tokens                              84
Total generated tokens                         256
Request throughput (req/s)                  12.507
Output throughput (tok/s)                  400.227
Time to First Token (TTFT)                   0.474
Time per Output Token (TPOT)                 0.002
Inter-token Latency (ITL)                    0.003
```

因为设置了：

```text
num_prompts = 8
sharegpt_output_len = 32
```

所以预期输出 token 数：

```text
8 * 32 = 256
```

结果中：

```text
Total generated tokens = 256
```

说明输出长度控制和统计逻辑正常。

### 4.5 查看 CSV

```bash
cat benchmark_pipeline_smoke.csv
```

示例：

```csv
backend,bs,dataset_name,sharegpt_output_len,random_input_len,random_output_len,random_range_ratio,num_prompts,completed,total_input_tokens,total_output_tokens,duration,request_throughput,input_throughput,output_throughput,mean_e2e_latency_ms,mean_ttft_ms,mean_tpot_ms,mean_itl_ms
pytorch,2,sharegpt,32,,,0.0,8,8,84,256,0.6396371420123614,12.507,131.324,400.227,528.399,473.842,1.760,3.209
```

## 5. Serving Benchmark 手动流程

### 5.1 启动服务

另开终端：

```bash
cd /mnt/shared-storage-user/llmrazor-share/yidingcheng/lmdeploy

CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server \
  /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --backend pytorch \
  --server-name 0.0.0.0 \
  --server-port 23335 \
  --tp 1 \
  --session-len 4096 \
  --cache-max-entry-count 0.5
```

看到：

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:23335
```

表示服务启动成功。

### 5.2 确认服务

另一个终端：

```bash
curl -i --noproxy '*' http://127.0.0.1:23335/v1/models
```

如果返回：

```text
HTTP/1.1 200 OK
content-type: application/json
```

并包含模型 id，则服务正常。

### 5.3 执行 serving benchmark

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
python3 benchmark/profile_restful_api.py \
  --backend lmdeploy-chat \
  --host 127.0.0.1 \
  --port 23335 \
  --dataset-name sharegpt \
  --dataset-path benchmark/smoke_sharegpt.json \
  --model /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --model-path /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --num-prompts 8 \
  --sharegpt-output-len 32 \
  --request-rate inf
```

参数解释：

| 参数 | 含义 |
|---|---|
| `--backend lmdeploy-chat` | 使用 `/v1/chat/completions` |
| `--host 127.0.0.1` | 服务地址 |
| `--port 23335` | 服务端口 |
| `--dataset-path` | 数据集路径 |
| `--model` | HTTP 请求体中的 model 字段 |
| `--model-path` | 本地 tokenizer 路径，用于 token 统计 |
| `--request-rate inf` | 一次性尽快发出请求 |

`--model` 和 `--model-path` 的区别非常重要：

```text
--model      是发给服务端的 OpenAI model id
--model-path 是 benchmark 本地加载 tokenizer 的路径
```

如果服务端通过 `--model-name` 暴露了别名，则 `--model` 必须使用这个别名。

### 5.4 成功输出

示例：

```text
Successful requests:                     8
Benchmark duration (s):                  1.21
Total input tokens:                      83
Total generated tokens:                  256
Request throughput (req/s):              6.63
Output token throughput (tok/s):         212.14
Mean TTFT (ms):                          989.74
Mean TPOT (ms):                          6.73
Mean ITL (ms):                           31.39
```

同样检查：

```text
8 * 32 = 256
```

与 `Total generated tokens` 一致。

## 6. YAML 自动编排流程

### 6.1 YAML 是什么

YAML 是配置文件格式，用来把很多命令行参数结构化保存下来。

手写命令适合调试：

```bash
lmdeploy serve api_server ...
python3 benchmark/profile_restful_api.py ...
```

YAML 适合正式实验：

```text
固定模型、服务端参数、数据集参数、输出长度、并发配置
```

好处：

- 实验参数可复现
- 不容易手工复制错
- 可以批量跑多组 engine/data 配置
- 方便后续 LMDeploy/vLLM/SGLang 对比

### 6.2 Smoke YAML

本次新增配置：

```text
benchmark/qwen25_05b_smoke.yml
```

内容：

```yaml
dataset_path: &dataset_path "benchmark/smoke_sharegpt.json"
dataset_name: &dataset_name "sharegpt"
model_path: &model_path "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

server:
  server_ip: "127.0.0.1"
  server_name: "127.0.0.1"
  server_port: 23336

engine:
  - model_path: *model_path
    model_name: "qwen25_05b_smoke"
    max_batch_size: 2
    cache_max_entry_count: 0.5
    session_len: 4096
    tp: 1

data:
  - dataset_name: *dataset_name
    dataset_path: *dataset_path
    backend: "lmdeploy-chat"
    model: "qwen25_05b_smoke"
    model_path: *model_path
    num_prompts: 8
    sharegpt_output_len: 32
    request_rate: .inf
```

YAML 中的 `&model_path` 和 `*model_path` 是锚点和引用：

```text
&model_path 定义一个可复用值
*model_path 引用这个值
```

这样可以避免长路径重复写多次。

### 6.3 `benchmark_serving.py` 做什么

执行：

```bash
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
CUDA_VISIBLE_DEVICES=2 python3 benchmark/benchmark_serving.py \
  pytorch \
  benchmark/qwen25_05b_smoke.yml
```

内部流程：

```text
读取 benchmark/qwen25_05b_smoke.yml
  -> 合并 server 和 engine 配置
  -> 拼出 lmdeploy serve api_server 命令
  -> 启动服务子进程
  -> 轮询 /v1/models
  -> 拼出 profile_restful_api.py 命令
  -> 执行 benchmark
  -> 写 CSV
  -> 关闭服务
```

示例启动命令：

```text
lmdeploy serve api_server /path/to/model --backend pytorch --server-name 127.0.0.1 --server-port 23336 --model-name qwen25_05b_smoke --max-batch-size 2 --cache-max-entry-count 0.5 --session-len 4096 --tp 1
```

示例 benchmark 命令：

```text
python3 benchmark/profile_restful_api.py --backend lmdeploy-chat --host 127.0.0.1 --port 23336 --dataset-name sharegpt --dataset-path benchmark/smoke_sharegpt.json --model qwen25_05b_smoke --model-path /path/to/model --num-prompts 8 --sharegpt-output-len 32 --request-rate inf --output-file benchmark_qwen25_05b_smoke_pytorch_bs2_tp1_cache0.5.csv
```

### 6.4 自动编排成功输出

关键日志：

```text
Starting api_server: lmdeploy serve api_server ...
Server is ready.
Running benchmark: python3 ... profile_restful_api.py ...
Successful requests: 8
Finished server process
```

示例结果：

```text
Backend:                                 lmdeploy-chat
Successful requests:                     8
Benchmark duration (s):                  0.49
Total input tokens:                      83
Total generated tokens:                  256
Request throughput (req/s):              16.32
Output token throughput (tok/s):         522.16
Mean TTFT (ms):                          227.20
Mean TPOT (ms):                          2.21
Mean ITL (ms):                           4.20
```

输出 CSV：

```text
benchmark_qwen25_05b_smoke_pytorch_bs2_tp1_cache0.5.csv
```

## 7. 本次对 `benchmark_serving.py` 的修正

为了让 smoke YAML 能走通，做了两个小修正。

### 7.1 `server_ip` 不传给 `lmdeploy serve`

原先 `benchmark_serving.py` 会把 YAML 里的所有 `server` 字段都转成 CLI 参数。

配置中有：

```yaml
server_ip: "127.0.0.1"
```

会被转成：

```bash
--server-ip 127.0.0.1
```

但 `lmdeploy serve api_server` 不认识 `--server-ip`，它使用：

```bash
--server-name
```

因此修正为：

```python
if key == 'server_ip':
    continue
```

`server_ip` 只给 wrapper 自己连接服务使用，不传给 `lmdeploy serve`。

### 7.2 data 中允许覆盖 benchmark backend

原脚本对 LMDeploy backend 默认传给 `profile_restful_api.py`：

```text
--backend lmdeploy
```

这会走 `/v1/completions`。

本次 smoke 希望走 chat completions：

```text
/v1/chat/completions
```

因此允许 data 配置中写：

```yaml
backend: "lmdeploy-chat"
```

对应代码：

```python
if backend in ['turbomind', 'pytorch']:
    backend = client_config.pop('backend', 'lmdeploy')
```

## 8. 常见问题

### 8.1 启动期间出现 Connection error

日志：

```text
connect to server http://127.0.0.1:23336 failed Connection error.
```

这是正常现象。`benchmark_serving.py` 启动服务子进程后，会立刻轮询 `/v1/models`。

服务启动需要时间：

```text
加载 tokenizer
加载权重
构建 engine
启动 Uvicorn
绑定端口
```

在 Uvicorn 开始监听前，轮询会失败。看到：

```text
Server is ready.
```

说明等待成功。

### 8.2 SOCKS proxy 报错

日志：

```text
Using SOCKS proxy, but the 'socksio' package is not installed
```

说明环境里有代理变量，`OpenAI` 或 `httpx` client 尝试走代理。跑本地服务时应清空代理：

```bash
HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
...
```

### 8.3 model 不存在

日志：

```text
The model '/path/to/model' does not exist.
```

原因通常是服务启动时设置了：

```bash
--model-name qwen25_05b_smoke
```

那么 HTTP 请求中的 `model` 必须是：

```text
qwen25_05b_smoke
```

而不是本地模型路径。

YAML 中应写：

```yaml
model: "qwen25_05b_smoke"
model_path: *model_path
```

### 8.4 端口被占用

日志：

```text
address already in use
```

处理方式：

```bash
ss -ltnp | grep 23336
```

或者换端口：

```yaml
server_port: 23337
```

### 8.5 Smoke 数字不能作为正式结论

Smoke 只跑 8 条请求，样本太小，数字会明显波动。

它的意义是：

```text
脚本能跑通
数据格式正确
模型路径正确
指标能输出
CSV 能保存
自动编排能完成
```

不是正式性能对比。

## 9. 三种 Benchmark 方式对比

| 方式 | 脚本 | 是否启动服务 | 是否 HTTP | 适合阶段 |
|---|---|---|---|---|
| Pipeline benchmark | `profile_pipeline_api.py` | 否 | 否 | 验证本地 API 与 engine 性能 |
| Serving 手动 benchmark | `profile_restful_api.py` | 否，需要手动启动 | 是 | 拆解排错、理解 HTTP benchmark |
| Serving 自动 benchmark | `benchmark_serving.py` | 是 | 是 | 正式实验编排、多后端对比 |

调用关系：

```text
benchmark_serving.py
  -> lmdeploy serve api_server / vllm serve / sglang.launch_server
  -> profile_restful_api.py
```

## 10. 后续正式 Benchmark 建议

第一阶段已经跑通：

```text
profile_pipeline_api.py
profile_restful_api.py
benchmark_serving.py
YAML 配置
自动启停服务
CSV 输出
```

下一步建议：

1. 扩大 smoke：

```text
num_prompts = 16
sharegpt_output_len = 64
concurrency/max_batch_size = 4
```

2. 找到正式 ShareGPT 数据集。

3. 增加正式配置：

```text
num_prompts = 100 / 1000 / 10000
sharegpt_output_len = 512 / 1024 / 2048
```

4. 换正式模型：

```text
Qwen3-30B-A3B
Qwen3-30B-A3B-Instruct
Qwen3-30B-A3B-FP8
```

5. 对比后端：

```text
LMDeploy PyTorch
LMDeploy TurboMind
vLLM
SGLang
```

6. 统一记录：

```text
模型路径
GPU 型号和卡数
backend
tp/dp/ep
输入输出长度
请求数
并发或 request rate
TTFT
TPOT/ITL
吞吐
显存占用
失败率
```

正式对比时必须保持模型、数据集、输出长度、GPU 数量、请求压力尽量一致。
