# Pipeline 与 Serving 上手流程

本文记录一次从本地模型路径开始，跑通 LMDeploy `pipeline` 和 OpenAI 兼容 `api_server` 的完整流程。目标是先理解接口、输入输出和源码调用链，再进入 benchmark。

## 1. 确认模型路径

如果模型在 HuggingFace cache 结构中，不能直接传 `models--xxx--yyy` 根目录，通常要传 `snapshots/<commit>` 目录。

例如：

```bash
find /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct \
  -name config.json
```

输出：

```text
/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/config.json
```

因此模型路径为：

```text
/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

如果 `config.json` 就在模型目录下，则直接传这个目录。

## 2. Pipeline 用例

本仓库中新增了一个 smoke test 脚本：

```text
examples/run_mistral_pipeline.py
```

虽然文件名里有 `mistral`，但它已经参数化，可以传任意模型路径。

运行 Qwen2.5-0.5B-Instruct：

```bash
cd /mnt/shared-storage-user/llmrazor-share/yidingcheng/lmdeploy

CUDA_VISIBLE_DEVICES=4 python3 examples/run_mistral_pipeline.py \
  --model-path /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --backend pytorch \
  --tp 1 \
  --session-len 4096 \
  --cache-max-entry-count 0.5 \
  --max-new-tokens 64
```

### 2.1 Pipeline 初始化

脚本中的核心代码：

```python
from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline


backend_config = PytorchEngineConfig(
    tp=1,
    session_len=4096,
    cache_max_entry_count=0.5,
)

gen_config = GenerationConfig(
    max_new_tokens=64,
    temperature=0.0,
)

with pipeline(model_path, backend_config=backend_config, log_level='WARNING') as pipe:
    ...
```

`backend_config` 控制推理引擎，常用参数：

| 参数 | 含义 |
|---|---|
| `tp` | Tensor Parallel 卡数 |
| `session_len` | 最大上下文长度 |
| `cache_max_entry_count` | KV cache 使用空闲显存的比例 |
| `backend` | 可选 `pytorch` 或 `turbomind` |

`gen_config` 控制生成行为：

| 参数 | 含义 |
|---|---|
| `max_new_tokens` | 最多生成 token 数 |
| `temperature=0.0` | 贪心生成，便于复现 |
| `top_p` / `top_k` | 采样参数 |
| `ignore_eos` | benchmark 中常用于强制生成到指定长度 |

### 2.2 普通 batch 输入

代码：

```python
prompts = [
    'Hello, my name is',
    'The capital of France is',
]
responses = pipe(prompts, gen_config=gen_config)

for resp in responses:
    print(resp.text)
```

输入类型：

```python
list[str]
```

输出类型：

```python
list[Response]
```

`Response` 常用字段：

```python
resp.text
resp.input_token_len
resp.generate_token_len
resp.finish_reason
resp.index
resp.token_ids
```

### 2.3 OpenAI messages 输入

代码：

```python
prompts = [
    [{
        'role': 'user',
        'content': 'Please introduce yourself briefly.',
    }],
    [{
        'role': 'user',
        'content': 'Write one sentence about Paris.',
    }],
]

responses = pipe(prompts, gen_config=gen_config)
```

输入类型：

```python
list[list[dict]]
```

这种输入更接近 chat 模型和 serving 的 `/v1/chat/completions`。Pipeline 内部会做 prompt 格式化和 chat template 处理。

### 2.4 流式输出 `stream_infer`

代码：

```python
chunks = {}
for item in pipe.stream_infer(prompts, gen_config=gen_config):
    chunks[item.index] = chunks.get(item.index, '') + item.text
    print(item.index, item.text, item.finish_reason)
```

输出类型：

```python
Iterator[Response]
```

重要点：

- `item.text` 是本次增量文本。
- `item.index` 表示 batch 中第几个请求。
- `item.finish_reason is None` 表示该请求还在生成。
- `item.finish_reason == 'stop'` 表示正常停止。
- `item.finish_reason == 'length'` 表示达到 `max_new_tokens` 上限。

### 2.5 多轮对话 `chat`

代码：

```python
session = pipe.chat('你好，我叫小明。请只回复一句话确认你记住了。', gen_config=gen_config)
print(session.response.text)

session = pipe.chat('我叫什么名字？', session=session, gen_config=gen_config)
print(session.response.text)
```

输出类型：

```python
Session
```

常用字段：

```python
session.session_id
session.response.text
session.history
session.step
```

`session` 会保存历史。第二轮传入同一个 `session`，模型可以利用第一轮上下文。

### 2.6 流式多轮 `chat`

代码：

```python
chunks = []
for item in pipe.chat('用一句话解释什么是 KV cache。', gen_config=gen_config, stream_response=True):
    chunks.append(item.text)
    print(item.text, end='', flush=True)

print(''.join(chunks))
```

输出类型：

```python
Iterator[Response]
```

## 3. Serving 用例

Serving 启动一个兼容 OpenAI API 的 HTTP 服务。

启动命令：

```bash
CUDA_VISIBLE_DEVICES=4 lmdeploy serve api_server \
  /mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --backend pytorch \
  --server-name 0.0.0.0 \
  --server-port 23334 \
  --tp 1 \
  --session-len 4096 \
  --cache-max-entry-count 0.5
```

看到以下日志表示服务启动成功：

```text
Application startup complete.
Uvicorn running on http://0.0.0.0:23334
```

如果出现：

```text
address already in use
```

说明端口被占用，可以换端口，比如 `23334`。

### 3.1 查询模型

```bash
curl -i --noproxy '*' http://127.0.0.1:23334/v1/models
```

输出示例：

```json
{
  "object": "list",
  "data": [
    {
      "id": "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
      "object": "model",
      "owned_by": "lmdeploy"
    }
  ]
}
```

后续请求中的 `model` 字段要使用这里返回的 `id`。

### 3.2 非流式 chat completions

```bash
curl --noproxy '*' http://127.0.0.1:23334/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍你自己。"}
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

输出示例：

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "/path/to/model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "我是Qwen，由阿里云开发的超大规模语言模型..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 36,
    "completion_tokens": 30,
    "total_tokens": 66
  }
}
```

### 3.3 流式 chat completions

```bash
curl --noproxy '*' http://127.0.0.1:23334/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "messages": [
      {"role": "user", "content": "请列出三个颜色。"}
    ],
    "max_tokens": 64,
    "temperature": 0,
    "stream": true
  }'
```

输出是 SSE：

```text
data: {"choices":[{"delta":{"content":"红"}}]}
data: {"choices":[{"delta":{"content":"色"}}]}
...
data: [DONE]
```

## 4. Pipeline 与 Serving 对比

| 项目 | Pipeline | Serving |
|---|---|---|
| 调用方式 | Python 本地 API | HTTP API |
| 入口 | `lmdeploy.pipeline()` | `lmdeploy serve api_server` |
| 输入 | `str` / `list[str]` / OpenAI messages / tuple multimodal | OpenAI JSON |
| 非流式输出 | `Response` / `list[Response]` | JSON |
| 流式输出 | `Iterator[Response]` | SSE |
| 多轮状态 | `Session` 对象保存历史 | 通常由客户端传完整 `messages` |
| 适合场景 | 本地调试、离线推理、源码阅读 | 部署服务、远程调用、压测 |

字段对照：

| 概念 | Pipeline | Serving |
|---|---|---|
| 生成文本 | `Response.text` | `choices[0].message.content` |
| 流式增量 | `item.text` | `choices[0].delta.content` |
| 请求编号 | `Response.index` | `choices[0].index` |
| 停止原因 | `Response.finish_reason` | `choices[0].finish_reason` |
| 输入 token | `Response.input_token_len` | `usage.prompt_tokens` |
| 输出 token | `Response.generate_token_len` | `usage.completion_tokens` |
| 总 token | 手动相加 | `usage.total_tokens` |

## 5. 源码调用链

### 5.1 Pipeline 源码链路

用户代码：

```python
from lmdeploy import pipeline

pipe = pipeline(model_path, backend_config=backend_config)
responses = pipe(prompts, gen_config=gen_config)
```

调用链：

```text
lmdeploy.pipeline(...)
  -> lmdeploy/api.py:pipeline
  -> lmdeploy/pipeline.py:Pipeline.__init__
  -> get_model(...)
  -> autoget_backend_config(...)
  -> get_task(...)
  -> 创建 async_engine
  -> async_engine.start_loop(...)
```

推理调用：

```text
pipe(...)
  -> Pipeline.__call__
  -> Pipeline.infer
  -> MultimodalProcessor.format_prompts
  -> Pipeline._request_generator
  -> Pipeline._infer
  -> async_engine
  -> Response
```

流式调用：

```text
pipe.stream_infer(...)
  -> MultimodalProcessor.format_prompts
  -> Pipeline._request_generator
  -> Pipeline._infer(multiplex=True)
  -> Iterator[Response]
```

多轮调用：

```text
pipe.chat(...)
  -> 获取或复用 Session
  -> session.update(...)
  -> stream_infer(...)
  -> 聚合 Response
  -> session.response = resp
  -> session.history.append(...)
```

重点文件：

| 文件 | 作用 |
|---|---|
| `lmdeploy/api.py` | 用户 API 入口 |
| `lmdeploy/pipeline.py` | Pipeline 主逻辑 |
| `lmdeploy/messages.py` | `GenerationConfig` / `Response` 等结构 |
| `lmdeploy/archs.py` | backend 自动识别与任务分发 |
| `lmdeploy/serve/core/async_engine.py` | engine 异步调度 |
| `lmdeploy/serve/managers/session_manager.py` | `Session` 管理 |

### 5.2 Serving 源码链路

启动命令：

```bash
lmdeploy serve api_server /path/to/model ...
```

HTTP 调用链：

```text
POST /v1/chat/completions
  -> lmdeploy/serve/openai/api_server.py
  -> serving_chat_completion.py
  -> AsyncEngine
  -> Response
  -> OpenAI JSON 或 SSE
```

重点文件：

| 文件 | 作用 |
|---|---|
| `lmdeploy/serve/openai/api_server.py` | FastAPI 路由入口 |
| `lmdeploy/serve/openai/serving_chat_completion.py` | chat completion 业务逻辑 |
| `lmdeploy/serve/openai/protocol.py` | OpenAI 协议请求/响应结构 |
| `lmdeploy/serve/openai/api_client.py` | LMDeploy 自带客户端 |
| `lmdeploy/serve/core/async_engine.py` | serving 底层推理引擎 |

一句话总结：

```text
Pipeline 是 Python 本地直接调 engine。
Serving 是用 HTTP/OpenAI 协议包装同类 engine。
```

## 6. 常见问题

### 6.1 本地路径被当成 HuggingFace repo id

报错类似：

```text
HFValidationError: Repo id must be in the form ...
```

原因通常是路径不存在，LMDeploy 判断它不是本地目录，于是当作 repo id 下载。检查路径拼写，特别是 `gpfs2` / `gpf2` 这类差异。

### 6.2 PyTorch backend 不能用 stdin 脚本

不要用：

```bash
python3 - <<'PY'
...
PY
```

PyTorch backend 会启动多进程，子进程需要重新加载主文件。stdin 脚本没有真实文件路径，会报：

```text
FileNotFoundError: ... <stdin>
```

正确做法是写成 `.py` 文件，并加：

```python
if __name__ == '__main__':
    main()
```

### 6.3 端口被占用

报错：

```text
address already in use
```

检查：

```bash
ss -ltnp | grep 23333
```

或者直接换端口：

```bash
--server-port 23334
```

### 6.4 curl 没输出

建议使用：

```bash
curl -i --noproxy '*' http://127.0.0.1:23334/v1/models
```

不要优先用 `0.0.0.0` 做客户端请求，`0.0.0.0` 更适合作为服务监听地址。

### 6.5 transformers 版本 warning

例如：

```text
LMDeploy requires transformers version: [4.33.0 ~ 5.3.0], but found version: 5.13.0
```

如果当前模型能正常加载和推理，可以先记录 warning，继续跑用例。若后续遇到模型加载或 tokenizer 兼容问题，再考虑调整环境版本。

## 7. 进入 Benchmark 前应掌握什么

进入 benchmark 前，应确认自己理解以下字段：

```python
Response.index
Response.generate_token_len
Response.input_token_len
Response.finish_reason
```

因为 `benchmark/profile_pipeline_api.py` 中会用它们统计每个请求的生成进度和完成状态：

```python
for output in self.pipe.stream_infer(prompts, gen_config=gen_configs, do_preprocess=False):
    index = output.index
    n_token = output.generate_token_len
    finish_reason = output.finish_reason
    sess[index].tick(n_token)
```

进入 serving benchmark 时，也要理解 OpenAI 响应中的：

```json
{
  "choices": [...],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

下一步建议先读：

```text
benchmark/profile_pipeline_api.py
benchmark/profile_restful_api.py
benchmark/benchmark_serving.py
```

再开始用小模型跑小样本 benchmark。
