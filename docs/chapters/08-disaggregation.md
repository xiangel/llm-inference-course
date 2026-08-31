---
title: 第 8 章 · 部署第一个 vLLM 服务
description: 用官方 vLLM serve 启动 OpenAI 兼容服务，并从 Python 调用它。
---

# 第 8 章 · 部署第一个 vLLM 服务

**学时** 1–2 小时 · **需要** Linux、受 vLLM 支持的 NVIDIA GPU、Python · **本章不讲** GPU 配置、集群部署或公网安全

一句话回答：**怎样把一个模型变成自己的 OpenAI 兼容 HTTP 服务？**

<Checklist
  slug="08-first-vllm-service"
  :items="[
    { id: 'gpu', label: '确认 NVIDIA GPU 与驱动可用' },
    { id: 'serve', label: '用 vLLM serve 启动模型' },
    { id: 'client', label: '用 OpenAI Python 客户端请求服务' },
    { id: 'quiz', label: '完成自测' }
  ]"
/>

## 为什么

直接在 Python 进程加载模型适合实验；应用需要一个可被多个程序调用的长期服务。vLLM 在服务端管理模型、Tokenizer、GPU 调度和流式输出，客户端只发送 HTTP 请求。

**硬件要求必须如实看待：**本章不能在免费 CPU 环境完成。你需要受支持的 NVIDIA GPU、正常的驱动，以及容纳模型权重和 KV Cache 的显存。模型卡、精度、上下文长度和并发都会改变需求；不能从模型名称猜测“必定能跑”。不要用假 Colab 截图替代真实验证。

## 心智模型

`vllm serve` 像餐厅的前台和厨房：模型权重是固定设备，HTTP 请求是订单。服务把文字变为 token、安排 GPU 工作并回传 token；调用者不必在自己的进程保存权重。

## 数据流

<RequestLifecycleFlow />

图中的“推理服务”就是本章的 vLLM 进程。请求经过 Tokenizer、Prefill 和逐 token Decode，最终作为完整或流式 HTTP 响应返回。

## 按步骤部署

### 1. 先检查机器

```bash
nvidia-smi
python --version
```

`nvidia-smi` 必须能列出 GPU 和驱动。若它不存在、驱动报错或容器没有 GPU，先修复环境；安装 Python 包不会提供 GPU。CUDA、PyTorch 与 vLLM 的匹配方式以官方安装页为准。

### 2. 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm openai
```

平台若要求特定 CUDA/PyTorch 安装方式，请遵守官方说明，不要混用随机 wheel 命令。

### 3. 启动官方服务

选择一个你有权访问且显存足够的模型。下例只是名称示例，不承诺任意 GPU 都能加载：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 --port 8000 --api-key local-dev-key
```

首次会下载权重。就绪后，在另一终端检查：

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer local-dev-key"
```

`127.0.0.1` 只允许本机访问。若对外监听，还需要 TLS、网关、访问控制、限流与超时；`--api-key` 不是完整的公网安全方案。

### 4. 用 OpenAI 兼容客户端调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1",
                api_key="local-dev-key")
reply = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "用一句话解释 KV Cache。"}],
    temperature=0.2,
    max_tokens=80,
)
print(reply.choices[0].message.content)
```

`base_url` 指向自己的服务；`model` 必须匹配服务加载的模型名或配置的 served model name。改为 `stream=True` 并逐块打印 `delta.content`，即可体验流式响应。

## 生产连接

请求成功只证明一条路径可用。生产还应记录 vLLM/模型版本、GPU、输入输出 token、延迟和错误，并设置健康检查、指标、限流和许可审查。主文档是 [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)。

## 常见误解

- “OpenAI 兼容”不代表每项功能、参数和模型行为完全相同。
- `curl` 成功不等于能承受并发；下一章解释调度。
- 小模型不一定能装下，KV Cache 也消耗显存。
- 暴露端口不是部署完成，而是安全工作的开始。

## 小结

- `vllm serve` 将模型包装成 HTTP 服务。
- OpenAI Python 客户端可借由自定义 `base_url` 调用它。
- 必须在真实、受支持的 GPU 上按实际容量验证。

## 自测

<Quiz :questions="[
  { prompt: '连接本章服务最关键的客户端设置是？', options: ['model 写 gpt-4', 'base_url 指向本机 /v1', '删除 api_key', '客户端加载权重'], answer: 1, explanation: 'base_url 决定请求发往自己的 vLLM 服务。' },
  { prompt: '为什么不能承诺任意小模型都能在任意 GPU 上运行？', options: ['Tokenizer 不支持 GPU', '显存还受精度、上下文、并发和 KV Cache 影响', 'HTTP 不支持小模型', '端口由模型决定'], answer: 1, explanation: '显存需求不只由参数量决定。' },
  { prompt: '对外监听后应优先补充什么？', options: ['无限输出', 'TLS、访问控制和限流', '关闭日志', '删除健康检查'], answer: 1, explanation: '公开服务需要完整的网络与应用层防护。' }
]" />

## 参考资料

- [vLLM：OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) — 本章命令与 API 的主来源。
- [vLLM：Installation](https://docs.vllm.ai/en/latest/getting_started/installation/) — 按平台确认硬件与安装组合。
- [OpenAI Python SDK](https://github.com/openai/openai-python) — 客户端与自定义 `base_url` 参考。
