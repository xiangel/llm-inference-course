---
title: 第 13 章 · 从 nano-vLLM 到 vLLM V1
description: 将小型引擎的职责映射到 vLLM V1 的 engine、request、KV cache 和 scheduler 源码。
---

# 第 13 章 · 从 nano-vLLM 到 vLLM V1

**学时** 2–3 小时 · **需要** Python 代码阅读能力 · **本章没有** Colab，也不要求 GPU

一句话回答：**当教学引擎扩展为生产引擎时，原来的 Request、Scheduler 与 block pool 分别在哪里？**

## 为什么

直接从 vLLM V1 源码跳入，会被类型、抽象、异步接口和版本演进淹没；只停在 nano-vLLM，又会误以为生产系统只是“把代码写长”。映射同一种职责能保留第 12 章的心智模型，同时看见生产系统为何需要更多边界。

## 心智模型

不要找“nano-vLLM 的一对一翻译”。把它看作职责拆分：engine 协调生命周期，request 保存状态，scheduler 决定本轮工作，KV cache manager 与 block pool 管理容量。生产版本把接口、配置、失败路径和并发协作显式化。

## 数据流

<RequestLifecycleFlow />

用下面路线给图中的服务端做标注：

```text
API/输入 → v1/engine/llm_engine.py → v1/request.py
          → v1/core/sched/... → v1/core/kv_cache_manager.py
          → v1/core/block_pool.py → executor/model runner → 输出
```

这是职责图，不是承诺每个版本的精确调用栈或文件行号。

## 按步骤映射源码

### 1. 固定 vLLM 版本，先从 engine 进入

在 [vLLM 仓库](https://github.com/vllm-project/vllm) 选择 tag 或 commit，记录它。打开 `vllm/v1/engine/llm_engine.py`，搜索请求加入、engine step 和输出处理等职责。它相当于第 12 章的总协调者：接收已处理的请求、调用 scheduler、交给执行层并处理输出。不要宣称教程对应“第几行”；代码持续演进。

### 2. 对照 request.py 的状态边界

阅读 `vllm/v1/request.py`。将它与 nano-vLLM 的 Request 对照：输入、生成进度、停止条件、输出与调度相关状态分别在哪里？特别追踪“新 token 被接受后”状态怎样更新，以及完成/取消后对象如何离开活跃路径。生产 Request 还需承载比教学例子更多的请求语义和配置。

### 3. 进入 core/sched：从策略到调度结果

浏览 `vllm/v1/core/sched/` 下的模块，从 scheduler 的入口和 schedule 输出类型开始。寻找它如何在 waiting/running 请求、token budget、prefill 与 decode 之间取舍。不要假设所有版本的文件名或策略一样；用仓库搜索定位 `schedule`、budget、scheduled request 等概念。

将结果与第 9 章对应：每轮动态选择工作就是连续批处理；但真实 scheduler 还必须面对请求取消、优先级、缓存容量和执行边界。

### 4. 跟踪 kv_cache_manager 的准入

打开 `vllm/v1/core/kv_cache_manager.py`。关注它向调度决策提供的能力：某请求是否能获得所需 KV 空间、可否命中前缀、完成后如何释放。它不负责“预测下一个 token”；它是容量和复用策略的守门人。

### 5. 跟踪 block_pool 的物理资源

打开 `vllm/v1/core/block_pool.py`，再回看第 10 章的块池模拟。找空闲块、分配、释放、缓存条目和引用/使用关系。真实实现可能用哈希与淘汰来复用前缀，且细节会随版本变化；以当前源码和测试为准，不能从简化模拟推出全部语义。

### 6. 建一张自己的职责表

| nano-vLLM 概念 | vLLM V1 阅读位置 | 要验证的问题 |
| --- | --- | --- |
| 引擎循环 | `v1/engine/llm_engine.py` | 一轮如何从调度到输出？ |
| 请求状态 | `v1/request.py` | token、停止和取消怎样更新？ |
| 调度器 | `v1/core/sched/` | token budget 如何决定本轮工作？ |
| cache 准入 | `v1/core/kv_cache_manager.py` | 空间/前缀如何影响能否调度？ |
| 物理块池 | `v1/core/block_pool.py` | 块如何分配、复用和回收？ |

最后选一条短请求，跨文件追一次：加入 engine → Request → scheduler 输出 → KV 决策 → 执行结果 → Request 完成。用 IDE 的“查找引用”和测试，而不是依赖博客中的旧行号。

## 生产连接

代码入口以 [vLLM V1 文档](https://docs.vllm.ai/en/latest/) 和 [vLLM 主仓库](https://github.com/vllm-project/vllm) 为准。阅读生产代码时还应查看相邻测试：测试往往定义了取消、缓存命中、回收和边界条件的可观察语义。若要运行服务，回到第 8 章的真实 GPU 要求。

## 常见误解

- 文件路径是阅读起点，不是稳定 API 承诺；不要写死行号。
- `block_pool` 管块不等于独自决定所有调度策略。
- engine 不是模型 kernel；它协调多个子系统。
- 能画出职责图不表示已经验证并发、抢占或 OOM 路径。

## 小结

- 从 `llm_engine.py` 进入，以 `request.py` 追踪状态。
- 在 `core/sched` 看每轮选择，在 `kv_cache_manager.py` 看容量，在 `block_pool.py` 看物理块。
- 用固定 commit、调用引用和测试对抗源码漂移。

## 自测

<Quiz :questions="[
  { prompt: '阅读 vLLM V1 时，为何不应宣称精确行号？', options: ['Python 没有行号', '源码会演进，tag/commit 才能固定上下文', '文件不能打开', 'GPU 不支持调试'], answer: 1, explanation: '路径和职责可作导航，精确位置随版本改变。' },
  { prompt: '哪个文件最直接对应请求状态追踪？', options: ['v1/request.py', 'README.md', 'setup.py', 'LICENSE'], answer: 0, explanation: 'Request 的输入、生成进度和完成状态应从此处开始追踪。' },
  { prompt: 'KV cache manager 在心智模型中主要负责什么？', options: ['预测 token', '容量、缓存命中与分配准入', '训练模型', '发送 HTTP 响应'], answer: 1, explanation: '模型执行产生 logits；cache manager 管理运行时缓存资源。' }
]" />

## 参考资料

- [vLLM 主仓库](https://github.com/vllm-project/vllm) — 固定 tag/commit 后阅读所列 V1 路径。
- [vLLM 文档](https://docs.vllm.ai/en/latest/) — 当前架构与运行时文档。
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) — 用于对照教学版的职责模型。
