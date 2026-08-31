---
title: 第 12 章 · 读懂 nano-vLLM：从请求到 token
description: 跟随 GeeeekExplorer/nano-vllm 的小型实现，建立推理引擎的源码阅读路径。
---

# 第 12 章 · 读懂 nano-vLLM：从请求到 token

**学时** 2–3 小时 · **需要** Python 阅读能力；本地克隆仓库可选 · **本章没有** Colab，也不要求 GPU

一句话回答：**一条生成请求在小型推理引擎中如何变成连续输出的 token？**

## 为什么

大型 serving 项目同时包含网络、并行、kernel、缓存和调度，很容易迷路。`GeeeekExplorer/nano-vllm` 用更小的代码展示同一类职责：请求状态、调度、KV block、模型执行和采样。它是学习地图，不是生产 vLLM 的替代品。

## 心智模型

把引擎当作一个循环而非一个巨大函数：外部提交请求；调度器选本轮工作；执行器计算；采样器产生 token；请求状态更新；完成请求离开，未完成请求回到下一轮。读源码时始终问：**状态在哪个对象里、谁有权改变它、下一轮如何看见它？**

## 数据流

<RequestLifecycleFlow />

将图中的“推理服务”展开为：`LLMEngine` 接收请求 → `Scheduler` 选择 prefill/decode → KV cache/block manager 准备位置 → model runner 执行 → sampler 给出 token → request 更新并输出。名称可能随仓库版本调整，先搜索职责，不要死记类名。

## 按步骤读源码

### 1. 固定版本并找到入口

克隆或在网页中浏览 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。先读 README 的运行入口与目录树，记录 commit hash；教程对应的是仓库结构，不承诺未来每个文件仍同名。搜索 `add_request`、`step`、`generate` 或 CLI/API 入口，顺着调用到 engine。

### 2. 读 Request：状态而非文本

找到请求对象，标出 request id、输入 token、已生成 token、最大 token 数、完成原因，以及与 KV 块相关的字段。写下两条不变量：已生成 token 只能追加；完成请求不应再被调度。Python `dataclass`、列表和队列足以理解这些状态，无需先懂 CUDA。

### 3. 读调度器的一个 tick

从 `schedule()` 或等价函数开始，只跟一轮：它从 waiting/running 队列取什么，怎样判断容量，输出怎样的 scheduled batch。区分 prefill（处理多个输入 token）与 decode（为活跃请求追加 token）。暂时跳过 kernel 细节，先画出队列状态变化。

### 4. 追踪 KV 块生命周期

查找 block、cache、allocate、free 等词。记录请求何时分配块、何时扩容、何时释放。用第 10 章模型检查：逻辑 token 位置、物理块和引用/回收是否分离。小实现可能为清晰而省略并发、哈希前缀或复杂淘汰；这是教学取舍，不是 bug 证据。

### 5. 追到模型和采样

查看 runner/worker 如何接收 token IDs 和位置信息，产出 logits；再找 sampler 如何选择下一个 token。确认 engine 在哪里把 token 写回 Request，在哪里检测 EOS 或长度上限并把结果交给调用者。

### 6. 用一条请求做“纸上断点”

选短 prompt，按 `add → schedule → execute → sample → update → finish` 写表。每步写队列、token 数和块数。读不通时先打印/断点这三个边界：scheduler 输出、runner 输入、request 更新；不要从模型内部随机向外读。

## 生产连接

这个仓库是概念性来源；生产语义应以 [vLLM 项目](https://github.com/vllm-project/vllm) 和文档为准。nano-vLLM 通常省略分布式执行、成熟的内存安全/恢复、可观测性、OpenAI 服务层、完整模型兼容和高性能 kernel。下一章将同一职责映射到 vLLM V1。

## 常见误解

- “nano”不表示行为与生产 vLLM 完全一致。
- 源码中没有看到线程，不代表生产调度没有并发问题。
- 调度器不生成 token；它决定本轮让谁获得计算机会。
- 能跑一个 demo 不能证明缓存回收、取消和 OOM 处理正确。

## 小结

- 以请求状态和每轮 `step` 为主线阅读。
- 依序追踪 Request、Scheduler、KV blocks、Runner、Sampler。
- 小项目提供心智模型，生产细节要回到 vLLM 验证。

## 自测

<Quiz :questions="[
  { prompt: '阅读 nano-vLLM 最稳妥的主线是？', options: ['先背全部 CUDA API', '从一条请求的进入、调度、执行、采样和更新追踪', '只看 README', '只看模型权重'], answer: 1, explanation: '端到端状态流能把各模块职责连起来。' },
  { prompt: '调度器在一轮中主要决定什么？', options: ['训练梯度', '哪些请求/Token 获得本轮执行机会', '模型许可证', 'Tokenizer 词典'], answer: 1, explanation: 'token 由模型和采样器产生，调度器安排工作。' },
  { prompt: '为何不能把 nano-vLLM 当生产实现规范？', options: ['它不是 Python', '教学实现可能省略并发、恢复、分布式与高性能细节', '它没有请求对象', '它不包含 token'], answer: 1, explanation: '小实现故意缩小问题空间。' }
]" />

## 参考资料

- [GeeeekExplorer/nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) — 本章源码阅读的主仓库；先固定所读 commit。
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm) — 生产实现的主来源。
- [vLLM 论文](https://arxiv.org/abs/2309.06180) — 请求调度与 KV block 管理的系统背景。
