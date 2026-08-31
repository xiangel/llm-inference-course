---
title: 第 9 章 · 连续批处理：让 GPU 不空转
description: 用一个 CPU 调度模拟器理解为什么推理服务不应等待整批请求结束。
---

# 第 9 章 · 连续批处理：让 GPU 不空转

**学时** 1–2 小时 · **需要** Python 与免费 Colab（本章模拟器只用 CPU）· **本章不需要** GPU 或真实模型

一句话回答：**不同长度的请求同时到达时，为什么要在每一轮重新组批？**

<Checklist
  slug="09-continuous-batching"
  :items="[
    { id: 'static', label: '能指出静态批处理的空槽' },
    { id: 'simulate', label: '在 Colab 运行 CPU 调度模拟器' },
    { id: 'admit', label: '能解释 admission 与 token budget' },
    { id: 'quiz', label: '完成自测' }
  ]"
/>

<a class="colab-link" href="https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/09_continuous_batching.ipynb" target="_blank" rel="noreferrer">在 Google Colab 打开 CPU 调度模拟器 ↗</a>

## 为什么

静态批处理会把一组请求绑在一起：短回答先完成后，它占的槽位只能闲着，直到最长回答结束。在线服务的请求长度和到达时间不可预测，GPU 因而会在可做工作时等待。

连续批处理让调度器在每轮生成后移除完成请求、加入等待请求。它提升的是利用率和吞吐，不保证每个请求都更快；队列太深时，排队延迟仍会上升。

## 心智模型

把 GPU 想成每回合能处理有限座位的班车。静态批处理要求同一车的人一起下车；连续批处理允许到站的人下车、候车的人立刻补位。每位乘客每回合只推进一个 decode token，长 prompt 的 prefill 则需要额外预算。

## 数据流

<RequestLifecycleFlow />

在这张已有图中，服务端并非一次只处理一条箭头：调度器会汇集多个请求的下一段工作，送入一次 GPU 执行，再依据结果更新每条请求。

## 按步骤实验

### 1. 运行模拟器

打开顶部 Colab。它创建若干带有 `arrival` 和 `remaining_tokens` 的请求；每个 tick 最多服务固定数量的活跃请求。输出会显示每个 tick 的活跃集合、完成事件和空槽数。

### 2. 观察静态批处理

模拟器先把第一批请求固定住。把一个请求的输出长度设为 2，另一个设为 12：短请求完成后的多个 tick 中，槽位仍然空着。这就是“GPU 不空转”要解决的浪费，虽然真实 GPU 的成本比玩具模型复杂得多。

### 3. 切到连续批处理

连续版本每 tick 都先移除完成项，再从等待队列补充。比较总 tick 数和空槽数。调度规则要有上限：真实系统按 token、KV 块和显存预算准入，不能无限加入请求。

### 4. 改一个变量再解释结果

依次改变最大活跃请求数、到达间隔和输出长度。记录吞吐、平均等待和最长等待。若把容量只调大而不考虑 KV Cache，模拟器虽“更快”，真实服务却可能 OOM；这是模型与模拟之间最重要的边界。

## 生产连接

[vLLM continuous batching 文档](https://docs.vllm.ai/en/latest/) 和其调度器源码才定义真实行为。生产调度还要处理 Prefill、Decode、取消、优先级、token budget、KV 块分配与抢占。用指标观察请求数、队列等待、TTFT、ITL、GPU 利用率和 KV Cache 使用量，而不是只看平均吞吐。

## 常见误解

- 连续批处理不是把所有请求“合并成一个 prompt”。
- 吞吐提高不必然降低 p99 延迟；没有公平策略，长或晚到请求可能饥饿。
- 一轮不总是恰好一个 token：chunked prefill 会将 prompt 工作也放入调度预算。
- CPU 模拟说明策略，不测量 GPU 性能。

## 小结

- 在线请求长度不同，静态批次会留下空槽。
- 连续批处理在每轮完成后换入等待请求。
- 真实准入必须受 token、KV Cache 与显存约束。

## 自测

<Quiz :questions="[
  { prompt: '静态批处理最明显的浪费何时出现？', options: ['所有请求刚到达', '短请求先完成、长请求仍在生成时', 'Tokenizer 完成时', '模型下载时'], answer: 1, explanation: '已完成请求的槽位无法被新请求补上。' },
  { prompt: '连续批处理每轮的重要动作是？', options: ['重新训练模型', '移除完成请求并补入等待请求', '清空所有 KV Cache', '合并用户文本'], answer: 1, explanation: '它动态维护活跃批次。' },
  { prompt: '为什么真实服务不能无限扩大活跃批次？', options: ['HTTP 只允许一个请求', 'KV Cache 和显存有容量上限', 'Python 不支持列表', '输出永远只有一个 token'], answer: 1, explanation: '每个活跃序列都会占用状态和显存。' }
]" />

## 参考资料

- [vLLM 文档](https://docs.vllm.ai/en/latest/) — 服务、调度和配置的官方主入口。
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu) — 迭代级调度/continuous batching 的原始系统论文。
- [vLLM 论文](https://arxiv.org/abs/2309.06180) — 将连续调度与 KV Cache 管理放进完整服务系统的背景。
