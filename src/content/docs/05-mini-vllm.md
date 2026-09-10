---
title: "第五篇 · 从零实现 Mini-vLLM"
description: "把 Engine、Scheduler、BlockManager、ModelRunner、Sampler 收成一个能跑的教学引擎。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**前面那些块怎样收成一个能跑的教学引擎？它不能说成 vLLM。**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 5.1 | 最小引擎的五块 | Engine 职责边界 |
| 5.2 | Request：一条生成从进入到结束的状态 | Request |
| 5.3 | Scheduler：选出这一 iteration 的 batch | 教学调度 |
| 5.4 | BlockManager：KV 块怎么发、怎么收回 | 教学分页 |
| 5.5 | ModelRunner：组 batch 并跑模型 | 执行 |
| 5.6 | Sampler 与流式输出 | 采样与流式 |
| 5.7 | 测一轮 TTFT / TPOT，并标明这不是 vLLM | 教学身份 |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
