---
title: "第四篇 · Batch、调度与分页 KV"
description: "静态 batch 为何失败，Continuous Batching、Chunked Prefill、Scheduler 与 Paged KV 如何接上。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**许多人同时来时，怎样共用 GPU 和 KV？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 4.1 | 静态 batch 如何让短请求给长请求买单 | 朴素 batch 的失败 |
| 4.2 | Continuous Batching：请求何时加入、何时离开 | iteration 级组 batch |
| 4.3 | Chunked Prefill：长 Prompt 怎样切成可调度的块 | Token budget |
| 4.4 | Scheduler：这一轮算谁、算多少 token | 调度策略 |
| 4.5 | Paged KV：按块分配，而不是按最大长度预留 | Block / BlockTable |
| 4.6 | Prefix Cache、抢占与拒绝 | 前缀复用与过载 |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
