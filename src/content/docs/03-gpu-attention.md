---
title: "第三篇 · GPU 与 Attention"
description: "GPU 如何执行 Transformer，Decode 为何 Memory Bound，FlashAttention 与 PagedAttention 的分工。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**Prefill 吃算力、Decode 吃带宽，在 GPU 上对应什么？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 3.1 | GPU 执行一次 forward 时发生了什么 | SM、HBM、Kernel launch |
| 3.2 | 算力、带宽和算术强度 | Roofline |
| 3.3 | Decode 为什么容易 Memory Bound | 读权重 + 读 KV |
| 3.4 | FlashAttention 减少的是哪一次写回 | FlashAttention |
| 3.5 | 和 PagedAttention 的分工 | 计算核 vs 显存管理 |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
