---
title: "GPU 执行与 FlashAttention"
description: "GPU 如何执行 Transformer，Decode 为何 Memory Bound，FlashAttention 与 PagedAttention 的分工。3.1 已开放。"
---

3.1 已开放：[GPU 是什么、现在有哪些卡](../03-gpu/01-hardware/)。

下面各节正文尚未开放。写的时候接在 3.1 的规格表之后，不要回头重讲“GPU 是什么”。

## 本篇要解决的问题

**Prefill 吃算力、Decode 吃带宽，在 GPU 上对应什么？**

## 章节目录

| 节 | 标题 | 这一节带出什么 | 状态 |
| --- | --- | --- | --- |
| 3.1 | [GPU 是什么、现在有哪些卡](../03-gpu/01-hardware/) | HBM、带宽、现有卡 | 已开放 |
| 3.2 | GPU 执行一次 forward 时发生了什么 | SM、Kernel launch | 写作中 |
| 3.3 | 算力、带宽和算术强度 | Roofline | 写作中 |
| 3.4 | Decode 为什么容易 Memory Bound | 读权重 + 读 KV | 写作中 |
| 3.5 | FlashAttention 减少的是哪一次写回 | FlashAttention | 写作中 |
| 3.6 | 和 PagedAttention 的分工 | 计算核 vs 显存管理 | 写作中 |

完整目录见 [后续篇章目录](../later-chapters/)。
