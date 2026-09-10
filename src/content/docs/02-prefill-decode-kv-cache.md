---
title: "第二篇 · Prefill、Decode 与 KV Cache"
description: "读问题和写回答为什么快慢完全不同，KV Cache 存什么、显存如何计算，以及 GQA 如何降低 KV。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**读问题和写回答为什么快慢完全不同？中间结果为什么越写越占显存？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 2.1 | Prefill：怎样一次性读完整段 Prompt | Prefill 的计算形状 |
| 2.2 | Decode：怎样每次只生成一个 token | Decode 一步 |
| 2.3 | 为什么 Prefill 吃算力、Decode 吃带宽 | 两段活的瓶颈分叉 |
| 2.4 | KV Cache 缓存的是什么、不缓存什么 | K、V，不是输出 |
| 2.5 | KV 显存怎么乘到 70B 的 2.5 GiB | $L,H_{kv},d_h,S,B$ |
| 2.6 | GQA：为什么 70B 的 KV 头是 8 不是 64 | GQA |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
