---
title: "第十四篇 · 成本与容量规划"
description: "从权重和 KV 推 GPU 数，再落到 QPS 与 Cost per Million Tokens。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**从参数量和 KV，怎样推到要几张卡、一百万 token 多少钱？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 14.1 | 从权重和 KV 推 GPU 数 | 容量下界 |
| 14.2 | 并发、上下文与 QPS | 负载模型 |
| 14.3 | Cost per Million Tokens | 成本口径 |
| 14.4 | 量化、切卡还是换引擎 | 决策 |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
