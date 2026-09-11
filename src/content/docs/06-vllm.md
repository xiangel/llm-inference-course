---
title: "第六篇 · 走进 vLLM"
description: "对照 Mini-vLLM 阅读 vLLM V1：生命周期、EngineCore、Scheduler、KV 与 Model Runner。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**生产引擎在 Mini-vLLM 之外多了哪些工程？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 6.1 | V1 的模块边界：哪些在引擎里、哪些不在 | 引擎 ≠ 平台 |
| 6.2 | 一次请求的生命周期 | Request lifecycle |
| 6.3 | EngineCore 主循环 | schedule / execute / update |
| 6.4 | Scheduler 与 KV Cache Manager | 生产调度与分页 |
| 6.5 | GPU Model Runner 与 CUDA Graph | 执行器 |
| 6.6 | 对照 Mini-vLLM：多出来的是工程，不是另一套数学 | 差异清单 |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
