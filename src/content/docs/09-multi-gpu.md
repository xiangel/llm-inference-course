---
title: "第九篇 · 多 GPU 推理"
description: "Tensor / Pipeline / Data / Expert Parallel 各切开什么，以及通信走在哪一层。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**一张卡放不下时，切开、复制、分层各解决什么？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 9.1 | 第一道墙：权重为什么一张卡放不下 | 必须多卡的原因 |
| 9.2 | Tensor Parallel：切开一个模型 | TP |
| 9.3 | Pipeline Parallel 与 Data Parallel | PP ≠ DP |
| 9.4 | MoE 与 Expert Parallel | EP |
| 9.5 | 通信走在 NVLink、IB 还是 NCCL | 互连与库 |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
