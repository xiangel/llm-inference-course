---
title: "第七篇 · GPU Kernel 与性能优化"
description: "Kernel 在栈上的位置，Triton Attention、PagedAttention 间接层与 Profiling。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**Kernel 在栈上的哪一层？怎样证明它变快了？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 7.1 | Kernel 在栈上的位置：它不是调度器 | 层的边界 |
| 7.2 | Softmax 与 Attention 的切块 | tiling |
| 7.3 | 用 Triton 写能讲明白的 Attention | 教学 kernel |
| 7.4 | PagedAttention kernel 的间接层 | 块表怎么进 kernel |
| 7.5 | Profiling：把慢从 Python 里找出来 | Nsight / 计时 |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
