---
title: "第三篇 · GPU 执行与 Attention"
description: "GPU 如何执行 Transformer，Decode 为何 Memory Bound，FlashAttention 与 PagedAttention 的分工。"
---

3.1 已开放，3.2–3.6 也已开放。本篇问题：**推理卡上有什么？快慢分叉在 GPU 上对应什么？**

| 节 | 标题 |
| --- | --- |
| 3.1 | [GPU 是什么、现在有哪些卡](../03-gpu/01-hardware/) |
| 3.2 | [GPU 执行一次 forward 时发生了什么](../03-gpu/02-forward/) |
| 3.3 | [算力、带宽和算术强度](../03-gpu/03-roofline/) |
| 3.4 | [Decode 为什么容易 Memory Bound](../03-gpu/04-decode-memory-bound/) |
| 3.5 | [FlashAttention 减少的是哪一次写回](../03-gpu/05-flashattention/) |
| 3.6 | [和 PagedAttention 的分工](../03-gpu/06-paged-vs-flash/) |

完整目录见 [后续篇章目录](../later-chapters/)。
