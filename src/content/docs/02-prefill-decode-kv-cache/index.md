---
title: "第二篇 · Prefill、Decode 与 KV Cache"
description: "读问题和写回答为什么快慢完全不同，KV Cache 存什么、显存如何计算，以及 GQA 如何降低 KV。"
---

本篇正文已开放。问题：**读问题和写回答为什么快慢完全不同？中间结果为什么越写越占显存？**

| 节 | 标题 |
| --- | --- |
| 2.1 | [Prefill：怎样一次性读完整段 Prompt](./01-prefill/) |
| 2.2 | [Decode：怎样每次只生成一个 token](./02-decode/) |
| 2.3 | [为什么 Prefill 吃算力、Decode 吃带宽](./03-compute-vs-bandwidth/) |
| 2.4 | [KV Cache 缓存的是什么、不缓存什么](./04-what-kv-stores/) |
| 2.5 | [KV 显存怎么乘到 70B 的 2.5 GiB](./05-kv-memory/) |
| 2.6 | [GQA：为什么 70B 的 KV 头是 8 不是 64](./06-gqa/) |

上一篇：[1.4 RoPE、Logits 与 Sampling](../01-transformer/04-rope-logits-sampling/)。下一篇：[3.1 GPU 是什么](../03-gpu/01-hardware/)。
