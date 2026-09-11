---
title: "第四篇 · Batch、调度与分页 KV"
description: "静态 batch 为何失败，Continuous Batching、Chunked Prefill、Scheduler 与 Paged KV 如何接上。"
---

本篇正文已开放。问题：**许多人同时来时，怎样共用 GPU 和 KV？**

| 节 | 标题 |
| --- | --- |
| 4.1 | [静态 batch 如何让短请求给长请求买单](./01-static-batch/) |
| 4.2 | [Continuous Batching：请求何时加入、何时离开](./02-continuous-batching/) |
| 4.3 | [Chunked Prefill：长 Prompt 怎样切成可调度的块](./03-chunked-prefill/) |
| 4.4 | [Scheduler：这一轮算谁、算多少 token](./04-scheduler/) |
| 4.5 | [Paged KV：按块分配，而不是按最大长度预留](./05-paged-kv/) |
| 4.6 | [Prefix Cache、抢占与拒绝](./06-prefix-preempt/) |

上一篇：[3.6 和 PagedAttention 的分工](../03-gpu/06-paged-vs-flash/)。下一篇：[5.1 最小引擎的五块](../05-mini-vllm/01-five-blocks/)。
