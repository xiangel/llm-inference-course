---
title: "第五篇 · 从零实现 Mini-vLLM"
description: "把 Engine、Scheduler、BlockManager、ModelRunner、Sampler 收成一个能跑的教学引擎。"
---

本篇正文已开放。问题：**前面那些块怎样收成一个能跑的教学引擎？它不能说成 vLLM。**

| 节 | 标题 |
| --- | --- |
| 5.1 | [最小引擎的五块](./01-five-blocks/) |
| 5.2 | [Request：一条生成从进入到结束的状态](./02-request/) |
| 5.3 | [Scheduler：选出这一 iteration 的 batch](./03-scheduler/) |
| 5.4 | [BlockManager：KV 块怎么发、怎么收回](./04-block-manager/) |
| 5.5 | [ModelRunner：组 batch 并跑模型](./05-model-runner/) |
| 5.6 | [Sampler 与流式输出](./06-sampler-stream/) |
| 5.7 | [测一轮 TTFT / TPOT，并标明这不是 vLLM](./07-not-vllm/) |

代码在仓库 `mini-vllm/`。测试：`python3 -m unittest tests.test_mini_vllm -v`。
