---
title: "第十一篇 · 高级推理技术"
description: "投机解码、长上下文、KV 压缩、Prefill/Decode 分离与 KV Transfer。"
---

本篇正文尚未开放。章节标题已排好，写正文时按这个顺序带概念。

## 本篇要解决的问题

**单引擎主路径之外，长上下文和高吞吐还靠什么？**

## 章节目录

| 节 | 标题 | 这一节带出什么 |
| --- | --- | --- |
| 11.1 | 投机解码：用小模型猜、大模型验 | Speculative decoding |
| 11.2 | 长上下文：KV 会先把显存吃完 | 长上下文压力 |
| 11.3 | KV 压缩与收回 | KV 压缩 |
| 11.4 | Prefill / Decode 分离 | PD 分离 |
| 11.5 | KV 在机器之间怎么传 | KV Transfer |

完整目录见 [后续篇章目录](../later-chapters/)。请先读完：

- [第 0 篇：这本书讲什么](../00-introduction/)
- [第一篇：理解 LLM 推理](../01-transformer/01-generate-one-token/)
