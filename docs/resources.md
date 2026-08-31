---
title: 参考资料
outline: deep
---

# 参考资料

下面按「现在就能读」和「后续章节预告」分开。带 **必读** 的三篇，建议在学第 0–2 章时精读，而不是当链接收藏。

## 第 0–2 章

| 资料 | 类型 | 为什么现在读 |
| --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · Vaswani et al., 2017 | 论文 | Transformer 原文。重点看 §3.2 Scaled Dot-Product Attention 与因果 mask 的位置。 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · Jay Alammar | 博客 | 张量形状的视觉对照。第 0 章之前花 30 分钟过一遍。 |
| [nanoGPT](https://github.com/karpathy/nanoGPT) · Karpathy | 代码 | 第 0 章实战的参照实现，比 HuggingFace 的抽象层更适合拆。 |
| [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) · Karpathy, 2023 | 视频 | 和 nanoGPT 配套。对「为什么是 decoder-only」讲得很清楚。 |
| **[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)** · kipply | 博客 | **必读。** KV 显存、decode 带宽上限、batch 何时有用，全部从算术推出来。 |
| [LLM Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) · Lilian Weng, 2023 | 博客 | 把量化、投机解码、并行放在一张地图上，后续章节的目录可以对照这篇。 |
| **[Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)** · Horace He | 博客 | **必读。** Roofline、算术强度、为什么「算得动但搬不动」。第 2 章的骨架。 |

## 后续章节会用到

现在不必精读，知道它们在地图上的位置即可。

| 资料 | 对应章节 |
| --- | --- |
| [FlashAttention](https://arxiv.org/abs/2205.14135) · Dao et al. | 第 3 章 |
| [GPTQ](https://arxiv.org/abs/2210.17323) / [AWQ](https://arxiv.org/abs/2306.00978) / [SmoothQuant](https://arxiv.org/abs/2211.10438) | 第 4 章 |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) · OSDI'22 | 第 5 章 |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) · SOSP'23 | 第 5 章 |
| [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) | 第 5 章 |
| [Speculative Decoding](https://arxiv.org/abs/2211.17192) · Leviathan et al. | 第 6 章 |
| [DistServe](https://arxiv.org/abs/2401.09670) · OSDI'24 | 第 8 章 |
| [Mooncake](https://arxiv.org/abs/2407.00079) · FAST'25 | 第 8 章 |
| [System-Aware KV Cache Optimization](https://aclanthology.org/2026.findings-acl.1916.pdf) · ACL'26 Findings | 第 8 章综述 |

## 代码仓库（后续实验）

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [gpt-fast](https://github.com/pytorch-labs/gpt-fast)
- [Mooncake](https://github.com/kvcache-ai/Mooncake)
