---
title: 第 11 章 · 量化与投机解码
description: 理解两类不同的推理加速手段，以及何时它们会带来反效果。
---

# 第 11 章 · 量化与投机解码

**学时** 1–2 小时 · **需要** 阅读即可 · **本章不需要** Colab 或 GPU

一句话回答：**在不改业务 API 的前提下，怎样减少每个 token 的成本或等待？**

## 为什么

推理慢有两种常见原因：搬运大权重和 KV 数据昂贵，或主模型必须逐 token 做大量工作。量化主要缩小表示与搬运成本；投机解码主要减少主模型逐步验证的轮数。它们不是同一个开关，也都不是免费午餐。

## 心智模型

量化像用更紧凑的编号记录货物：车能装更多，但细节可能损失，且不同硬件未必更快。投机解码像让较快的助手先写草稿，主编一次审多句：草稿越常被接受越划算；若常被否决，审稿和草稿都成了额外工作。

## 数据流建议

复用第 8 章的 <code>&lt;RequestLifecycleFlow /&gt;</code>：量化影响“模型权重”与 KV 的表示；投机解码在“预测下一个 token”前插入 draft 生成和 target 验证分支。没有新增图组件时，请在纸上画出这两处变化，而不要把它们混为一个优化。

## 按步骤理解

### 1. 先选对瓶颈

记录 TTFT、每输出 token 时间、吞吐、GPU 显存、质量任务的通过率和错误样例。没有基线就无法判断“更快”是加载快、首字快，还是持续输出快。

### 2. 量化：以更少位表示数值

常见权重格式包括 FP16/BF16、FP8、INT8 和 4-bit 方案；也可量化 KV Cache。实际效果取决于量化算法、校准数据、模型结构、kernel、GPU 和工作负载。更低位宽通常减少显存与带宽压力，但可能降低质量、引入兼容限制，或因转换开销而没有加速。

从可信的目标模型/运行时文档选择已经发布的量化变体；用你的任务集比较质量和延迟。不要把“4-bit”当作统一格式，也不要把一台 GPU 的结果外推到所有 GPU。

### 3. 投机解码：草稿再验证

draft model（或 n-gram/prompt lookup 等候选器）先提出连续 token，target model 验证它们并保持目标模型的生成分布。接受多时，target 每次前进更多 token；接受少时，额外的草稿计算、同步和 KV 管理会抵消收益。

它适合能取得较高接受率、输出较长、服务实现支持且测量显示 decode 为瓶颈的场景。草稿模型过大、请求太短、batch 很小或目标模型本身不慢时，收益可能很小甚至变负。

### 4. 用生产实验而非口号决策

固定模型、提示词集、并发、最大输出和硬件；分别测试基线、量化、投机及二者组合。报告质量门槛、TTFT、ITL、吞吐、显存和失败率。若质量或尾延迟超过 SLO，就不是成功优化。

## 生产连接

运行参数和受支持模型随版本变化。使用 [vLLM quantization 文档](https://docs.vllm.ai/en/latest/features/quantization/) 与 [speculative decoding 文档](https://docs.vllm.ai/en/latest/features/spec_decode.html) 核实当前后端。任何量化权重还应核对模型发布者的许可、校验和与适用硬件。

## 常见误解

- 位宽更低不必然更快，也不保证质量可接受。
- 投机解码不是让小模型替代大模型；target 仍负责验证。
- 两个优化可以组合，但组合效果不能从单项结果相加推出。
- 基准只看 tokens/s 会掩盖首字、尾延迟和质量退化。

## 小结

- 量化主要改变表示、显存和带宽权衡。
- 投机解码以候选 token 换取更少的 target 解码轮次。
- 先量基线，再用真实任务和 SLO 作选择。

## 自测

<Quiz :questions="[
  { prompt: '量化首先试图改善什么？', options: ['训练数据质量', '数值表示、显存和数据搬运成本', 'HTTP 协议版本', '用户提示词含义'], answer: 1, explanation: '量化改变权重或 KV 等运行时表示。' },
  { prompt: '投机解码何时更可能有效？', options: ['候选经常被 target 接受', '草稿永远被拒绝', '没有 target model', '输出恒为零 token'], answer: 0, explanation: '高接受率才能抵消草稿与验证开销。' },
  { prompt: '评估优化最不充分的指标是？', options: ['质量、TTFT、ITL、吞吐、显存', '只看单次 tokens/s', '尾延迟和失败率', '固定工作负载下的对照实验'], answer: 1, explanation: '单一吞吐数字会遗漏体验、稳定性和质量。' }
]" />

## 参考资料

- [vLLM：Quantization](https://docs.vllm.ai/en/latest/features/quantization/) — 当前受支持格式与后端。
- [vLLM：Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 当前运行时选项。
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 投机解码的主要论文。
