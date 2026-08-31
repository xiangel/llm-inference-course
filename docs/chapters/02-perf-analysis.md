---
title: 第 2 章 · Transformer 全貌
description: 用一张图看懂生成式模型的整体结构，以及 Decoder-only 为什么适合持续生成文本。
---

# 第 2 章 · Transformer 全貌：模型怎样续写文本

**学时** 1–2 小时 · **需要** 阅读即可 · **本章不讲** Attention 计算细节、矩阵乘法和性能调优

一句话回答：**一个生成式 Transformer 如何利用前文，选择下一个 token？**

<Checklist
  slug="02-transformer-overview"
  :items="[
    { id: 'three-families', label: '能区分 Encoder-only、Decoder-only 和 Encoder-Decoder 的用途' },
    { id: 'trace-transformer', label: '能按图复述 Decoder-only 的数据流' },
    { id: 'name-blocks', label: '能说明 Attention、FFN、残差连接各自的工作' },
    { id: 'quiz-2-new', label: '完成本章自测' }
  ]"
/>

## 先看问题

第 0 章说过：模型会反复预测下一个 token。这里的关键问题是：

> 当模型准备续写「今天天气很」时，它怎样知道应该更多参考“天气”，而不是只盯着最后一个“很”？

答案不是把句子塞进一个普通的循环。Transformer 会让当前 token 查看它前面的 token，再用多层处理把这些上下文信息变成下一词的判断。

## 用一个比喻理解

把一句话想成一次多人会议。轮到最后一个词发言时：

- **Attention** 让它翻看之前所有发言，并决定哪些信息更相关；
- **FFN** 让它独自消化这些信息；
- **残差连接** 相当于保留原始笔记，避免新总结把旧信息完全覆盖；
- 重复很多轮后，模型根据最终笔记选择最可能的下一个 token。

这不是人类理解语言的方式，但它是理解 Transformer 数据流的好起点。

## 图：从文字到下一个 token

<TransformerFlow />

先从左到右看一次，再注意底部的回路：新生成的 token 会被接回输入，因此模型可以持续续写。

## 拆开看

### 第一步：三类 Transformer 不做同一种事

| 类型 | 能看哪里 | 擅长任务 | 例子 |
| --- | --- | --- | --- |
| Encoder-only | 可同时看整段输入 | 分类、检索、抽取 | BERT |
| Decoder-only | 只能看当前位置之前 | 对话、续写、代码生成 | GPT、Llama、Qwen |
| Encoder-Decoder | Encoder 看输入；Decoder 生成输出 | 翻译、摘要 | T5 |

本课程关注 Decoder-only，因为大多数聊天与代码生成模型属于这一类。

“只能看前文”不是功能限制，而是生成能力的前提。模型在预测第 5 个 token 时不能先看第 6 个 token，否则训练和实际生成就不一致。

### 第二步：一个 Transformer Block 有什么

真实模型会把同一个 Block 堆叠几十层。先看一层：

<DecoderDiagram />

| 部件 | 不用数学的解释 | 为什么需要它 |
| --- | --- | --- |
| RMSNorm | 整理数值尺度 | 让多层计算更稳定 |
| Q / K / V 投影 | 为“提问、索引、内容”准备不同版本的信息 | Attention 用它决定关注谁 |
| 因果 Attention | 当前 token 查看之前 token | 把上下文带进当前判断 |
| FFN / SwiGLU | 单独加工每个 token 的信息 | 学习更复杂的模式 |
| 残差连接 | 将输入直接加回输出 | 保留旧信息，让深层 Block 更容易工作 |

不需要在此刻记住 Q、K、V 的具体计算方式。下一章会用一段很小的 Python 代码可视化“当前 token 只能看前文”这件事。

### 第三步：模型如何停止

模型每轮给词表中每个 token 一个分数；采样规则从中选出一个新 token。直到发生下面任一条件：

- 选到结束标记；
- 达到调用方设定的最大输出 token 数；
- 应用检测到停止字符串。

第 4 章会专门讨论 temperature、top-k、top-p 如何影响“从分数中选哪个”。

## 连接真实系统

- Llama、Qwen、GPT 是 Decoder-only 模型：它们通过不断续写 token 产生回答。
- [Hugging Face Transformer 架构说明](https://huggingface.co/docs/course/chapter1/4) 对比了三类模型和对应任务。
- 第 9 章以后启动 vLLM 时，服务管理的是多条这样的 Decoder-only 生成循环。

## 常见误解

- Transformer 不是数据库。它并不会在推理时逐句搜索训练文本。
- Attention 不等于“模型完全理解了上下文”。它是把上下文信息混合进当前表示的一种机制。
- 一个 Block 不是一轮生成。模型通常是在所有 Block 都跑完后，才得到一个新 token。
- Decoder-only 可以生成文本，是因为它持续预测下一个 token，不是因为它一次构造完整句子。

## 小结

- 生成式模型通常使用 Decoder-only Transformer。
- 文字经过 Tokenizer、Embedding、多层 Block 和输出层，得到下一个 token。
- 每层 Block 的核心是：用 Attention 看上下文，用 FFN 加工信息，用残差连接保留输入。
- 下一章会开始深入 Attention，但仍先从可视化和小代码实验入手。

## 参考资料

- [Hugging Face：How do Transformers work?](https://huggingface.co/docs/course/chapter1/4) — 重点阅读三种架构和 Attention 的直觉介绍。
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 图形化补充材料；无需一次读完。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始论文，当前只看结构图即可；细节会在后续章节分解。

## 自测

<Quiz :questions="[
  {
    prompt: '为什么聊天模型通常采用 Decoder-only 架构？',
    options: ['它只能处理英文', '它可以根据已生成的前文持续预测下一个 token', '它不需要 Tokenizer', '它总是比其他架构参数更少'],
    answer: 1,
    explanation: 'Decoder-only 的因果限制正好匹配“根据已有前文续写”的生成过程。'
  },
  {
    prompt: '在一个 Transformer Block 中，Attention 最主要负责什么？',
    options: ['把文字变成 token IDs', '让当前 token 参考前文中相关的信息', '决定 HTTP 请求的路由', '把模型保存到磁盘'],
    answer: 1,
    explanation: 'Tokenizer 负责文字转 ID；Attention 的工作是把上下文混合进当前位置。'
  },
  {
    prompt: '模型生成一个新 token 后，下一轮会怎样？',
    options: ['立刻清空之前的上下文', '把新 token 接到已有输入后继续预测', '重新训练模型', '停止使用 Tokenizer'],
    answer: 1,
    explanation: '新 token 成为前文的一部分，所以模型能够持续续写。'
  }
]" />
