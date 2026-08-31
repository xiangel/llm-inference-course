---
title: Attention：当前 token 如何看前文
description: 用不含公式的方式理解 Transformer 怎样从上下文取信息，以及它为何是生成质量与延迟的共同起点。
---

# Attention：当前 token 如何看前文

**适合** 会写 Python、刚接触 Transformer 的工程师 · **本章目标** 说清一个新 token 如何利用此前文字；不要求推导矩阵公式。

## 为什么要关心它

用户问“巴黎是法国的首都。那么它在哪里？”，模型要把“它”与“巴黎”关联，而不是只看最后几个字。代码补全也一样：补一个变量名时，模型可能要回看函数参数、导入和上面的分支。

这件“回看并挑重点”的工作就是 Attention。它决定长 prompt 中哪些信息有机会影响当前输出，也解释了为什么长上下文会带来计算与显存压力。

## 一个朴素心智模型

把上下文想成一叠索引卡。每张卡代表一个已读 token，并带着两样东西：

- **索引线索（Key）**：这张卡大致在讲什么；
- **内容（Value）**：若选中它，应该取回的实际信息。

当前 token 先提出一个“我要找什么”的查询（Query），再给所有旧卡打相关性分数，重点读取分数高的卡。读取结果与当前位置自己的信息合并，交给下一层继续处理。

这不是数据库的精确 `WHERE` 查询：它是连续的、可学习的软选择，多个位置都能贡献一点信息。

## 图：一次生成中的两种工作

<PrefillDecodeDiagram />

图中 Prefill 先处理完整 prompt；之后每次 Decode 只加入一个 token。无论哪种阶段，Attention 的职责都相同：让当前位置可参考允许看到的前文。生成时的“因果遮罩”会禁止它偷看未来 token。

## 跟着一次注意力走

设 prompt 是“订单号 A17 已退款，给用户发送通知”。模型正在生成下一个词。

1. Tokenizer 已把文本变成 token；每层为各 token 产生内部表示。
2. 当前生成位置形成 Query，可粗略理解为“现在需要与什么信息有关？”。
3. 它对前文每个位置的 Key 评估相关性。“订单号”“退款”“用户”可能比标点更相关。
4. 它按相关性混合对应的 Value，得到一份包含前文重点的信息。
5. 后续网络把这份信息转成下一个 token 的候选分数；采样器再决定实际输出哪个 token。

多头 Attention 相当于同时让多组不同的“阅读视角”做这件事：一组可能偏向指代关系，另一组可能偏向语法或代码作用域。不要把单个头的行为当作人类可读的规则；模型通常把信息分散在层和头之间。

## 动手观察（可选）

对本章而言，纸上追踪比运行大模型更有帮助。想试验可在 Colab 新建 notebook，加载一个很小的公开 Transformer，打印 token 并查看某一层的 attention 权重。重点只问两件事：当前行能否看到未来列？哪些前文位置权重较高？

权重图是调试线索，不是因果证明。一个较高权重不自动等于“模型因为这个词才做出答案”。

## 连接真实生产系统

- 聊天服务会把系统提示、历史和当前请求一起放入上下文；模型只能从实际送入的 token 中 Attention。
- RAG 把检索到的片段放进 prompt，目的正是给 Attention 可读取的证据；检索错了、排序差了，模型也无法凭空读取原文。
- 长 prompt 的 Prefill 与逐 token Decode 有不同资源形状。下一章解释如何选择 token，随后 KV Cache 章节解释如何避免重复处理旧卡。

## 常见误解

- **“Attention 就是搜索。”** 它提供可微的相关性混合，不保证精确匹配，也不等于外部检索。
- **“模型能关注所有内容，所以必然记得。”** 能访问不等于可靠使用；位置、提示词结构、训练和上下文噪声都会影响结果。
- **“只要看 Attention 图就能解释模型。”** 图只展示一个中间信号，不能单独证明因果或正确性。

## 小结

- Attention 让当前 token 从允许的前文位置取回相关信息。
- Query 像问题，Key 像索引线索，Value 像可取回内容。
- 生成不能看未来；长上下文的处理成本是后续性能优化的根源。

## 自测

<Quiz :questions="[
  { prompt: '在生成第 t 个 token 时，因果 Attention 可以读取什么？', options: ['未来 token', '当前 token 及此前允许的位置', '训练集全文', '服务器磁盘'], answer: 1, explanation: '因果遮罩阻止当前位置读取未来 token。' },
  { prompt: 'Key 和 Value 的朴素比喻分别是什么？', options: ['输出和标签', '索引线索和卡片内容', 'GPU 和 CPU', '字符和字节'], answer: 1, explanation: 'Key 用于判断相关性，Value 是被混合取回的信息。' },
  { prompt: '为什么 RAG 要把检索结果放进 prompt？', options: ['让 tokenizer 更小', '给 Attention 可读取的外部证据', '关闭因果遮罩', '减少模型层数'], answer: 1, explanation: '未放进上下文的文档不能被本次推理直接读取。' }
]" />

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 的原始论文；可先读图和摘要，公式留待需要时再回看。
- [Hugging Face：LLM Course，Transformer](https://huggingface.co/learn/llm-course/chapter1/4) — 面向实践者的 Transformer 结构导读。
- [PyTorch `scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — 官方 Attention API；其中的 `is_causal` 对应生成时不能看未来。
