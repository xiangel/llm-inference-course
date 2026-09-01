---
title: 自回归生成与采样
description: 理解语言模型如何一个 token 接一个 token 地输出，以及 temperature、top-k、top-p 如何改变选择。
---

# 自回归生成与采样

**建议** 用小 Colab 亲手改一次采样参数。

## 为什么要关心它

同一个 prompt 有时应给稳定答案（例如提取 JSON），有时应提供多个合理措辞（例如写标题）。模型每一步给出的只是“下一个 token 的候选倾向”；服务端的生成策略决定怎样从中选一个。

这直接影响可复现性、事实风险、格式稳定性、测试方式和用户体验。它不是“模型参数越大越好”的问题，而是每次请求都要做的产品选择。

## 一个朴素心智模型

语言模型像一个只会完成下一格的预测器：读完已有 token，给词表中每个候选一个未归一化分数（logits），选出一个 token，附到末尾，再重复。因为新 token 又成为下一步输入，过程叫**自回归**。

“贪心”策略永远选最高分，通常更稳定但可能重复或保守。采样则按候选概率随机抽取，能产生多样结果，但需要限制长尾候选。

## 图：调整一次候选分布

<SamplingLab />

该组件使用示意 logits，不代表任何模型或基准。拖动参数，先观察“候选集合”与“概率如何重新分配”，不要把显示的熵当成真实服务质量指标。

## 跟着一次请求走

假设用户要求“用一句话总结日志”。

1. **Prefill**：模型读入 prompt，输出第一个位置的候选分数。
2. **处理候选**：可用 temperature 改变分布尖锐程度；较低 temperature 更偏向高分项，较高值更分散。
3. **截断选择**：`top-k` 只保留概率最高的 k 个候选；`top-p` 从高到低保留累计概率达到 p 的最小集合。两者可同时使用，但会叠加限制。
4. **抽取并追加**：按处理后的概率选一个 token，把它追加到序列。
5. **停止检查**：遇到结束 token、达到输出上限，或命中应用定义的停止序列时结束；否则进入下一轮 Decode。

工程上应记录模型版本、prompt、随机种子（若 API 支持）、采样设置和输出上限。即使固定种子，跨硬件或服务版本也未必保证逐 token 完全一致。

## 可选代码实验

<a class="colab-link" href="https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/04_generation_sampling.ipynb" target="_blank" rel="noreferrer">在 Google Colab 打开本章采样实验 ↗</a>

Notebook 用一组小型 logits 实现 softmax、temperature、top-k 和 top-p，并用固定随机数生成器重复抽样。建议依次：

1. 固定 seed，只改变 temperature，比较输出序列；
2. 令 `top_k=1`，确认它等价于在保留集合内没有随机性；
3. 对一组固定候选分别使用 top-k 与 top-p，打印保留下来的 token。

实验不加载或评测真实大模型；它的目的只是验证“先过滤、再重新归一化、最后抽样”的顺序。

## 连接真实生产系统

- 面向检索、分类、抽取和工具调用时，常使用低随机性，并用 JSON schema / 约束解码做格式保障；采样参数不能替代验证。
- 创作产品可允许更高多样性，但应设置 `max_tokens`、停止条件和滥用防护。
- 流式 API 通常每生成一个 token（或小块）就发送给客户端；首字速度与之后的逐 token 速度是不同指标。

## 常见误解

- **“temperature=0 总是严格确定。”** 不同服务对零值的处理不同，浮点与并行执行也可能影响并列候选；以提供方语义为准。
- **“top-p 是保留固定数量。”** 它保留的是累计概率阈值，候选数量会随分布变化。
- **“采样可以修复错误事实。”** 它只改变候选选择方式，不能补充证据或验证答案。

## 小结

- 自回归生成是“预测一个、追加一个、再预测”。
- temperature 调整尖锐度；top-k 和 top-p 限制参与抽样的候选。
- 生产请求需要把采样设置与停止/验证策略一起设计和记录。

## 自测

<Quiz :questions="[
  { prompt: '自回归生成的下一轮输入包含什么新信息？', options: ['刚刚选出的 token', '训练数据的标签', '新的模型权重', 'GPU 驱动日志'], answer: 0, explanation: '选出的 token 被追加到序列，成为下一轮上下文的一部分。' },
  { prompt: 'top-p 保留候选的依据是？', options: ['固定 token 数', '字符长度', '从高概率开始的累计概率阈值', '词表字母顺序'], answer: 2, explanation: '因此不同 prompt 下保留的候选数可以不同。' },
  { prompt: '对必须符合固定 JSON 结构的输出，最可靠的补充措施是？', options: ['只提高 temperature', '使用约束/校验并在应用侧处理失败', '取消输出上限', '随机更换模型'], answer: 1, explanation: '采样本身不能保证结构正确，必须增加约束或验证。' }
]" />

## 参考资料

- [Hugging Face Transformers：Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) — 官方生成与采样参数说明。
- [Hugging Face Transformers：GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation) — 参数默认值与 API 语义，以实际安装版本文档为准。
- [OpenAI：Reproducible outputs](https://platform.openai.com/docs/guides/production-best-practices/reproducible-outputs) — 说明 seed 与系统变化对可复现性的边界。
