---
title: 第 0 章 · 一次 LLM 请求
description: 从一段 Prompt 到逐字出现的回答：模型、权重、推理服务和 Tokenizer 各做什么。
---

# 第 0 章 · 一次 LLM 请求发生了什么

**学时** 1–2 小时 · **需要** Python 与免费 Colab · **本章不讲** Transformer 内部和 GPU 优化

一句话回答：**当 Python 程序把一段 Prompt 发给模型后，谁把它变成答案？**

<Checklist
  slug="00-first-request"
  :items="[
    { id: 'name-parts', label: '能说清模型、权重、Tokenizer 和推理服务的区别' },
    { id: 'run-first-request', label: '在 Colab 中跑通一次最小文本生成' },
    { id: 'trace-flow', label: '能按图复述请求和响应的路径' },
    { id: 'quiz-0-new', label: '完成本章自测' }
  ]"
/>

## 先看问题

你可能已经用过这样的代码：

```python
answer = client.chat.completions.create(
    model="some-model",
    messages=[{"role": "user", "content": "解释什么是 KV Cache"}],
)
```

这几行代码很好用，但隐藏了很多东西：为什么模型需要知道 Token 数？为什么回答是一个字一个字出现的？模型文件、Tokenizer 和 API 服务是不是同一个东西？

这一章只把这些角色分开。先不关心它们内部的数学。

## 用一个比喻理解

把 LLM 当成一家餐厅：

- **你的 Python 程序**是顾客，提交订单（Prompt）。
- **推理服务**是服务员，接收请求、排队、把结果流式送回。
- **Tokenizer**是点单系统，把文字翻译成厨房能处理的编号。
- **模型权重**是厨师积累的经验；它不存你的问题，而是决定看到编号后更可能接什么编号。
- **GPU/CPU**是厨房设备，真正执行计算。

服务员、点单系统和厨师不是同一个人；这也是为什么生产系统会分别优化 API、Tokenization、调度和模型计算。

## 图：一次请求的生命周期

<RequestLifecycleFlow />

图中最后一段会重复：模型先给出一个新 token；服务把它转回文字并发给你；该 token 再成为下一轮输入的一部分。于是网页上的回答看起来像“逐字流出”。

## 拆开看

### 第一步：Prompt 不是模型的直接输入

你写的是 Python 字符串。模型只能处理数字，所以服务先交给 Tokenizer。Tokenizer 会把字符串拆成 token，并把每个 token 映射为整数 ID。

不要把 token 理解成“一个汉字”或“一个英文单词”。它是模型词典中的片段。第 1 章会实际打印这些片段。

### 第二步：权重不是知识库

模型权重是一大组已训练好的数字文件。它们决定“在当前上下文后面，哪个 token 更可能出现”。

这和检索系统不同：检索系统会在文档库中找原文；模型权重是在计算中产生下一个 token 的分数。一个真实产品可能同时使用两者，但它们是不同组件。

### 第三步：推理服务负责把模型变成 API

Hugging Face 的 `pipeline()` 可以让你在一个 Python 进程里直接调用模型。vLLM 则把模型包装成可并发访问的服务，并提供与 OpenAI 兼容的接口。

以后会遇到的 TTFT、并发数、KV Cache 和连续批处理，都属于“如何把模型服务给很多请求”的问题，而不是 Prompt 字符串的问题。

## 动手：运行一个极小模型

点击下方代码片段上方的 Colab 按钮。Notebook 会下载一个很小的公开文本生成模型，并运行一次生成。它同时打印：

1. 你传入的 Prompt；
2. Tokenizer 产生的 token ID；
3. 模型补出的文本。

这不是为了得到高质量回答，而是为了看到完整链路。免费 CPU 可以运行。

核心代码只有两步：

<a class="colab-link" href="https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/00_first_llm_request.ipynb" target="_blank" rel="noreferrer">在 Google Colab 打开并运行这段代码 ↗</a>

```python
from transformers import pipeline

generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")
result = generator("Python 程序发送的 Prompt：", max_new_tokens=20)
print(result[0]["generated_text"])
```

`pipeline()` 帮你加载了 Tokenizer 和权重。输入是字符串；返回值是包含生成文本的列表。下一章会拆开 `pipeline()`，直接查看 Tokenizer 的输出。

**试着改：**

1. 把 Prompt 换成英文，观察输出；
2. 把 `max_new_tokens` 从 20 改成 5；
3. 在 Colab 中打印 `generator.tokenizer`，确认它是一个独立对象。

## 连接真实系统

- [Hugging Face pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) 适合本地快速调用。
- [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) 展示了如何把模型启动为 OpenAI 兼容服务。
- 后续源码章节会从 vLLM 的 `LLMEngine.add_request()` 开始，追踪这张图里的“服务收到 Prompt”。

## 常见误解

- “模型 API”不等于“模型”。API 是调用入口；权重才是模型运行所需的文件。
- Tokenizer 不是可有可无的预处理。权重训练时使用了特定词典，替换它会让输入数字失去意义。
- 流式输出不代表模型一次计算出整段答案。通常它是逐 token 生成、逐 token 返回。

## 小结

- 一次请求至少涉及程序、服务、Tokenizer、权重和计算硬件。
- 模型预测下一个 token；服务负责把 token 作为文字交给你。
- 下一章先解决一个基础问题：token 到底是什么。

## 参考资料

- [Hugging Face Course：Introduction](https://huggingface.co/learn/llm-course/chapter1/1) — 从 Python 调用模型开始的官方课程。
- [Hugging Face Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) — 本章 Colab 使用的高层 API。
- [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html) — 后续服务化章节的官方入口。

## 自测

<Quiz :questions="[
  {
    prompt: '你把 Python 字符串交给 LLM 服务后，最先必须发生的转换是什么？',
    options: ['字符串变为 token IDs', '直接读取 GPU 显存', '把回答写进向量数据库', '把模型重新训练一次'],
    answer: 0,
    explanation: '模型处理的是数字 ID，不是 Python 字符串。Tokenizer 负责这一步。'
  },
  {
    prompt: '哪一项最准确地描述了“模型权重”？',
    options: ['用户问题的原始文本库', '决定下一个 token 分数的一组训练后数字', 'HTTP API 的地址', '浏览器流式显示组件'],
    answer: 1,
    explanation: '权重是训练产生的参数；API、文档库和 UI 都是另外的系统部件。'
  },
  {
    prompt: '回答以流式形式出现，通常意味着什么？',
    options: ['模型已经一次生成完整答案，只是网络故意拆开', '服务通常在每个新 token 生成后就返回它', 'Tokenizer 正在重新训练', '模型不需要计算'],
    answer: 1,
    explanation: '自回归模型通常逐 token 生成；流式服务能把每一步尽早发给用户。'
  }
]" />
