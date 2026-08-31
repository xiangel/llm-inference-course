---
title: 第 1 章 · Token 与上下文
description: 看懂 Tokenizer 如何把文本变成 ID，以及 token 数如何影响上下文限制、成本与速度。
---

# 第 1 章 · Token：模型眼中的文本

**学时** 1–2 小时 · **需要** Python 与免费 Colab · **本章不讲** Attention、向量数学和 KV Cache

一句话回答：**模型为什么不直接读文字，而要先把文字拆成一串 token ID？**

<Checklist
  slug="01-tokens"
  :items="[
    { id: 'explain-token', label: '能解释 token 不是字也不是单词' },
    { id: 'compare-tokenizers', label: '在 Colab 中比较中英文的 token 数' },
    { id: 'context-budget', label: '能用 token 预算解释上下文为什么会满' },
    { id: 'quiz-1-new', label: '完成本章自测' }
  ]"
/>

## 先看问题

两个看似相同的需求，价格和速度可能不同：

```text
“请总结这篇文章。”              → 很短
“请总结这篇文章：<粘贴 50 页 PDF>” → 很长
```

模型不是按字符数、字数或文件大小来处理输入，而是按 **token 数**。上下文窗口、输入计费、首字等待时间和 KV Cache 的增长，都与 token 直接有关。

## 用一个比喻理解

Tokenizer 像压缩词典。词典里没有必要只收录单个字母或整词；它会收录常见片段。

例如英文 `unbelievable` 可能被拆为 `un`、`believ`、`able`；中文的一个字有时是一个 token，有时会和相邻文字按另一种方式组合。模型看见的不是这些字符串，而是它们在词典里的编号。

这就是同样长的一段中英文，token 数经常不同的原因。

## 图：从文字到 ID

<TokenFlow />

## 拆开看

### 第一步：Tokenizer 有自己的词典

每个模型通常带有自己的 Tokenizer。它定义：

- 哪些文字片段是合法 token；
- 每个片段对应哪个整数 ID；
- 文本开头、结尾、填充等特殊标记怎么表示。

模型权重和 Tokenizer 必须配套。把 Qwen 的 Tokenizer 换给 Llama 权重，就像用错误的字典翻译机器指令：数字虽然能输入，但含义错了。

### 第二步：上下文窗口是一份 token 预算

模型都有最大上下文长度。它要同时容纳：

```text
系统提示词 + 历史消息 + 当前问题 + 预计生成的回答
```

这些都按 token 计数。若总数超过模型限制，服务可能拒绝请求、截断前文，或减少最大输出长度。

因此 API 中的 `max_tokens` / `max_completion_tokens` 不是“想要多少字”，而是“预留多少个输出 token”。应用程序应在发送请求前留出回答预算。

### 第三步：token 数也是推理成本的起点

更多输入 token 通常意味着：

- Tokenizer 要处理更长的文本；
- 模型首次开始回答前，需要处理更长的 prompt；
- 后续章节里的 KV Cache 会存下更多历史状态。

不需要在本章计算显存。先记住：**token 数是推理系统最基础的容量单位。**

## 动手：观察不同文本如何分词

点击顶部 Colab 链接。Notebook 使用一个公开 Tokenizer，分别处理中文、英文、代码和带空格的文本。它会打印：

```text
原文
→ token 片段
→ token IDs
→ token 总数
```

核心代码：

<a class="colab-link" href="https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/01_tokens_and_context.ipynb" target="_blank" rel="noreferrer">在 Google Colab 打开并运行这段代码 ↗</a>

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "Hello, world!"
ids = tokenizer.encode(text)
pieces = [tokenizer.decode([token_id]) for token_id in ids]

print(pieces)
print(ids)
```

`encode()` 的输入是字符串，输出是整数 ID 列表。`decode()` 把 ID 还原成文字；这里每次只传一个 ID，方便观察每个 token 对应的片段。

**试着改：**

1. 比较一段等义的中英文，记录 token 数；
2. 把一段 Python 代码粘进去，观察空格和换行会怎样；
3. 把 `gpt2` 换成另一个公开 Tokenizer，比较拆分结果。

## 连接真实系统

- [Hugging Face Tokenizers](https://huggingface.co/docs/transformers/main_classes/tokenizer) 说明了 `encode`、`decode` 和批量输入。
- vLLM 会在推理请求进入执行核心前处理输入；后续源码阅读会对照 `vllm/v1/engine/input_processor.py`。
- API 服务的上下文限制与输出预算应以模型卡和官方文档为准，不能从字符数猜测。

## 常见误解

- token 数不是固定的“字符数除以四”。这是特定语言和特定 Tokenizer 下的粗略经验，不能作为容量检查。
- `max_tokens=100` 不是最多输出 100 个汉字，也不是最多输出 100 个英文单词。
- 把完整聊天历史每次都原样发送很方便，但历史越长，首字时间和成本越高。

## 小结

- Tokenizer 把文字转为模型可处理的整数 ID。
- Token 是词典片段，不等于字符或单词。
- 上下文、成本和性能首先都按 token 计数。

## 参考资料

- [Hugging Face Course：Tokenizer](https://huggingface.co/learn/nlp-course/chapter2/1) — 分词器的官方入门。
- [Hugging Face：Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer) — 本章 Colab 的 API 参考。
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — 可视化理解 token 的辅助工具；具体生产规则仍应以目标模型 Tokenizer 为准。

## 自测

<Quiz :questions="[
  {
    prompt: 'Token 最准确的定义是？',
    options: ['一个汉字', '一个英文单词', 'Tokenizer 词典中的一个文本片段及其整数 ID', '一个字符的 UTF-8 字节'],
    answer: 2,
    explanation: 'token 是词典中的片段；它可能是一个字、一个词的一部分、空格或特殊标记。'
  },
  {
    prompt: '向模型发送很长的聊天历史，最可能直接增加哪项？',
    options: ['模型词典大小', '输入 token 数和首次响应等待', '模型训练数据量', 'GPU 的物理显存容量'],
    answer: 1,
    explanation: '长历史首先变成更多输入 token，因此需要更多 Prefill 工作；后面还会占用更多 KV Cache。'
  },
  {
    prompt: '为什么不能随意把一个模型的 Tokenizer 换成另一个模型的？',
    options: ['Tokenizer 只能在 CPU 上运行', '不同 Tokenizer 可能把同一文字映射为不同 ID，权重不认识这些含义', 'Token 数会永远变成零', '模型会自动重新训练'],
    answer: 1,
    explanation: '模型在训练时学到的是特定 ID 与含义的对应关系；错误 Tokenizer 会造成错误输入。'
  }
]" />
