# Chapter template

Use this structure as the default. Keep only sections that serve the chapter's learner question.

```md
---
title: 第 N 章 · [主题]
description: [用一句话说明程序员学完能做什么]
---

# 第 N 章 · [主题]

**学时** [范围] · **需要** [Python/免费 Colab/GPU（可选）]

一句话回答：**[本章要解决的真实问题]**

<Checklist ... />

<a class="colab-link" href="[Colab URL]" target="_blank" rel="noreferrer">
  在 Google Colab 打开并运行本章代码 ↗
</a>

## 先看问题
[一个具体场景、输入和可见的失败/成本。]

## 用一个比喻理解
[最多 2 段。不替代技术解释，只帮助建立直觉。]

## 图：发生了什么
<[Topic]Flow />

## 拆开看
### 第一步：[名称]
[执行顺序、输入、输出、为什么要这样做。]

### 第三步：[必要公式；可选]
[先解释它回答的问题，再给变量表、代入实例和边界。]

## 动手：最小实验
[先解释实验会显示什么。]

```python
# 仅放需要运行的最小代码
```

**预计看到：** `[可观察的输出/图]`

**试着改：**
1. [...]
2. [...]

## 连接真实系统
[Hugging Face / vLLM / SGLang / llama.cpp 中的对应概念、文件或命令。]

## 常见误解
- [...]
- [...]

## 小结
- [...]
- [...]

## 参考资料
- [资料标题](URL) — 为什么值得读，建议读哪一节。

## 自测
<Quiz :questions="[...]" />
```

## Colab link format

For this repository:

```text
https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/<notebook>.ipynb
```

The link will work only after the notebook is committed and pushed to `main` on GitHub.
