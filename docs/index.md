---
layout: home
hero:
  name: 大模型推理
  text: 从第一性原理到生产系统
  tagline: 为会 Python、但不熟悉算法和 Transformer 的程序员设计。先从一次请求、Token 和 Transformer 全貌建立直觉，再进入 KV Cache、性能与服务化。
  actions:
    - theme: brand
      text: 从第 0 章开始
      link: /chapters/00-prerequisites
    - theme: alt
      text: 参考资料
      link: /resources
features:
  - title: 从程序员的视角开始
    details: 先看 Prompt 如何成为回答，再拆开 Tokenizer、模型权重和推理服务；不预设算法或深度学习基础。
  - title: 图、代码、再到系统
    details: 每章先用图建立直觉。只有代码能帮助理解时才给可运行 Colab；公式保持最少且提供完整解释。
  - title: 再读真实推理引擎
    details: 先读简化 nano-vLLM，再对照 vLLM 源码。不会把生产级十万行工程当作入门教材。
---

## 为什么这样排

大模型推理不是「把 `model.generate()` 跑通」。生产环境里真正值钱的，是把 **TTFT、每 token 延迟、吞吐、显存** 同时讲清楚，并且知道每一档优化对应哪一个瓶颈。

本课按真实推理栈分成四层：

| 层次 | 章节 | 对应系统组件 |
| --- | --- | --- |
| 基础直觉 | 第 0–3 章 | 请求、Token、Transformer、生成 |
| 推理基础 | 第 4–7 章 | 采样、KV Cache、指标、GPU 瓶颈 |
| 推理服务 | 第 8–11 章 | vLLM、调度、分页、量化与投机解码 |
| 源码与集群 | 第 12–14 章 | nano-vLLM、vLLM V1、并行、PD 分离 |

现阶段先把阶段 A 做厚：先理解模型、Token 和生成循环，后面谈 KV Cache、vLLM 才不会变成背名词。

## 学习路线

<Roadmap />

## 怎么用这个站点

- 左侧目录可以随时跳章。正文是 Markdown，代码块自带复制。
- 只有必要时才用公式；公式后会有变量解释、数值例子和适用边界。交互实验是页面里的 Vue 组件。
- 每章末尾有自测和学习清单；清单进度存在你的浏览器本地。
- 没有 GPU 也能把第 0–2 章学完。第 0、1 章的 Colab 在免费 CPU 上就能跑；第 2 章目前是概念与图解，不需要 Colab。
- 发布到 GitHub Pages 时，只需改 `VITEPRESS_BASE`（仓库名）。详见仓库 README。

## 建议节奏

1. 第 0 章：运行第一个小模型，按图复述请求路径。
2. 第 1 章：在 Colab 中观察文本如何变成 Token IDs。
3. 第 2 章：区分三类 Transformer，并解释 Decoder-only 如何持续续写。
4. 能自然复述这三章后，再进入 Attention 与生成循环。
