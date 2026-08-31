---
layout: home
hero:
  name: 大模型推理
  text: 从第一性原理到生产系统
  tagline: 一门循序渐进的课程。先搞清楚一次请求在算什么、瓶颈在哪；再谈 kernel、量化、服务化与集群。目前开放第 0–2 章。
  actions:
    - theme: brand
      text: 从第 0 章开始
      link: /chapters/00-prerequisites
    - theme: alt
      text: 参考资料
      link: /resources
features:
  - title: 理论先于框架
    details: 每一章先用手写玩具版建立直觉，再用 vLLM / SGLang / llama.cpp 对照生产系统。框架会改，算术不会。
  - title: KV Cache 是主线
    details: 手写它、量化它、分页管理它、前缀复用它、跨机传输它。整门课围着内存带宽瓶颈转。
  - title: 能在网页里算
    details: 第 1 章有 KV 显存计算器与采样实验室，第 2 章有 Roofline 图。公式不只是写在纸上。
---

## 为什么这样排

大模型推理不是「把 `model.generate()` 跑通」。生产环境里真正值钱的，是把 **TTFT、每 token 延迟、吞吐、显存** 同时讲清楚，并且知道每一档优化对应哪一个瓶颈。

本课按真实推理栈分成四层：

| 层次 | 章节 | 对应系统组件 |
| --- | --- | --- |
| 算法层 | 第 0–1 章 | 自回归、采样、KV Cache |
| 算子 / 硬件层 | 第 2–4 章 | Roofline、FlashAttention、量化 |
| 单机系统层 | 第 5–6 章 | 调度器、投机解码 |
| 集群系统层 | 第 7–9 章 | 并行、PD 分离、KV 池 |

现阶段先把阶段 A 做厚：没有 KV Cache 的形状和 Roofline 的判断，后面谈 vLLM 只是在背名词。

## 学习路线

<Roadmap />

## 怎么用这个站点

- 左侧目录可以随时跳章。正文是 Markdown，代码块自带复制。
- 公式用 $\mathrm{\TeX}$ 渲染。交互实验是页面里的 Vue 组件，不需要另开 Notebook。
- 每章末尾有自测和学习清单；清单进度存在你的浏览器本地。
- 没有 GPU 也能把第 0–2 章学完。第 0 章的代码在 CPU 上就能跑；第 2 章的 profiler 实验如果没有 CUDA，就用理论手算代替。
- 发布到 GitHub Pages 时，只需改 `VITEPRESS_BASE`（仓库名）。详见仓库 README。

## 建议节奏

1. 第 0 章：把 Decoder-only 的张量形状写下来，跑通最小 GPT。
2. 第 1 章：亲手给前向加上 KV Cache，用计算器对比 8B / 70B。
3. 第 2 章：用 Roofline 解释「为什么 Decode 吃的是带宽」。
4. 停下来。如果这三章的公式你能闭卷写出来，再继续后面的框架章节。
