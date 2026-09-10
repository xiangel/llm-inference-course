# 《大模型推理系统》

副标题：从 Transformer、KV Cache 到 vLLM 与万卡推理

**LLM Inference Systems** — From Transformer and KV Cache to vLLM and Large-Scale Inference Infrastructure

这是一本推理系统构建书，不是 Transformer 入门、vLLM API 教程或 CUDA 手册。站点用 [Astro Starlight](https://starlight.astro.build/) 构建。

线上（GitHub Pages）：[xiangel.github.io/llm-inference-course](https://xiangel.github.io/llm-inference-course/)  
仓库：[github.com/xiangel/llm-inference-course](https://github.com/xiangel/llm-inference-course)

当前开放：**第 0 篇**（这本书讲什么）与 **第一篇**（理解 LLM 推理）。第二篇起正文未写，[后续篇章目录](src/content/docs/later-chapters.md) 已列出各节标题。

## 本地预览

需要 Node.js 18+（CI 使用 22）和 Python 3.10+（示例需要 NumPy）。

```bash
npm install
npm run dev          # http://localhost:43217
python3 -m pip install -r requirements-examples.txt
python3 -m unittest tests.test_ch00_ch01 -v
```

```bash
npm run build        # 产出 dist/
npm run preview
```

## GitHub Pages

推送到 `main` 后，Deploy GitHub Pages 会构建 `dist/`。仓库 Pages 路径是 `/llm-inference-course/`。

## 仓库结构

```
src/content/docs/     书籍正文（Starlight）
examples/             可运行教学脚本（NumPy）
tests/                对教学数字与算法的单测
mini-vllm/            第五篇教学引擎（占位）
```

## 进度

| 篇 | 状态 |
| --- | --- |
| 第 0 篇 这本书讲什么 | 已开放 |
| 第一篇 理解 LLM 推理（1.1–1.4） | 已开放 |
| 第二篇 Prefill、Decode 与 KV Cache | 节标题已排，正文写作中 |
| 第三篇 GPU 与 Attention | 节标题已排，正文写作中 |
| 第四篇 Batch、调度与分页 KV | 节标题已排，正文写作中 |
| 第五篇 从零实现 Mini-vLLM | 节标题已排，正文写作中 |
| 第六篇 走进 vLLM | 节标题已排，正文写作中 |
| 第七篇 GPU Kernel 与性能优化 | 节标题已排，正文写作中 |
| 第八篇 量化 | 节标题已排，正文写作中 |
| 第九篇 多 GPU 推理 | 节标题已排，正文写作中 |
| 第十篇 主流推理引擎 | 节标题已排，正文写作中 |
| 第十一篇 高级推理技术 | 节标题已排，正文写作中 |
| 第十二篇 生产级推理平台 | 节标题已排，正文写作中 |
| 第十三篇 性能工程 | 节标题已排，正文写作中 |
| 第十四篇 成本与容量规划 | 节标题已排，正文写作中 |
| 第十五篇 万卡推理系统 | 节标题已排，正文写作中 |
| 第十六篇 构建教学生产架构 | 节标题已排，正文写作中 |
