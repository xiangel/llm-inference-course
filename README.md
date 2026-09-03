# 《大模型推理系统》

副标题：从 Transformer、KV Cache 到 vLLM 与万卡推理

**LLM Inference Systems** — From Transformer and KV Cache to vLLM and Large-Scale Inference Infrastructure

这是一本推理系统构建书，不是 Transformer 入门、vLLM API 教程或 CUDA 手册。站点用 [Astro Starlight](https://starlight.astro.build/) 构建。

线上（GitHub Pages）：[xiangel.github.io/llm-inference-course](https://xiangel.github.io/llm-inference-course/)  
仓库：[github.com/xiangel/llm-inference-course](https://github.com/xiangel/llm-inference-course)

当前开放：**第 0 篇**（这本书讲什么）与 **第一篇**（理解 LLM 推理）。后续篇章在目录中占位。

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
| 第一篇 理解 LLM 推理 | 已开放 |
| 第二篇–第十六篇 | 目录占位，正文写作中 |
