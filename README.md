# 大模型推理课程

一门从第一性原理讲到生产系统的 LLM 推理课。站点用 [VitePress](https://vitepress.dev/) 构建，正文是 Markdown，交互实验是 Vue 组件。

仓库：[github.com/xiangel/llm-inference-course](https://github.com/xiangel/llm-inference-course)  
线上：[xiangel.github.io/llm-inference-course](https://xiangel.github.io/llm-inference-course/)

目前开放 **第 0–2 章**：Transformer 预备、KV Cache、性能分析（Roofline）。

## 本地预览

需要 Node.js 18+。

```bash
npm install
npm run dev
```

浏览器打开终端里给出的地址（默认 `http://localhost:43217`）。

```bash
npm run build    # 产出 docs/.vitepress/dist
npm run preview  # 预览构建结果
```

## GitHub Pages

推送到 `main` 后，[Deploy GitHub Pages](https://github.com/xiangel/llm-inference-course/actions) 会构建并发布。第一次需要在仓库 **Settings → Pages → Source** 选 **GitHub Actions**。

项目站路径是 `/llm-inference-course/`，构建时已写入，不必再设变量。

## 课程结构

| 阶段 | 章节 | 状态 |
| --- | --- | --- |
| A 单请求原理 | 0 预备知识 · 1 KV Cache · 2 性能分析 | 已开放 |
| B 单卡优化 | 3 Attention kernel · 4 量化 | 即将上线 |
| C 服务化 | 5 调度与分页 · 6 投机解码 | 即将上线 |
| D 集群与前沿 | 7 分布式 · 8 PD 分离 · 9 Reasoning · 10 毕业项目 | 即将上线 |
