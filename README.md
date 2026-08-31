# 大模型推理课程

一门从第一性原理讲到生产系统的 LLM 推理课。站点用 [VitePress](https://vitepress.dev/) 构建，正文是 Markdown，交互实验是 Vue 组件。

仓库：[github.com/xiangel/llm-inference-course](https://github.com/xiangel/llm-inference-course)  
线上：[xiangel.github.io/llm-inference-course](https://xiangel.github.io/llm-inference-course/)

目前开放完整 **15 章**：从一次 LLM 请求、Token、Transformer 与 KV Cache，到 vLLM 服务化、nano-vLLM 源码阅读和毕业项目。

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

## 访问统计

站点使用免费的 [Cloudflare Web Analytics](https://developers.cloudflare.com/web-analytics/) 统计页面浏览、访客、来源、热门页面和 Core Web Vitals。它通过手动 beacon 接入 GitHub Pages，并开启 SPA 路由统计。

浏览器广告拦截器可能阻止统计脚本；这不会影响课程网站本身。

## 课程结构

| 阶段 | 章节 | 状态 |
| --- | --- | --- |
| A 基础直觉 | 0 请求 · 1 Token · 2 Transformer · 3 Attention | 已开放 |
| B 推理基础 | 4 采样 · 5 KV Cache · 6 性能指标 · 7 GPU 瓶颈 | 已开放 |
| C 推理服务与优化 | 8 部署 vLLM · 9 连续批处理 · 10 PagedAttention · 11 量化与投机解码 | 已开放 |
| D 源码与毕业项目 | 12 nano-vLLM · 13 vLLM V1 · 14 毕业项目 | 已开放 |
