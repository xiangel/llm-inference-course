# 大模型推理课程

一门从第一性原理讲到生产系统的 LLM 推理课。站点用 [VitePress](https://vitepress.dev/) 构建，正文是 Markdown，交互实验是 Vue 组件，可以直接发布到 GitHub Pages 或 Vercel。

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

## 发布到 GitHub Pages

1. 仓库 Settings → Pages → Source 选 GitHub Actions。
2. 如果这是 **项目站**（`https://<user>.github.io/<repo>/`），在仓库 Settings → Variables 里增加变量：

   `VITEPRESS_BASE=/<repo>/`

   用户站（`https://<user>.github.io/`）保持 `/` 即可，不必设。
3. 推送到 `main`，`.github/workflows/deploy-pages.yml` 会构建并发布。

也可以不用 Actions，把 `docs/.vitepress/dist` 发到 `gh-pages` 分支。

## 发布到 Vercel

Build Command: `npm run build`  
Output Directory: `docs/.vitepress/dist`  
Install Command: `npm install`

根路径部署时不要设置 `VITEPRESS_BASE`。

## 课程结构

| 阶段 | 章节 | 状态 |
| --- | --- | --- |
| A 单请求原理 | 0 预备知识 · 1 KV Cache · 2 性能分析 | 已开放 |
| B 单卡优化 | 3 Attention kernel · 4 量化 | 即将上线 |
| C 服务化 | 5 调度与分页 · 6 投机解码 | 即将上线 |
| D 集群与前沿 | 7 分布式 · 8 PD 分离 · 9 Reasoning · 10 毕业项目 | 即将上线 |

## 为什么用 VitePress 而不是从零写前端

课程主体是长文、公式、代码和参考文献，文档引擎已经把侧栏、搜索、代码复制、暗色主题、静态导出做完了。需要计算器、采样实验室、自测时，再在 Markdown 里嵌 Vue 组件。Hugo 也能做纯文稿，但这类交互会变成手写 shortcode + 裸 JS；从零上 Next.js 则要把本该写进正文的精力花在路由和组件库上。
