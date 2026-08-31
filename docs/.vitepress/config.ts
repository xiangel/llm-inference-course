import { defineConfig } from "vitepress";

const base =
  process.env.VITEPRESS_BASE ||
  (process.env.GITHUB_ACTIONS ? "/llm-inference-course/" : "/");

export default defineConfig({
  lang: "zh-CN",
  title: "大模型推理",
  description:
    "从第一性原理到生产系统：Transformer 推理、KV Cache、性能分析、服务化与分布式。",
  base,
  appearance: "dark",
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,
  head: [
    ["link", { rel: "icon", href: `${base === "/" ? "" : base}/favicon.svg` }],
    ["meta", { name: "theme-color", content: "#c4923a" }],
    [
      "script",
      {
        type: "module",
        src: "https://static.cloudflareinsights.com/beacon.min.js",
        "data-cf-beacon": '{"token":"5eed777106234f33b399a5324579af6c","spa":true}',
      },
    ],
  ],
  markdown: {
    math: true,
    theme: {
      light: "github-light",
      dark: "dark-plus",
    },
  },
  themeConfig: {
    logo: "/favicon.svg",
    siteTitle: "大模型推理",
    outline: { level: [2, 3], label: "本页目录" },
    search: {
      provider: "local",
      options: {
        translations: {
          button: { buttonText: "搜索", buttonAriaLabel: "搜索" },
          modal: {
            noResultsText: "没有结果",
            resetButtonTitle: "清空",
            footer: { selectText: "选择", navigateText: "切换", closeText: "关闭" },
          },
        },
      },
    },
    nav: [
      { text: "路线图", link: "/" },
      { text: "开始学习", link: "/chapters/00-prerequisites" },
      { text: "参考资料", link: "/resources" },
    ],
    sidebar: [
      {
        text: "阶段 A · 先建立直觉",
        items: [
          { text: "第 0 章 一次 LLM 请求", link: "/chapters/00-prerequisites" },
          { text: "第 1 章 Token 与上下文", link: "/chapters/01-kv-cache" },
          { text: "第 2 章 Transformer 全貌", link: "/chapters/02-perf-analysis" },
          { text: "第 3 章 Attention", link: "/chapters/03-flashattention" },
        ],
      },
      {
        text: "阶段 B · 推理基础",
        items: [
          { text: "第 4 章 生成与采样", link: "/chapters/04-quantization" },
          { text: "第 5 章 KV Cache", link: "/chapters/05-serving" },
          { text: "第 6 章 性能指标", link: "/chapters/06-speculative" },
          { text: "第 7 章 GPU 瓶颈", link: "/chapters/07-distributed" },
        ],
      },
      {
        text: "阶段 C · 推理服务与优化",
        items: [
          { text: "第 8 章 部署 vLLM", link: "/chapters/08-disaggregation" },
          { text: "第 9 章 连续批处理", link: "/chapters/09-reasoning" },
          { text: "第 10 章 PagedAttention", link: "/chapters/10-capstone" },
          { text: "第 11 章 量化与投机解码", link: "/chapters/11-quantization-speculation" },
        ],
      },
      {
        text: "阶段 D · 源码与毕业项目",
        items: [
          { text: "第 12 章 nano-vLLM 源码", link: "/chapters/12-nano-vllm" },
          { text: "第 13 章 vLLM V1 对照", link: "/chapters/13-vllm-v1" },
          { text: "第 14 章 毕业项目", link: "/chapters/14-capstone" },
        ],
      },
    ],
    socialLinks: [
      { icon: "github", link: "https://github.com/xiangel/llm-inference-course" },
    ],
    footer: {
      message: "完整 15 章课程 · 使用 VitePress 构建，可发布到 GitHub Pages",
      copyright: "课程内容以第一性原理为主线：KV Cache 是贯穿全课的第一等公民",
    },
    docFooter: {
      prev: "上一章",
      next: "下一章",
    },
    returnToTopLabel: "回到顶部",
    sidebarMenuLabel: "目录",
    darkModeSwitchLabel: "外观",
  },
});
