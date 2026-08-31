import { defineConfig } from "vitepress";

const base = process.env.VITEPRESS_BASE || "/";

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
        text: "阶段 A · 单请求原理",
        items: [
          { text: "第 0 章 预备知识", link: "/chapters/00-prerequisites" },
          { text: "第 1 章 KV Cache", link: "/chapters/01-kv-cache" },
          { text: "第 2 章 性能分析", link: "/chapters/02-perf-analysis" },
        ],
      },
      {
        text: "阶段 B · 单卡优化",
        items: [
          { text: "第 3 章 Attention 算子 · 即将上线", link: "/chapters/03-flashattention" },
          { text: "第 4 章 量化 · 即将上线", link: "/chapters/04-quantization" },
        ],
      },
      {
        text: "阶段 C · 服务化系统",
        items: [
          { text: "第 5 章 调度与分页 · 即将上线", link: "/chapters/05-serving" },
          { text: "第 6 章 投机解码 · 即将上线", link: "/chapters/06-speculative" },
        ],
      },
      {
        text: "阶段 D · 集群与前沿",
        items: [
          { text: "第 7 章 分布式推理 · 即将上线", link: "/chapters/07-distributed" },
          { text: "第 8 章 分离式服务 · 即将上线", link: "/chapters/08-disaggregation" },
          { text: "第 9 章 Reasoning · 即将上线", link: "/chapters/09-reasoning" },
          { text: "第 10 章 毕业项目 · 即将上线", link: "/chapters/10-capstone" },
        ],
      },
    ],
    socialLinks: [],
    footer: {
      message: "第 0–2 章先行公开 · 使用 VitePress 构建，可发布到 GitHub Pages",
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
