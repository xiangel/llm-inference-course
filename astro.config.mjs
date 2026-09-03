import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import mermaid from "astro-mermaid";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

const base =
  process.env.STARLIGHT_BASE ||
  (process.env.GITHUB_ACTIONS ? "/llm-inference-course/" : "/");

export default defineConfig({
  site: "https://xiangel.github.io",
  base,
  integrations: [
    mermaid({
      theme: "dark",
      autoTheme: true,
    }),
    starlight({
      title: "大模型推理系统",
      description:
        "从 Transformer、KV Cache 到 vLLM 与万卡推理：一本推理系统构建书",
      defaultLocale: "root",
      locales: {
        root: { label: "简体中文", lang: "zh-CN" },
      },
      logo: {
        src: "./src/assets/favicon.svg",
        alt: "大模型推理系统",
      },
      favicon: "/favicon.svg",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/xiangel/llm-inference-course",
        },
      ],
      customCss: ["./src/styles/custom.css", "katex/dist/katex.min.css"],
      lastUpdated: true,
      pagination: true,
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
      head: [
        {
          tag: "meta",
          attrs: { name: "theme-color", content: "#c4923a" },
        },
        {
          tag: "script",
          attrs: {
            type: "module",
            src: "https://static.cloudflareinsights.com/beacon.min.js",
            "data-cf-beacon":
              '{"token":"5eed777106234f33b399a5324579af6c","spa":true}',
          },
        },
      ],
      sidebar: [
        {
          label: "第 0 篇 · 这本书讲什么",
          items: [
            { label: "0.1 大模型推理系统全景", slug: "00-introduction/01-landscape" },
            { label: "0.2 为什么推理是一个系统问题", slug: "00-introduction/02-systems-problem" },
            { label: "0.3 从一个 Token 到万卡集群", slug: "00-introduction/03-token-to-cluster" },
            { label: "0.4 全书学习路线", slug: "00-introduction/04-roadmap" },
          ],
        },
        {
          label: "第一篇 · 理解 LLM 推理",
          items: [
            { label: "1.1 大模型如何生成一个 Token", slug: "01-transformer/01-generate-one-token" },
            { label: "1.2 Transformer 在推理时做了什么", slug: "01-transformer/02-transformer-inference" },
            { label: "1.3 Attention 到底在计算什么", slug: "01-transformer/03-attention" },
            { label: "1.4 RoPE、Logits 与 Sampling", slug: "01-transformer/04-rope-logits-sampling" },
          ],
        },
        {
          label: "后续篇章（写作中）",
          collapsed: true,
          items: [
            { label: "第二篇 Prefill、Decode 与 KV Cache", slug: "02-prefill-decode-kv-cache" },
            { label: "第三篇 GPU 与 Attention", slug: "03-gpu-attention" },
            { label: "第四篇 Batch、Scheduler 与 KV Cache", slug: "04-batching-scheduler" },
            { label: "第五篇 从零实现 Mini-vLLM", slug: "05-mini-vllm" },
            { label: "第六篇 走进 vLLM", slug: "06-vllm" },
            { label: "第七篇 GPU Kernel 与性能优化", slug: "07-kernel" },
            { label: "第八篇 LLM Quantization", slug: "08-quantization" },
            { label: "第九篇 Multi-GPU 推理", slug: "09-multi-gpu" },
            { label: "第十篇 主流 LLM 推理引擎", slug: "10-engines" },
            { label: "第十一篇 高级推理技术", slug: "11-advanced" },
            { label: "第十二篇 生产级推理平台", slug: "12-serving" },
            { label: "第十三篇 性能工程", slug: "13-performance" },
            { label: "第十四篇 成本与容量规划", slug: "14-capacity" },
            { label: "第十五篇 万卡推理系统", slug: "15-scale" },
            { label: "第十六篇 终极实战", slug: "16-project" },
          ],
        },
      ],
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
});
