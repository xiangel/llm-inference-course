<script setup lang="ts">
import { withBase } from "vitepress";

const layers = [
  {
    kicker: "阶段 A · 已开放",
    title: "先建立直觉",
    desc: "从一次请求、Token 和 Transformer 全貌开始。默认你会 Python，但还不了解算法与深度学习。",
    items: [
      { href: "/chapters/00-prerequisites", code: "00", title: "一次 LLM 请求发生了什么", open: true },
      { href: "/chapters/01-kv-cache", code: "01", title: "Token：模型眼中的文本", open: true },
      { href: "/chapters/02-perf-analysis", code: "02", title: "Transformer 全貌：模型怎样续写文本", open: true },
    ],
  },
  {
    kicker: "阶段 A · 已开放",
    title: "先建立直觉",
    desc: "从请求、Token、Transformer 与 Attention 入门；默认你会 Python，但不要求算法背景。",
    items: [
      { href: "/chapters/03-flashattention", code: "03", title: "Attention：当前 token 如何看前文", open: true },
    ],
  },
  {
    kicker: "阶段 B · 已开放",
    title: "推理基础",
    desc: "从生成、KV Cache、性能指标到 GPU 瓶颈，逐步理解一次推理为什么会慢。",
    items: [
      { href: "/chapters/04-quantization", code: "04", title: "自回归生成与采样", open: true },
      { href: "/chapters/05-serving", code: "05", title: "KV Cache：避免重复计算", open: true },
      { href: "/chapters/06-speculative", code: "06", title: "推理性能的语言", open: true },
      { href: "/chapters/07-distributed", code: "07", title: "GPU 与推理瓶颈", open: true },
    ],
  },
  {
    kicker: "阶段 C · 已开放",
    title: "推理服务与优化",
    desc: "启动 vLLM，再读懂连续批处理、PagedAttention、量化和投机解码的工程取舍。",
    items: [
      { href: "/chapters/08-disaggregation", code: "08", title: "部署第一个 vLLM 服务", open: true },
      { href: "/chapters/09-reasoning", code: "09", title: "连续批处理：让 GPU 不空转", open: true },
      { href: "/chapters/10-capstone", code: "10", title: "PagedAttention 与前缀缓存", open: true },
      { href: "/chapters/11-quantization-speculation", code: "11", title: "量化与投机解码", open: true },
    ],
  },
  {
    kicker: "阶段 D · 已开放",
    title: "源码与毕业项目",
    desc: "先用 nano-vLLM 建立源码阅读路径，再映射到 vLLM V1 的生产级结构。",
    items: [
      { href: "/chapters/12-nano-vllm", code: "12", title: "读懂 nano-vLLM：从请求到 token", open: true },
      { href: "/chapters/13-vllm-v1", code: "13", title: "从 nano-vLLM 到 vLLM V1", open: true },
      { href: "/chapters/14-capstone", code: "14", title: "毕业项目", open: true },
    ],
  },
];
</script>

<template>
  <div class="roadmap">
    <section v-for="layer in layers" :key="layer.title" class="roadmap-layer">
      <p style="margin:0;font-size:11px;letter-spacing:0.14em;text-transform:uppercase;color:var(--vp-c-brand-1)">
        {{ layer.kicker }}
      </p>
      <h3>{{ layer.title }}</h3>
      <p style="margin:0 0 0.6rem;color:var(--vp-c-text-2);font-size:14px;line-height:1.7">{{ layer.desc }}</p>
      <a
        v-for="c in layer.items"
        :key="c.code"
        class="chapter"
        :class="{ coming: !c.open }"
        :href="withBase(c.href)"
      >
        <span>
          <code style="margin-right:0.4rem;color:var(--vp-c-brand-1)">{{ c.code }}</code>
          {{ c.title }}
        </span>
        <span :class="c.open ? 'badge-open' : 'badge-soon'">{{ c.open ? "开放" : "即将上线" }}</span>
      </a>
    </section>
  </div>
</template>
