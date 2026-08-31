<script setup lang="ts">
import { withBase } from "vitepress";

const layers = [
  {
    kicker: "阶段 A · 已开放",
    title: "单请求原理",
    desc: "搞清楚一次推理请求在算什么、把时间花在哪。KV Cache 是贯穿全课的第一等公民。",
    items: [
      { href: "/chapters/00-prerequisites", code: "00", title: "预备知识：Transformer 与 GPU", open: true },
      { href: "/chapters/01-kv-cache", code: "01", title: "自回归解码与 KV Cache", open: true },
      { href: "/chapters/02-perf-analysis", code: "02", title: "性能分析：瓶颈在哪里", open: true },
    ],
  },
  {
    kicker: "阶段 B · 即将上线",
    title: "单卡优化",
    desc: "从 Attention kernel 与量化切入，把单卡算力和显存吃满。",
    items: [
      { href: "/chapters/03-flashattention", code: "03", title: "Attention 算子：FlashAttention / GQA / MLA", open: false },
      { href: "/chapters/04-quantization", code: "04", title: "量化：GPTQ / AWQ / FP8 / KV 量化", open: false },
    ],
  },
  {
    kicker: "阶段 C · 即将上线",
    title: "服务化系统",
    desc: "连续批处理、分页 KV、前缀复用与投机解码。",
    items: [
      { href: "/chapters/05-serving", code: "05", title: "Continuous Batching 与 PagedAttention", open: false },
      { href: "/chapters/06-speculative", code: "06", title: "投机解码与结构化输出", open: false },
    ],
  },
  {
    kicker: "阶段 D · 即将上线",
    title: "集群与前沿",
    desc: "多卡并行、Prefill/Decode 分离、分布式 KV 池，以及 reasoning 模型的新负载。",
    items: [
      { href: "/chapters/07-distributed", code: "07", title: "分布式推理：TP / PP / EP", open: false },
      { href: "/chapters/08-disaggregation", code: "08", title: "分离式服务与 KV 基础设施", open: false },
      { href: "/chapters/09-reasoning", code: "09", title: "推理模型与 Test-time Compute", open: false },
      { href: "/chapters/10-capstone", code: "10", title: "毕业项目", open: false },
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
