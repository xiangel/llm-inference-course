<script setup lang="ts">
import { withBase } from "vitepress";

defineProps<{
  title: string;
  hours?: string;
  outline: string[];
}>();
</script>

<template>
  <div class="lab-card">
    <p style="margin:0;font-size:11px;letter-spacing:0.14em;text-transform:uppercase;color:var(--vp-c-brand-1)">即将上线</p>
    <h2 style="margin:0.4rem 0 0.35rem;border:0;padding:0">{{ title }}</h2>
    <p v-if="hours" style="margin:0 0 0.8rem;color:var(--vp-c-text-2)">预计学时 {{ hours }}</p>
    <p>这一章还在编写。下面是已经排好的提纲，方便你对照整体路线。</p>
    <ol>
      <li v-for="item in outline" :key="item">{{ item }}</li>
    </ol>
    <p style="margin-bottom:0">
      目前开放的是
      <a :href="withBase('/chapters/00-prerequisites')">第 0 章</a>、
      <a :href="withBase('/chapters/01-kv-cache')">第 1 章</a>、
      <a :href="withBase('/chapters/02-perf-analysis')">第 2 章</a>。
    </p>
  </div>
</template>
