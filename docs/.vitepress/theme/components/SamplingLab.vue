<script setup lang="ts">
import { computed, ref } from "vue";

const TOKENS = ["the", "a", "to", "of", "and", "in", "is", "that"];
const LOGITS = [3.4, 2.1, 1.6, 0.9, 0.4, -0.2, -0.8, -1.5];

const temp = ref(1);
const k = ref(8);
const p = ref(1);

function softmax(logits: number[], t: number) {
  const tau = Math.max(t, 1e-4);
  const scaled = logits.map((x) => x / tau);
  const m = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - m));
  const z = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / z);
}

function topk(probs: number[], kk: number) {
  const idx = probs
    .map((prob, i) => [prob, i] as const)
    .sort((a, b) => b[0] - a[0])
    .slice(0, kk)
    .map((x) => x[1]);
  const keep = new Set(idx);
  const masked = probs.map((prob, i) => (keep.has(i) ? prob : 0));
  const z = masked.reduce((a, b) => a + b, 0);
  return masked.map((x) => x / z);
}

function topp(probs: number[], pp: number) {
  const ranked = probs
    .map((prob, i) => [prob, i] as const)
    .sort((a, b) => b[0] - a[0]);
  let acc = 0;
  const keep = new Set<number>();
  for (const [prob, i] of ranked) {
    keep.add(i);
    acc += prob;
    if (acc >= pp) break;
  }
  const masked = probs.map((prob, i) => (keep.has(i) ? prob : 0));
  const z = masked.reduce((a, b) => a + b, 0);
  return masked.map((x) => x / z);
}

const probs = computed(() => topp(topk(softmax(LOGITS, temp.value), k.value), p.value));
const entropy = computed(() =>
  -probs.value.reduce((s, x) => (x > 0 ? s + x * Math.log2(x) : s), 0)
);
</script>

<template>
  <div class="lab-card">
    <label class="field">
      <span class="row">temperature <b>{{ temp.toFixed(2) }}</b></span>
      <input v-model.number="temp" type="range" min="0.1" max="2" step="0.05" />
    </label>
    <label class="field">
      <span class="row">top-k <b>{{ k }}</b></span>
      <input v-model.number="k" type="range" min="1" max="8" />
    </label>
    <label class="field">
      <span class="row">top-p <b>{{ p.toFixed(2) }}</b></span>
      <input v-model.number="p" type="range" min="0.1" max="1" step="0.05" />
    </label>

    <div style="margin-top: 0.8rem; display: grid; gap: 0.4rem">
      <div v-for="(tok, i) in TOKENS" :key="tok" style="display: flex; align-items: center; gap: 0.7rem">
        <code style="width: 3rem; font-size: 12px; opacity: 0.75">{{ tok }}</code>
        <div class="bar" style="flex: 1; height: 10px">
          <span :style="{ width: probs[i] * 100 + '%' }" />
        </div>
        <span style="width: 3.4rem; text-align: right; font-family: var(--vp-font-family-mono); font-size: 11px">
          {{ (probs[i] * 100).toFixed(1) }}%
        </span>
      </div>
    </div>
    <p style="margin: 0.8rem 0 0; font-size: 12px; color: var(--vp-c-text-2); line-height: 1.6">
      分布熵 ≈ {{ entropy.toFixed(2) }} bits。温度升高会摊平概率；top-k / top-p 砍掉长尾。
      这组 logits 是示意数据，不是某个真实模型的输出。
    </p>
  </div>
</template>
