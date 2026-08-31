<script setup lang="ts">
import { computed, ref, watch } from "vue";

const PRESETS = [
  { id: "llama3-8b", name: "Llama 3 8B", layers: 32, kvHeads: 8, headDim: 128, paramsB: 8 },
  { id: "llama3-70b", name: "Llama 3 70B", layers: 80, kvHeads: 8, headDim: 128, paramsB: 70 },
  { id: "qwen25-7b", name: "Qwen2.5 7B", layers: 28, kvHeads: 4, headDim: 128, paramsB: 7.6 },
  { id: "mha-7b", name: "7B · 纯 MHA", layers: 32, kvHeads: 32, headDim: 128, paramsB: 7 },
] as const;

const DTYPES = [
  { id: "fp16", label: "FP16 / BF16", bytes: 2 },
  { id: "fp8", label: "FP8", bytes: 1 },
  { id: "int4", label: "INT4 KV", bytes: 0.5 },
] as const;

const GPUS = [
  { name: "RTX 4090 24GB", bytes: 24e9 },
  { name: "A100 80GB", bytes: 80e9 },
  { name: "H100 80GB", bytes: 80e9 },
];

const presetId = ref<(typeof PRESETS)[number]["id"]>("llama3-8b");
const layers = ref(32);
const kvHeads = ref(8);
const headDim = ref(128);
const seq = ref(8192);
const batch = ref(1);
const dtype = ref<(typeof DTYPES)[number]["id"]>("fp16");
const paramsB = ref(8);

watch(presetId, (id) => {
  const p = PRESETS.find((x) => x.id === id)!;
  layers.value = p.layers;
  kvHeads.value = p.kvHeads;
  headDim.value = p.headDim;
  paramsB.value = p.paramsB;
});

const bytesPerElem = computed(() => DTYPES.find((d) => d.id === dtype.value)!.bytes);
const perToken = computed(() => 2 * layers.value * kvHeads.value * headDim.value * bytesPerElem.value);
const kvBytes = computed(() => perToken.value * seq.value * batch.value);
const weightBytes = computed(() => paramsB.value * 1e9 * 2);
const total = computed(() => kvBytes.value + weightBytes.value);

function fmt(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${Math.round(n)} B`;
}
</script>

<template>
  <div class="lab-card">
    <div class="chips">
      <button
        v-for="p in PRESETS"
        :key="p.id"
        class="chip"
        :class="{ active: presetId === p.id }"
        type="button"
        @click="presetId = p.id"
      >
        {{ p.name }}
      </button>
    </div>

    <label class="field">
      <span class="row">层数 n_layers <b>{{ layers }}</b></span>
      <input v-model.number="layers" type="range" min="8" max="128" />
    </label>
    <label class="field">
      <span class="row">KV 头数 n_kv_heads <b>{{ kvHeads }}</b></span>
      <input v-model.number="kvHeads" type="range" min="1" max="64" />
    </label>
    <label class="field">
      <span class="row">head_dim <b>{{ headDim }}</b></span>
      <input v-model.number="headDim" type="range" min="32" max="256" step="16" />
    </label>
    <label class="field">
      <span class="row">序列长度 seq <b>{{ seq }}</b></span>
      <input v-model.number="seq" type="range" min="128" max="131072" step="128" />
    </label>
    <label class="field">
      <span class="row">batch <b>{{ batch }}</b></span>
      <input v-model.number="batch" type="range" min="1" max="64" />
    </label>
    <label class="field">
      <span class="row">权重大约（十亿参数，按 FP16） <b>{{ paramsB }}</b></span>
      <input v-model.number="paramsB" type="range" min="1" max="70" step="0.1" />
    </label>

    <div class="chips">
      <button
        v-for="d in DTYPES"
        :key="d.id"
        class="chip"
        :class="{ active: dtype === d.id }"
        type="button"
        @click="dtype = d.id"
      >
        {{ d.label }}
      </button>
    </div>

    <dl class="stat-grid">
      <div class="stat">
        <dt>每个 token 的 KV</dt>
        <dd>{{ fmt(perToken) }}</dd>
      </div>
      <div class="stat">
        <dt>当前 KV Cache</dt>
        <dd>{{ fmt(kvBytes) }}</dd>
      </div>
      <div class="stat">
        <dt>权重 FP16 + KV</dt>
        <dd>{{ fmt(total) }}</dd>
      </div>
    </dl>

    <div style="margin-top: 1rem; display: grid; gap: 0.55rem">
      <div v-for="g in GPUS" :key="g.name">
        <div class="row" style="margin-bottom: 0.25rem">
          <span>{{ g.name }}</span>
          <b>{{ Math.min(100, (total / g.bytes) * 100).toFixed(0) }}%</b>
        </div>
        <div class="bar">
          <span :style="{ width: Math.min(100, (total / g.bytes) * 100) + '%' }" />
        </div>
      </div>
    </div>
  </div>
</template>
