<script setup lang="ts">
import { computed, ref } from "vue";

const GPUS = [
  { id: "h100", name: "H100 SXM", tflops: 989, tbw: 3.35 },
  { id: "a100", name: "A100 80GB", tflops: 312, tbw: 2.039 },
  { id: "4090", name: "RTX 4090", tflops: 82.6, tbw: 1.008 },
  { id: "l4", name: "L4", tflops: 121, tbw: 0.3 },
] as const;

const gpuId = ref<(typeof GPUS)[number]["id"]>("h100");
const paramsB = ref(8);
const seq = ref(2048);
const batch = ref(1);
const bytesPerParam = ref(2);

const gpu = computed(() => GPUS.find((g) => g.id === gpuId.value)!);
const ridge = computed(() => (gpu.value.tflops * 1e12) / (gpu.value.tbw * 1e12));

const stats = computed(() => {
  const N = paramsB.value * 1e9;
  const decodeFlops = 2 * N;
  const decodeBytes = N * bytesPerParam.value;
  const decodeAI = decodeFlops / decodeBytes;
  const prefillFlops = 2 * N * seq.value * batch.value;
  const prefillAI = prefillFlops / decodeBytes;
  const decodeTokS = (gpu.value.tbw * 1e12) / decodeBytes;
  return {
    decodeAI,
    prefillAI,
    decodeTokS,
    decodeBound: decodeAI >= ridge.value ? "compute" : "memory",
    prefillBound: prefillAI >= ridge.value ? "compute" : "memory",
  };
});

function logScale(v: number, min: number, max: number) {
  return (Math.log10(Math.max(v, min)) - Math.log10(min)) / (Math.log10(max) - Math.log10(min));
}

const plot = computed(() => {
  const w = 640;
  const h = 280;
  const pad = { l: 48, r: 16, t: 20, b: 40 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const xMin = 0.2;
  const xMax = 4000;
  const peak = gpu.value.tflops;
  const bw = gpu.value.tbw;
  const yMax = peak * 1.15;
  const xOf = (ai: number) => pad.l + logScale(ai, xMin, xMax) * innerW;
  const yOf = (tflops: number) => pad.t + (1 - tflops / yMax) * innerH;
  const ridgeX = xOf(ridge.value);
  const peakY = yOf(peak);
  return {
    w,
    h,
    pad,
    innerW,
    innerH,
    path: `M ${pad.l} ${yOf(Math.min(yMax, bw * xMin))} L ${ridgeX} ${peakY} L ${pad.l + innerW} ${peakY}`,
    decode: {
      x: xOf(stats.value.decodeAI),
      y: yOf(Math.min(peak, stats.value.decodeAI * bw)),
    },
    prefill: {
      x: xOf(stats.value.prefillAI),
      y: yOf(Math.min(peak, stats.value.prefillAI * bw)),
    },
  };
});
</script>

<template>
  <div class="lab-card">
    <div class="chips">
      <button
        v-for="g in GPUS"
        :key="g.id"
        class="chip"
        :class="{ active: gpuId === g.id }"
        type="button"
        @click="gpuId = g.id"
      >
        {{ g.name }}
      </button>
    </div>
    <label class="field">
      <span class="row">参数量（B） <b>{{ paramsB }}</b></span>
      <input v-model.number="paramsB" type="range" min="1" max="70" />
    </label>
    <label class="field">
      <span class="row">Prefill 序列长度 <b>{{ seq }}</b></span>
      <input v-model.number="seq" type="range" min="128" max="32768" step="128" />
    </label>
    <label class="field">
      <span class="row">batch <b>{{ batch }}</b></span>
      <input v-model.number="batch" type="range" min="1" max="64" />
    </label>
    <label class="field">
      <span class="row">权重精度 <b>{{ bytesPerParam }} B/param</b></span>
      <select v-model.number="bytesPerParam" style="width:100%;height:2rem;border-radius:8px;border:1px solid var(--vp-c-border);background:var(--vp-c-bg);color:inherit;padding:0 0.5rem">
        <option :value="2">FP16 / BF16</option>
        <option :value="1">FP8 / INT8</option>
        <option :value="0.5">INT4 权重</option>
      </select>
    </label>

    <svg :viewBox="`0 0 ${plot.w} ${plot.h}`" style="width:100%;margin-top:0.6rem">
      <line
        :x1="plot.pad.l"
        :y1="plot.pad.t + plot.innerH"
        :x2="plot.pad.l + plot.innerW"
        :y2="plot.pad.t + plot.innerH"
        stroke="currentColor"
        opacity="0.2"
      />
      <line
        :x1="plot.pad.l"
        :y1="plot.pad.t"
        :x2="plot.pad.l"
        :y2="plot.pad.t + plot.innerH"
        stroke="currentColor"
        opacity="0.2"
      />
      <path :d="plot.path" fill="none" stroke="#e4b56a" stroke-width="2" />
      <circle :cx="plot.decode.x" :cy="plot.decode.y" r="6" fill="#7dd3c7" />
      <text :x="plot.decode.x + 10" :y="plot.decode.y - 8" font-size="11" fill="currentColor">Decode</text>
      <circle :cx="plot.prefill.x" :cy="plot.prefill.y" r="6" fill="#e4b56a" />
      <text :x="plot.prefill.x + 10" :y="plot.prefill.y - 8" font-size="11" fill="currentColor">Prefill</text>
      <text :x="plot.pad.l" :y="plot.h - 10" font-size="11" fill="currentColor" opacity="0.6">
        算术强度（FLOP / byte）→ 对数轴
      </text>
    </svg>

    <dl class="stat-grid" style="margin-top: 0.8rem; grid-template-columns: 1fr 1fr">
      <div class="stat">
        <dt>Ridge point</dt>
        <dd style="font-size:1rem">{{ ridge.toFixed(0) }} FLOP/B</dd>
      </div>
      <div class="stat">
        <dt>Decode AI</dt>
        <dd style="font-size:1rem">{{ stats.decodeAI.toFixed(2) }} · {{ stats.decodeBound }}</dd>
      </div>
      <div class="stat">
        <dt>Prefill AI</dt>
        <dd style="font-size:1rem">{{ stats.prefillAI.toFixed(0) }} · {{ stats.prefillBound }}</dd>
      </div>
      <div class="stat">
        <dt>Decode 理论上限</dt>
        <dd style="font-size:1rem">{{ stats.decodeTokS.toFixed(0) }} tok/s</dd>
      </div>
    </dl>
    <p style="margin: 0.8rem 0 0; font-size: 12px; color: var(--vp-c-text-2); line-height: 1.65">
      这里省略 Attention 额外 IO，只保留「读一遍权重」。batch=1 的 Decode 算术强度约等于
      2 / 每参数字节数，远远落在带宽屋顶下方。
    </p>
  </div>
</template>
