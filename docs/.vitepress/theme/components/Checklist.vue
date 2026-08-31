<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";

const props = defineProps<{
  slug: string;
  items: { id: string; label: string }[];
}>();

const KEY = "llm-inference-progress-v1";
const done = ref<string[]>([]);
const ready = ref(false);

function loadAll(): Record<string, string[]> {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "{}");
  } catch {
    return {};
  }
}

onMounted(() => {
  done.value = loadAll()[props.slug] || [];
  ready.value = true;
});

watch(done, (list) => {
  if (!ready.value) return;
  const all = loadAll();
  all[props.slug] = list;
  localStorage.setItem(KEY, JSON.stringify(all));
});

function toggle(id: string) {
  done.value = done.value.includes(id)
    ? done.value.filter((x) => x !== id)
    : [...done.value, id];
}

const pct = computed(() =>
  props.items.length ? Math.round((done.value.length / props.items.length) * 100) : 0
);
</script>

<template>
  <div class="lab-card">
    <div class="row" style="margin-bottom: 0.55rem">
      <strong>学习清单</strong>
      <span style="font-family: var(--vp-font-family-mono); font-size: 12px">{{ done.length }}/{{ items.length }} · {{ pct }}%</span>
    </div>
    <div class="bar" style="margin-bottom: 0.8rem">
      <span :style="{ width: pct + '%' }" />
    </div>
    <label
      v-for="item in items"
      :key="item.id"
      style="display: flex; gap: 0.55rem; align-items: flex-start; margin: 0.45rem 0; cursor: pointer"
    >
      <input type="checkbox" :checked="done.includes(item.id)" @change="toggle(item.id)" />
      <span>{{ item.label }}</span>
    </label>
    <p style="margin: 0.7rem 0 0; font-size: 12px; color: var(--vp-c-text-2)">
      进度保存在浏览器本地，不会上传。
    </p>
  </div>
</template>
