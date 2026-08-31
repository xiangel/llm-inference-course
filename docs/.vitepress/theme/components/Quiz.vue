<script setup lang="ts">
import { computed, ref } from "vue";

type Q = {
  prompt: string;
  options: string[];
  answer: number;
  explanation: string;
};

const props = defineProps<{ questions: Q[] }>();
const picked = ref<Record<number, number>>({});
const revealed = ref(false);

const score = computed(
  () => props.questions.filter((q, i) => picked.value[i] === q.answer).length
);
</script>

<template>
  <div>
    <div v-for="(q, qi) in questions" :key="qi" class="lab-card">
      <p style="margin: 0 0 0.6rem">
        <code style="margin-right: 0.4rem; color: var(--vp-c-brand-1)">{{ String(qi + 1).padStart(2, "0") }}</code>
        {{ q.prompt }}
      </p>
      <button
        v-for="(opt, oi) in q.options"
        :key="oi"
        type="button"
        class="quiz-option"
        :class="{
          selected: picked[qi] === oi && !revealed,
          correct: revealed && oi === q.answer,
          wrong: revealed && picked[qi] === oi && oi !== q.answer,
        }"
        @click="picked[qi] = oi"
      >
        <span style="font-family: var(--vp-font-family-mono); font-size: 11px; opacity: 0.7">
          {{ String.fromCharCode(65 + oi) }}
        </span>
        <span>{{ opt }}</span>
      </button>
      <p v-if="revealed" style="margin: 0.7rem 0 0; font-size: 14px; color: var(--vp-c-text-2); line-height: 1.7">
        {{ q.explanation }}
      </p>
    </div>
    <button
      type="button"
      class="chip active"
      style="padding: 0.45rem 0.9rem; font-size: 14px"
      :disabled="revealed"
      @click="revealed = true"
    >
      {{ revealed ? `已揭晓 · ${score} / ${questions.length}` : "核对答案" }}
    </button>
  </div>
</template>
