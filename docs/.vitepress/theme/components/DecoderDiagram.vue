<template>
  <div class="lab-card">
    <p style="margin:0 0 0.7rem;font-size:11px;letter-spacing:0.16em;text-transform:uppercase;color:var(--vp-c-text-2)">
      Decoder 一层 · 从底向上
    </p>
    <ol style="list-style:none;padding:0;margin:0;display:grid;gap:0.45rem">
      <li
        v-for="(b, i) in blocks"
        :key="b"
        style="display:flex;align-items:center;gap:0.7rem;border:1px solid var(--vp-c-border);border-radius:10px;padding:0.5rem 0.7rem;background:var(--vp-c-bg)"
      >
        <span style="width:1.5rem;height:1.5rem;display:grid;place-items:center;border-radius:6px;background:var(--vp-c-brand-soft);color:var(--vp-c-brand-1);font-family:var(--vp-font-family-mono);font-size:11px">
          {{ i + 1 }}
        </span>
        <span>{{ b }}</span>
        <span style="margin-left:auto;font-size:12px;color:var(--vp-c-text-2)">
          {{ i < blocks.length - 1 ? "↓ hidden" : "→ 下一层" }}
        </span>
      </li>
    </ol>
    <p style="margin:0.8rem 0 0;font-size:13px;color:var(--vp-c-text-2);line-height:1.7">
      推理时每一层都读上一层的 hidden，再写出新的 hidden。KV Cache 发生在第 3–4 步：K、V 被存下来，下一步解码不再重算历史 token 的投影。
    </p>
  </div>
</template>

<script setup lang="ts">
const blocks = [
  "Token + RoPE",
  "RMSNorm",
  "Q / K / V 投影",
  "因果 Attention",
  "残差 + RMSNorm",
  "SwiGLU FFN",
  "残差",
];
</script>
