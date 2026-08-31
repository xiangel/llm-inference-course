<template>
  <div class="two-col" style="margin: 1.2rem 0">
    <div class="lab-card" style="border-color: rgba(228,181,106,0.28); background: rgba(228,181,106,0.08); margin: 0">
      <p style="margin:0;font-size:11px;letter-spacing:0.16em;text-transform:uppercase;color:var(--vp-c-brand-1)">Prefill</p>
      <p style="margin:0.35rem 0 0.6rem;font-size:1.15rem;font-weight:650">一次吃完整段 prompt</p>
      <ul style="margin:0;padding-left:1.1rem;line-height:1.75">
        <li>输入形状：<code>(B, S)</code></li>
        <li>Attention 是 S×S 的矩阵</li>
        <li>产出整段 KV，以及最后一个位置的 logits</li>
        <li>算力密度高，通常 compute-bound</li>
      </ul>
    </div>
    <div class="lab-card" style="border-color: rgba(125,211,199,0.28); background: rgba(125,211,199,0.08); margin: 0">
      <p style="margin:0;font-size:11px;letter-spacing:0.16em;text-transform:uppercase;color:#7dd3c7">Decode</p>
      <p style="margin:0.35rem 0 0.6rem;font-size:1.15rem;font-weight:650">一次只追加 1 个 token</p>
      <ul style="margin:0;padding-left:1.1rem;line-height:1.75">
        <li>输入形状：<code>(B, 1)</code></li>
        <li>Attention 是 1×(S+t) 的查询</li>
        <li>每步读完全部权重 + 已有 KV</li>
        <li>算术强度极低，通常 memory-bound</li>
      </ul>
    </div>
  </div>
</template>
