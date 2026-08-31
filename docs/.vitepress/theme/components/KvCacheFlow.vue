<template>
  <figure class="flow-figure kv-flow">
    <figcaption>
      <span>KV Cache 在哪出现</span>
      Prefill 一次，Decode 逐 token 复用
    </figcaption>
    <svg viewBox="0 0 900 300" role="img" aria-label="KV Cache 的 Prefill 和 Decode 数据流图">
      <defs>
        <marker id="kv-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M 0 0 L 8 4.5 L 0 9 z" fill="currentColor" />
        </marker>
      </defs>
      <text x="30" y="38" class="flow-kicker">① Prefill：完整 prompt 一次进入模型</text>
      <rect class="flow-node input" x="30" y="58" width="178" height="68" rx="13" />
      <text x="119" y="86" text-anchor="middle" class="flow-title">“今天天气很”</text>
      <text x="119" y="108" text-anchor="middle" class="flow-sub">4 个 token 同时计算</text>
      <path class="flow-line" d="M208 92 H282" />
      <rect class="flow-node model" x="282" y="58" width="174" height="68" rx="13" />
      <text x="369" y="86" text-anchor="middle" class="flow-title">Transformer</text>
      <text x="369" y="108" text-anchor="middle" class="flow-sub">算出每个位置的 K、V</text>
      <path class="flow-line" d="M456 92 H530" />
      <rect class="flow-node cache" x="530" y="47" width="190" height="90" rx="13" />
      <text x="625" y="79" text-anchor="middle" class="flow-title">KV Cache</text>
      <text x="625" y="104" text-anchor="middle" class="flow-sub">K₁…K₄ · V₁…V₄</text>
      <path class="flow-line" d="M456 92 H785" style="opacity:0" />
      <rect class="flow-node output" x="758" y="58" width="112" height="68" rx="13" />
      <text x="814" y="86" text-anchor="middle" class="flow-title">“好”</text>
      <text x="814" y="108" text-anchor="middle" class="flow-sub">下一个 token</text>

      <text x="30" y="180" class="flow-kicker">② Decode：只算新 token，直接读旧 Cache</text>
      <rect class="flow-node sample" x="30" y="200" width="178" height="68" rx="13" />
      <text x="119" y="228" text-anchor="middle" class="flow-title">新 token：“好”</text>
      <text x="119" y="250" text-anchor="middle" class="flow-sub">只处理 1 个 token</text>
      <path class="flow-line" d="M208 234 H282" />
      <rect class="flow-node model" x="282" y="200" width="174" height="68" rx="13" />
      <text x="369" y="228" text-anchor="middle" class="flow-title">Transformer</text>
      <text x="369" y="250" text-anchor="middle" class="flow-sub">只新增 K₅、V₅</text>
      <path class="flow-line" d="M456 234 H530" />
      <rect class="flow-node cache" x="530" y="189" width="190" height="90" rx="13" />
      <text x="625" y="221" text-anchor="middle" class="flow-title">复用 + 追加</text>
      <text x="625" y="246" text-anchor="middle" class="flow-sub">K₁…K₄ + K₅</text>
      <path class="flow-line" d="M720 234 H758" />
      <rect class="flow-node output" x="758" y="200" width="112" height="68" rx="13" />
      <text x="814" y="228" text-anchor="middle" class="flow-title">下一词</text>
      <text x="814" y="250" text-anchor="middle" class="flow-sub">再重复</text>

      <path class="flow-return" d="M625 137 V178" />
      <text x="645" y="165" class="flow-note">读旧 K、V</text>
    </svg>
  </figure>
</template>
