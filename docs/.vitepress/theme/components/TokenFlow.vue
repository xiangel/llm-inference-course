<template>
  <figure class="flow-figure">
    <figcaption>
      <span>文字怎样变成 Token</span>
      模型不认识字符；它只处理一串整数 ID
    </figcaption>
    <svg class="token-flow" viewBox="0 0 900 246" role="img" aria-label="文本 Tokenization 流程图">
      <defs>
        <marker id="token-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M 0 0 L 8 4.5 L 0 9 z" fill="currentColor" />
        </marker>
      </defs>
      <path class="flow-line" d="M190 102 H265" />
      <path class="flow-line" d="M465 102 H540" />
      <path class="flow-line" d="M720 102 H745" />
      <rect class="flow-node input" x="25" y="56" width="165" height="92" rx="14" />
      <text x="107" y="91" text-anchor="middle" class="flow-title">输入文字</text>
      <text x="107" y="117" text-anchor="middle" class="flow-sub">“今天学 LLM”</text>
      <rect class="flow-node token" x="265" y="56" width="200" height="92" rx="14" />
      <text x="365" y="91" text-anchor="middle" class="flow-title">Tokenizer 的词典</text>
      <text x="365" y="117" text-anchor="middle" class="flow-sub">把文字切成模型见过的片段</text>
      <rect class="flow-node embed" x="540" y="56" width="180" height="92" rx="14" />
      <text x="630" y="91" text-anchor="middle" class="flow-title">Token IDs</text>
      <text x="630" y="117" text-anchor="middle" class="flow-sub">[123, 456, 789]</text>
      <rect class="flow-node model" x="745" y="56" width="130" height="92" rx="14" />
      <text x="810" y="91" text-anchor="middle" class="flow-title">模型</text>
      <text x="810" y="117" text-anchor="middle" class="flow-sub">只接收数字</text>
      <text x="25" y="190" class="flow-kicker">重要</text>
      <text x="25" y="216" class="flow-explain">Token 不是“字”也不是“词”。同一段文字在不同模型的 Tokenizer 下，</text>
      <text x="25" y="236" class="flow-explain">token 数可能不同；上下文限制、成本和速度都按 token 计算。</text>
    </svg>
  </figure>
</template>
