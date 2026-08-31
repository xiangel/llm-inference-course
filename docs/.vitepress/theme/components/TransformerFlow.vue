<template>
  <figure class="flow-figure">
    <figcaption>
      <span>Decoder-only Transformer</span>
      一次请求从文字到下一个 token
    </figcaption>
    <svg viewBox="0 0 900 372" role="img" aria-label="Decoder-only Transformer 数据流图">
      <defs>
        <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M 0 0 L 8 4.5 L 0 9 z" fill="currentColor" />
        </marker>
        <linearGradient id="modelFill" x1="0" x2="1">
          <stop stop-color="#c4923a" stop-opacity=".18" />
          <stop offset="1" stop-color="#7dd3c7" stop-opacity=".12" />
        </linearGradient>
      </defs>

      <path class="flow-line" d="M145 102 H190" />
      <path class="flow-line" d="M330 102 H375" />
      <path class="flow-line" d="M525 102 H570" />
      <path class="flow-line" d="M720 102 H755" />

      <rect class="flow-node input" x="25" y="55" width="120" height="94" rx="14" />
      <text x="85" y="90" text-anchor="middle" class="flow-title">输入文本</text>
      <text x="85" y="116" text-anchor="middle" class="flow-sub">“今天天气很”</text>

      <rect class="flow-node token" x="190" y="55" width="140" height="94" rx="14" />
      <text x="260" y="88" text-anchor="middle" class="flow-title">Tokenizer</text>
      <text x="260" y="116" text-anchor="middle" class="flow-sub">文字 → token IDs</text>

      <rect class="flow-node embed" x="375" y="55" width="150" height="94" rx="14" />
      <text x="450" y="88" text-anchor="middle" class="flow-title">Embedding</text>
      <text x="450" y="116" text-anchor="middle" class="flow-sub">ID → 数字向量</text>

      <rect class="flow-node model" x="570" y="31" width="150" height="142" rx="14" />
      <text x="645" y="75" text-anchor="middle" class="flow-title">Transformer</text>
      <text x="645" y="101" text-anchor="middle" class="flow-sub">Block × L 层</text>
      <text x="645" y="130" text-anchor="middle" class="flow-note">看前文，整理信息</text>

      <rect class="flow-node output" x="755" y="55" width="120" height="94" rx="14" />
      <text x="815" y="90" text-anchor="middle" class="flow-title">LM Head</text>
      <text x="815" y="116" text-anchor="middle" class="flow-sub">词表分数</text>

      <path class="flow-line" d="M815 149 V220" />
      <rect class="flow-node sample" x="700" y="220" width="175" height="92" rx="14" />
      <text x="787" y="255" text-anchor="middle" class="flow-title">采样下一个 token</text>
      <text x="787" y="281" text-anchor="middle" class="flow-sub">“好”  62% · “热” 18%</text>

      <path class="flow-return" d="M700 266 H545 Q500 266 500 220 V198 Q500 185 513 185 H570" />
      <text x="510" y="314" text-anchor="middle" class="flow-note">把新 token 接回输入，继续生成</text>

      <text x="25" y="205" class="flow-kicker">一次前向</text>
      <text x="25" y="230" class="flow-explain">模型对已有文字计算一次，</text>
      <text x="25" y="252" class="flow-explain">只取最后一个位置的词表分数。</text>
      <text x="25" y="289" class="flow-kicker">生成循环</text>
      <text x="25" y="314" class="flow-explain">每选出一个 token，就把它加到</text>
      <text x="25" y="336" class="flow-explain">句尾，再运行下一步。</text>
    </svg>
  </figure>
</template>
