<template>
  <figure class="flow-figure">
    <figcaption>
      <span>一次请求发生了什么</span>
      从你写的 Python 到浏览器里逐字出现的回答
    </figcaption>
    <svg class="request-flow" viewBox="0 0 900 270" role="img" aria-label="LLM 请求生命周期图">
      <defs>
        <marker id="request-arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">
          <path d="M 0 0 L 8 4.5 L 0 9 z" fill="currentColor" />
        </marker>
      </defs>
      <path class="flow-line" d="M145 105 H188" />
      <path class="flow-line" d="M328 105 H371" />
      <path class="flow-line" d="M520 105 H563" />
      <path class="flow-line" d="M710 105 H753" />
      <rect class="flow-node input" x="25" y="61" width="120" height="88" rx="14" />
      <text x="85" y="94" text-anchor="middle" class="flow-title">你的程序</text>
      <text x="85" y="119" text-anchor="middle" class="flow-sub">Prompt 字符串</text>
      <rect class="flow-node token" x="188" y="61" width="140" height="88" rx="14" />
      <text x="258" y="94" text-anchor="middle" class="flow-title">推理服务</text>
      <text x="258" y="119" text-anchor="middle" class="flow-sub">接收 API 请求</text>
      <rect class="flow-node embed" x="371" y="61" width="149" height="88" rx="14" />
      <text x="445" y="94" text-anchor="middle" class="flow-title">Tokenizer</text>
      <text x="445" y="119" text-anchor="middle" class="flow-sub">文字变 token IDs</text>
      <rect class="flow-node model" x="563" y="61" width="147" height="88" rx="14" />
      <text x="636" y="94" text-anchor="middle" class="flow-title">模型权重</text>
      <text x="636" y="119" text-anchor="middle" class="flow-sub">预测下一个 token</text>
      <rect class="flow-node output" x="753" y="61" width="122" height="88" rx="14" />
      <text x="814" y="94" text-anchor="middle" class="flow-title">流式响应</text>
      <text x="814" y="119" text-anchor="middle" class="flow-sub">token 变回文字</text>
      <path class="flow-return" d="M814 149 V195 Q814 214 795 214 H618 Q600 214 600 196 V170" />
      <text x="704" y="246" text-anchor="middle" class="flow-note">每生成一个 token，模型都会重复最后两步，直到结束</text>
      <text x="25" y="36" class="flow-kicker">输入路径</text>
    </svg>
  </figure>
</template>
