---
title: 第 1 章 · KV Cache
description: Prefill 与 Decode、KV Cache 显存公式、采样，以及给最小 GPT 接上 cache。
---

# 第 1 章 · 自回归解码与 KV Cache

**学时** 6–8 小时 · **阶段 A** 单请求原理 · CPU 可完成核心实验

如果只能从这门课带走一个公式，带走下面这一行。后面所有系统工作——分页、量化、前缀复用、跨机传输——都是在给它找容器。

$$
\mathrm{KV\ bytes} = 2 \cdot L \cdot H_{kv} \cdot d \cdot S \cdot B \cdot e
$$

其中 $e$ 是每个元素的字节数（FP16 为 2，FP8 为 1）。因子 $2$ 来自 Key 和 Value 各一份。

<Checklist
  slug="01-kv-cache"
  :items="[
    { id: 'derive-kv', label: '自己推导一遍 KV 显存公式' },
    { id: 'use-calculator', label: '用计算器对比 8B / 70B 的 KV 开销' },
    { id: 'implement-cache', label: '实现带 cache 的解码循环' },
    { id: 'quiz-1', label: '完成第 1 章自测' }
  ]"
/>

<a class="colab-link" href="https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/01_kv_cache.ipynb" target="_blank" rel="noreferrer">在 Google Colab 打开并运行本章代码 ↗</a>

## 自回归在算什么

语言模型定义的是

$$
p(x_{1:T}) = \prod_{t=1}^{T} p(x_t \mid x_{<t})
$$

生成时我们从左往右采样。第 $t$ 步的输入是已经得到的前缀，输出是 $x_t$ 在词表上的分布。例如模型已经写出「今天天气」，才有条件猜下一个词「很」。所以 **Decode 天生是串行的**：没生成前一个 token，就不知道后一个 token 的输入。

没有 KV Cache 的朴素实现：每一步都把完整前缀再跑一遍网络。生成 $T$ 个 token，Attention 的代价按 $1^2 + 2^2 + \cdots + T^2 \sim O(T^3)$ 增长。这会慢到没法用。

## Prefill 与 Decode

请求进到服务端，会先经历一次 **Prefill**，再进入逐步 **Decode**。两者的算子相同，工作集完全不同。

<PrefillDecodeDiagram />

Prefill 的输出有两样东西：

1. prompt 最后一个位置的 logits → 第一个生成 token；
2. 每一层、每一个位置的 $K,V$ → 写入 Cache。

之后每一步 Decode 只吃 **一个** 新 token，从 Cache 里读历史 $K,V$，把新的 $k_t, v_t$ 追加进去。TTFT（Time To First Token）几乎就是 Prefill 的时间；后面的 TPOT（Time Per Output Token）是 Decode 步的时间。产品上「首字慢、后面稳」通常不是 bug，是这两段的物理差异。

<KvCacheFlow />

<div class="callout insight">
<span class="label">要点</span>
Prefill 在算一个大 GEMM + 一段 S×S Attention；Decode 在反复把整份权重从 HBM 搬出来，只为生产 1 个 token。优化杠杆因此不同：Prefill 吃算力融合和切块；Decode 吃量化、更大的 batch、以及少搬 KV。
</div>

## KV Cache 的数学与显存

先说人话：Attention 每次都需要“当前 token 对历史每个 token 的 Key 和 Value”。历史 token 的 Key 和 Value 一旦算出就不会变，因此把它们保存在显存里，下次直接拿来用，而不是重新计算。这块保存空间就是 KV Cache。

对任意一个新 token，模型会新算出一小份 K 和 V。Cache 的工作很简单：保留旧 token 的 K、V，再把这一小份新 K、V 追加在末尾。第 5 个 token 到来后，Cache 里就有前 5 个 token 的 K、V。

对程序而言，一层 Cache 可以看作一个四维数组：`(batch, KV 头数, 已缓存 token 数, 每头维度)`。把这样的数组放在所有层里，就得到文首的显存估算式。

### 公式里的每个变量是什么

$$
\mathrm{KV\ bytes} = 2 \cdot L \cdot H_{kv} \cdot d \cdot S \cdot B \cdot e
$$

| 符号 | 名称 | 通俗解释 | 例：Llama 3 8B |
| --- | --- | --- | --- |
| $2$ | Key + Value | 每个位置要同时存 K 和 V 两份数据 | 2 |
| $L$ | 层数 | 模型有多少个 Transformer Block；每层都有自己的 Cache | 32 |
| $H_{kv}$ | KV 头数 | 每层要存几组 Key/Value；GQA 会让它小于普通注意力头数 | 8 |
| $d$ | 每头维度 | 每一组 K 或 V 里有多少个数字 | 128 |
| $S$ | 已缓存长度 | prompt 加上已经生成的 token 数；越长越占显存 | 例如 8192 |
| $B$ | batch size | 同时服务多少条请求；每条都有自己的 Cache | 1 |
| $e$ | 每元素字节数 | FP16/BF16 是 2；FP8/INT8 是 1；INT4 是 0.5 | 2 |

如何读这条公式：

1. 先算一个 token 在一层、一个 KV 头上要放多少数字（`d`）。
2. 因为要存 K 和 V，乘以 `2`；再乘每个数字占的字节数 `e`。
3. 同样的数据要在 `Hkv` 个头、`L` 层、`S` 个 token、`B` 条请求里各存一份，因此逐个相乘。

它只回答“KV Cache 本身占多少显存”。**不包括**模型权重、临时激活、CUDA 缓存和框架的预留空间，所以部署时必须预留余量。

**心算：Llama 3 8B，FP16，batch = 1**

- 代入：32 层、8 个 KV 头、每头 128 个数、FP16（每个数 2 字节）。
- 每增加 1 个 token，所有层合计多占约 **128 KB**。
- 到 8K context，Cache 约 **1 GB**。
- 到 128K context，Cache 约 **16 GB**——已经和一份 8B FP16 权重差不多大。

70B 用 GQA 后 KV 头数仍是 8，但层数增加到 80；每个 token 的 Cache 会到约 **320 KB**。长上下文下 **KV 会超过权重**。这就是为什么第 4 章要量化 KV、第 5 章要分页、第 8 章要把 KV 当成集群里的一等资源。

把 KV 头数从 8 改成 32（纯 MHA 的常见情况），同量级模型每个 token 的 Cache 会到约 **512 KB**。GQA 不是精度技巧，是推理系统能活下来的架构选择。DeepSeek 的 MLA 走得更远，把 KV 压进低秩潜变量——那是第 3 章的故事。

## 交互工具：显存计算器

拖动层数、KV 头、序列长度，看权重 + KV 何时把 24 GB / 80 GB 卡撑满。先对比「Llama 3 8B」和「7B · 纯 MHA」：同样量级的模型，MHA 的 KV 会难看很多。

<KvCalculator />

建议自己做的三次计算：

1. 8B，FP16，batch=1，4K / 32K / 128K 各是多少 KV。
2. 同一设置改 FP8 KV。每个元素从 2 字节减到 1 字节，容量直接翻倍。
3. 70B，batch=8，8K——这已经是「一张 80 GB 卡未必放得下」的区域，后面会看到为什么服务端要用 paging 而不是按 `max_seq_len` 预留整块。

## 采样：从 greedy 到 nucleus

`lm_head` 给出 $z \in \mathbb{R}^{V}$。采样是把 $z$ 变成一个 token id 的过程。它不改变前向，但决定输出的熵，也决定「同一个 prompt 能否命中前缀缓存」（温度 > 0 时对话会分叉）。

常见四件套：

| 方法 | 做什么 | 典型用途 |
| --- | --- | --- |
| Greedy | 总是选分数最高的 token | 确定性任务、评测复现 |
| Temperature | 调整分数差距后再转概率 | 小于 1 更保守；大于 1 更多样 |
| Top-k | 只保留分数最高的 $k$ 个再归一化 | 砍掉长尾胡话 |
| Top-p（nucleus） | 按概率从大到小累加到 $p$ | 随分布自适应截断 |

Min-p、typical sampling 是同一家族的变体。服务框架里它们通常是 logits processor，插在前向和 `multinomial` 之间。

下面这个小实验室用一组固定分数，让你直接看 temperature、top-k、top-p 如何改写分布。把 temperature 拉到 0.2 再拉到 1.8，观察概率条的变化。

<SamplingLab />

实现时注意两件实现细节：

1. **先 mask 再 softmax**，或等价地对被扔掉的位置填 `-inf`。先 softmax 再把概率置零，数值上会漏质量，需要重新归一化。
2. **greedy 不是“把温度调到接近零”。** 有的库会用很小的温度冒充 greedy，但直接用 `argmax` 更干净，也避免数值问题。

```python
import torch
import torch.nn.functional as F

def sample(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = cumsum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)
```

## 代码实战：手写 KV Cache

回到第 0 章的 `CausalSelfAttention`。先不用担心全部细节：Cache 就是沿序列维拼接的 `k` 和 `v`。Prefill 时传入整段 prompt；之后 Decode 每步传入一个新 token（$S=1$）。

下面代码的关键是 `cache`：

- 第一次调用：`cache=None`，正常算出 prompt 的 K、V。
- 后续调用：取出旧的 `pk, pv`，把新 token 的 `k, v` 接在后面。
- 返回 `new_cache`：让下一步继续使用。

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.n_head, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if cache is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if cache is None:
            mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(y), new_cache
```

解码循环：

```python
@torch.no_grad()
def generate(model, idx, n_new, temperature=0.8, top_p=0.9):
    caches = [None] * len(model.blocks)
    # prefill
    logits, caches = model(idx, caches)
    next_id = sample(logits[:, -1], temperature, top_p=top_p)
    out = [next_id]
    # decode
    cur = next_id[:, None]
    for _ in range(n_new - 1):
        logits, caches = model(cur, caches)
        next_id = sample(logits[:, -1], temperature, top_p=top_p)
        out.append(next_id)
        cur = next_id[:, None]
    return torch.stack(out, dim=1)
```

`model.forward` 需要把每层返回的 cache 收成一份 list——这和 HuggingFace 的 `past_key_values` 是同一个对象，只是没有那么多层包装。

<div class="callout lab">
<span class="label">在 Colab 里运行</span>
点击本章顶部的链接。Notebook 会先运行一个非常小的 Attention，再打印每次 Decode 后的 KV 长度：<code>4 → 5 → 6 ...</code>。看懂这个长度增长，比背完整代码重要。
</div>

**对照实验（强烈建议做）：**

1. 写一个 `generate_no_cache`：每一步把 `idx` 全量再喂进去。
2. 在 CPU 上用 $D=256, L=8, S_{\text{prompt}}=64$，分别生成 16 / 64 / 256 个 token，记录墙钟。
3. 画两条曲线。无 cache 的时间应该接近二次甚至三次爬升；有 cache 的 Decode 段接近线性。

你还会注意到：`torch.cat` 每步复制整段 K、V，这是玩具实现。生产系统不会这么干——vLLM 用分页块，避免拷贝与碎片。第 5 章再换掉这个 `cat`。现在先让算术正确。

<div class="callout lab">
<span class="label">实验完成标准</span>
有 cache 与无 cache 在相同采样种子下，逐 token 输出一致（greedy 最容易核对）。然后再看时间曲线。正确性先于速度。
</div>

## 参考资料

- **kipply，[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)**。把文首公式和「每 token 128 KB」这一类心算练到不假思索。
- Lilian Weng，[Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- HuggingFace `generate` 文档中 `past_key_values` / `Cache` 类的说明，对照你刚写的 list of tuples
- Llama 3 模型卡：核对 $L, H, H_{kv}, d$

## 自测

<Quiz :questions="[
  {
    prompt: '为什么 Decode 不能轻易用「把序列维并行掉」的办法加速？',
    options: [
      '因为 GPU 没有足够的 Tensor Core',
      '因为 token t 的输入依赖于 t-1 的采样结果，这是模型定义而不是实现缺陷',
      '因为 KV Cache 禁止并行',
      '因为词表太大，softmax 必须串行'
    ],
    answer: 1,
    explanation: '自回归分解本身是串行的。投机解码是在保持同一分布的前提下并行猜测，不是取消条件依赖。'
  },
  {
    prompt: 'Llama 3 8B、FP16、batch=1。生成到 8K context 时 KV 大约是？',
    options: [
      '128 MB',
      '1 GB',
      '8 GB',
      '16 GB'
    ],
    answer: 1,
    explanation: '每 token 128 KB × 8192 ≈ 1 GB。128K 才会到 16 GB。'
  },
  {
    prompt: 'GQA 把 H_kv 做成 H 的四分之一，主要收益是？',
    options: [
      '训练 loss 更低',
      'Prefill 的 FLOP 变成原来的 1/4',
      'KV Cache 显存和 Decode 时的 KV 带宽变成大约 1/4',
      '词表可以更大'
    ],
    answer: 2,
    explanation: 'Q 头数不变，Attention FLOP 几乎不变；省下的是 KV 存储与读取。这正是 Decode 的痛点。'
  },
  {
    prompt: '玩具实现里对 K、V 每步 torch.cat，生产系统最想避免的是？',
    options: [
      'softmax 的数值不稳定',
      '反复整段拷贝造成的内存带宽浪费和碎片',
      'RoPE 被 cat 破坏',
      '词表采样变慢'
    ],
    answer: 1,
    explanation: 'cat 会分配新张量并拷贝历史 KV。PagedAttention 用固定大小的块做预留与拼接，就是在消灭这件事。'
  }
]" />
