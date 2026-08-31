---
title: 第 0 章 · 预备知识
description: Decoder-only Transformer 的数据流、张量形状、RoPE，以及 GPU 的算力/带宽画像。
---

# 第 0 章 · 预备知识：Transformer 与 GPU

**学时** 4–6 小时 · **阶段 A** 单请求原理 · 不需要 GPU

这一章先用最朴素的方式回答三个问题：Transformer 是什么、它如何从一句话预测下一个词、以及大模型为什么要用 GPU。读完后，再去看 Llama 或 Qwen 的代码会轻松很多。

<Checklist
  slug="00-prerequisites"
  :items="[
    { id: 'read-decoder', label: '读完 Decoder-only 与张量形状' },
    { id: 'run-mingpt', label: '跑通最小 GPT 前向并核对形状' },
    { id: 'quiz-0', label: '完成第 0 章自测' }
  ]"
/>

<a class="colab-link" href="https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/00_transformer_basics.ipynb" target="_blank" rel="noreferrer">在 Google Colab 打开并运行本章代码 ↗</a>

## 为什么从这里开始

推理系统和训练系统看 Transformer 的方式不一样。

训练关心：梯度能否稳定、数据吞吐、ZeRO 把优化器状态放哪。推理关心的几乎全是另一张清单：

- 每个新 token 要读多少权重？
- KV Cache 随序列如何膨胀？
- Prefill 一次能吃多长的 prompt，而不把 Decode 饿死？

这些数字全部由 **层数、头数、head_dim、精度** 决定。所以第 0 章先把「形状」钉死，第 1 章才有资格谈 Cache，第 2 章才有资格谈瓶颈。

::: tip 学习约定
本课默认 Decoder-only（GPT / Llama / Qwen / DeepSeek 这一路）。Encoder-decoder（T5）和纯 Encoder（BERT）只在需要对比时出现。
:::

## Transformer 整体架构：先看全图

Transformer 可以把它理解成一个“根据前文猜下一个词”的机器。假设输入是「今天天气很」，它会给词表中的每一个词一个分数：`好` 可能最高，`糟糕` 也有一些可能。我们按照这些分数选出一个新词，然后把新词接到句子后面，重复这个过程。

一个 Decoder-only Transformer 的整体流水线只有四步：

1. **分词（Tokenizer）**：把文字变成整数 ID，例如「今天天气很」→ `[123, 456, 789]`。
2. **词向量（Embedding）**：把每个整数 ID 查表，变成一串浮点数；这就是模型真正处理的输入。
3. **多层 Transformer Block**：每层有两部分。Attention 让一个词查看它之前的词；FFN 则独立地加工每个位置的信息。两者之间都有残差连接，避免信息在深层网络里消失。
4. **输出层（LM Head）**：把最后一个位置的向量映射到整个词表，得到下一个 token 的概率分布。

```text
文字 → Token IDs → Embedding → [Attention + FFN] × L 层 → LM Head → 下一个 token 的概率
```

这里的 **Decoder-only** 指“只能看左边，不能偷看右边”。生成「今天天气很 _」时，模型只能参考已经出现的「今天天气很」，不能先看答案。这条限制由 Attention 里的因果遮罩（causal mask）实现。

<div class="callout insight">
<span class="label">一个简单比喻</span>
把每个 token 当成会议室里的一位参会者。Attention 让当前参会者查看前面所有人的发言，并决定谁更重要；FFN 则让它独自整理听到的信息。经过多轮会议后，最后一位参会者投票选出下一个词。
</div>

## Decoder-only 数据流

现在来看 Block 内部。Llama / Qwen 把 LayerNorm 换成 RMSNorm，把 GELU FFN 换成 SwiGLU，把绝对位置换成 RoPE，但整体骨架没有改变。

<DecoderDiagram />

一次完整前向，对长度为 $S$ 的序列：

1. Token id $\rightarrow$ embedding，得到 $X \in \mathbb{R}^{B \times S \times D}$。
2. 每一层：Attention + FFN，形状保持 $B \times S \times D$。
3. 最后 RMSNorm + `lm_head`，得到 $B \times S \times V$ 的 logits。

推理时我们通常只要 **最后一个位置** 的 logits。但 Prefill 阶段仍必须把 $S$ 个位置都算完，因为后面 Decode 要用到整段 KV。

## 张量形状清单

第一次看可以不用背公式。先把 $D$ 理解成“每个 token 用多少个数字描述自己”，$H$ 理解成“Attention 同时从多少种角度看上下文”。配置文件里的 `n_embd` / `hidden_size` / `d_model`，指的都是 $D$。

| 符号 | 含义 | Llama 3 8B |
| --- | --- | --- |
| $B$ | batch | 由服务端调度决定 |
| $S$ | 序列长度 | Prefill 时是 prompt 长 |
| $D$ | hidden size | 4096 |
| $H$ | query 头数 | 32 |
| $H_{kv}$ | KV 头数（GQA） | 8 |
| $d$ | head_dim $= D / H$ | 128 |
| $L$ | 层数 | 32 |
| $V$ | 词表 | 128256 |

一组必须能默写的形状：

$$
Q \in \mathbb{R}^{B \times H \times S \times d},\quad
K,V \in \mathbb{R}^{B \times H_{kv} \times S \times d}
$$

GQA 把 $K,V$ 按组重复到 $H$ 个头上再做 attention。对推理来说，这件事的全部意义是：**KV Cache 按 $H_{kv}$ 计，不按 $H$ 计。** Llama 3 8B 的 KV 因此是纯 MHA 的 $8/32 = 1/4$。

FFN（SwiGLU）通常是两到三个 $D \times D_{ff}$ 的 GEMM，$D_{ff}$ 约为 $8/3 \cdot D$ 再对齐。参数量的大头在这里，不在 Attention 的 $QKV$ 投影——但 **Decode 的时间大头往往不在 FLOP，而在把这些权重从 HBM 搬到寄存器**。第 2 章会把这句话变成不等式。

<div class="callout warn">
<span class="label">易错</span>
不要把 <code>num_attention_heads</code> 和 <code>num_key_value_heads</code> 当成同一个数。读 HuggingFace <code>config.json</code> 时先找这两个字段。很多「我算的 KV 显存对不上」来自这里。
</div>

## RoPE 与推理

RoPE 把位置写进 $Q$ 和 $K$ 的复数旋转里，而不是加在 embedding 上。推理时有两个直接后果：

1. **可以增量计算。** 新 token 只旋转自己的 $q_t, k_t$，历史 $K$ 已经在 Cache 里带着正确的位置。
2. **外推不是免费的。** 训练长度之外的位置，旋转角度会走到模型没见过的区域。YaRN、NTK-aware scaling、位置插值，都是在修这件事。本课到长上下文章节再展开。

实现上你会看到两种写法：`complex64` 视图，或把偶/奇维拆开做 `cos/sin`。对形状没有影响——RoPE 不改变 $Q,K$ 的 $(B,H,S,d)$。

## GPU：算力、带宽、Roofline 预告

先建立三个量，不必一次学完 CUDA：

| 量 | 它在问什么 | 数量级（H100 SXM） |
| --- | --- | --- |
| 峰值算力 | 每秒最多多少 FLOP | ~989 TFLOP/s（FP16 Tensor Core） |
| 显存带宽 | 每秒最多从 HBM 搬多少字节 | ~3.35 TB/s |
| 显存容量 | 权重 + KV + 激活还能不能放下 | 80 GB |

**算术强度** $=$ 完成这些计算需要的 FLOP / 需要搬的字节。强度高，瓶颈在算力；强度低，瓶颈在带宽。

粗算一个 Decode 步（batch = 1，忽略 Attention 的额外 IO）：

- FLOP $\approx 2N$（每个参数一次乘加）
- 字节 $\approx 2N$（FP16 下每个参数 2 字节）
- 强度 $\approx 1$ FLOP/byte

H100 的「屋顶脊点」大约是 $989 / 3.35 \approx 295$ FLOP/byte。Decode 的 1 离 295 差了两个数量级——所以它几乎一定是 **memory-bound**。第 2 章会把这条曲线画出来，并解释为什么 Prefill 可以翻到脊点右侧。

现在只需要记住一句话：**推理优化的默认假说是「在搬数据，不是在算」。** 任何声称加速的技术，都应该能指出它少搬了什么，或把算术强度抬到了哪。

## 代码实战：最小 GPT 前向

下面是一个刻意写短的 Decoder 层。它没有加载任何真实模型，也不会生成有意义的句子；作用是让你看清数据如何流动。直接在浏览器里用 Colab 运行即可，免费 CPU 足够。

**先读代码的地图：**

- `RMSNorm`：把数字的尺度整理得更稳定。
- `CausalSelfAttention`：计算“当前位置应该关注前面哪些位置”。
- `SwiGLU`：对 Attention 汇总后的信息再做一次非线性加工。
- `Block`：把 Attention 与 FFN 用残差连接串起来；真实大模型会堆叠几十到上百个 Block。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)

class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.head_dim = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)                     # (B, S, H, d)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(y)

class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Block(nn.Module):
    def __init__(self, d, n_head, d_ff):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_head)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x
```

上面的代码有三个值得停下来看一眼的地方：

1. `qkv = ...view(B, S, 3, H, d)`：一次线性层同时生成 Q、K、V；`unbind(2)` 再把它们拆开。
2. `att = q @ k.transpose(-2, -1)`：这是“每个词和所有历史词打分”。下面的 `mask` 会挡住未来位置。
3. `x = x + ...`：这是残差连接。新计算出的信息加回原来的 `x`，让深层网络仍保留原始信息。

用一组很小的“假 Llama”尺寸来运行。不要下载 8B 权重。

```python
B, S, D, H = 1, 16, 64, 4
d_ff = 128
x = torch.randn(B, S, D)
block = Block(D, H, d_ff)
y = block(x)
assert y.shape == (B, S, D)
print("ok", tuple(y.shape))
```

<div class="callout lab">
<span class="label">建议怎么跑</span>
点击本章顶部的 Colab 链接；在菜单选择 <code>运行时 → 全部运行</code>。输出 <code>ok (1, 16, 64)</code> 就表示最小 Block 已经跑通。
</div>

**如果想多练一步：**

1. 在 `CausalSelfAttention.forward` 里对 `q, k, v, att` 各打一行 `print(shape)`。`att` 必须是 `(B, H, S, S)`。
2. 把 `n_head` 改成不能整除 $D$ 的数，确认 `assert` 会响——形状约束要变成肌肉记忆。
3. （可选）把 `nn.Linear` 换成手写 `x @ W.T`，体会 Decode 时「读 $W$」和「读 $x$」谁更大。$W$ 是 $D \times D$ 量级，$x$ 只是 $1 \times D$。

对照阅读：[nanoGPT 的 `model.py`](https://github.com/karpathy/nanoGPT/blob/master/model.py)。你会发现它用的是 MHA + GELU，没有 GQA 和 SwiGLU；差在细节，骨架相同。

<div class="callout lab">
<span class="label">实验完成标准</span>
你能不看代码，在白纸上写出 <code>qkv</code> 投影之后 <code>view / unbind / transpose</code> 的形状变化。做不到就不要进第 1 章——KV Cache 只是把这里的 <code>k, v</code> 沿序列维拼起来。
</div>

## 参考资料

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) 与视频 [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Llama 2 / Llama 3 技术报告中的 GQA 与 SwiGLU 段落（知道 $H_{kv}$ 从哪来）

下一章会给上面的 `CausalSelfAttention` 加上 cache，并写出显存公式。

## 自测

<Quiz :questions="[
  {
    prompt: 'Llama 3 8B 的 hidden size 是 4096、query 头数 32、KV 头数 8。head_dim 是多少？KV 相对纯 MHA 省了多少显存？',
    options: [
      'head_dim = 128；KV 是 MHA 的 1/4',
      'head_dim = 512；KV 是 MHA 的 1/4',
      'head_dim = 128；KV 是 MHA 的 1/8',
      'head_dim = 256；GQA 不改变 KV 大小'
    ],
    answer: 0,
    explanation: 'head_dim = D/H = 4096/32 = 128。KV 按 H_kv 计，8/32 = 1/4。'
  },
  {
    prompt: 'Prefill 一段长度为 S 的 prompt，因果 Attention 的得分矩阵形状是？',
    options: [
      '(B, H, 1, S)',
      '(B, H, S, S)',
      '(B, H, S, d)',
      '(B, S, D)'
    ],
    answer: 1,
    explanation: '每个头都是 S 个 query 对 S 个 key。Decode 一步才是 (B, H, 1, S+t)。'
  },
  {
    prompt: 'batch=1 的 Decode 步，粗算算术强度大约是 1 FLOP/byte（FP16）。这意味着什么？',
    options: [
      '已经喂饱 Tensor Core，优化重点是算子融合',
      '主要瓶颈是从 HBM 搬权重，继续堆 FLOP 收益很小',
      '只要换一张算力更高的卡，延迟就会线性下降',
      'KV Cache 会把强度自动抬到 ridge point 以上'
    ],
    answer: 1,
    explanation: '强度远低于 GPU 脊点（H100 约 300 FLOP/byte）。Decode 是搬权重的游戏。KV Cache 减少的是计算，不是权重流量。'
  }
]" />
