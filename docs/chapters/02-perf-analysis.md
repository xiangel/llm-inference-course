---
title: 第 2 章 · 性能分析
description: TTFT / TPOT、Roofline、算术强度，以及为什么 Prefill 吃算力、Decode 吃带宽。
---

# 第 2 章 · 性能分析：瓶颈在哪里

**学时** 4–6 小时 · **阶段 A** 单请求原理 · 有 GPU 更佳，没有也能做完手算

第 0 章给了形状，第 1 章给了 KV。现在要回答：**时间到底花在哪？** 选错瓶颈，后面所有优化都是在给错误的屋顶添瓦。本章的判断工具是 Roofline；结论你会在第 1 章末尾已经猜到——Decode 在搬权重。这里把它变成可计算的命题。

<Checklist
  slug="02-perf-analysis"
  :items="[
    { id: 'metrics', label: '能口头解释 TTFT 与 TPOT 的差别' },
    { id: 'roofline-lab', label: '在 Roofline 实验室里对比 Prefill / Decode' },
    { id: 'handcalc', label: '完成一次理论解码上限手算' },
    { id: 'quiz-2', label: '完成第 2 章自测' }
  ]"
/>

## 该看哪些指标

服务一个生成请求，墙上时钟可以切成两段，对应第 1 章的两阶段：

| 指标 | 全称 | 它在衡量 | 用户能感觉到 |
| --- | --- | --- | --- |
| TTFT | Time To First Token | Prefill + 调度排队 | 点下发送之后要等多久才开始出字 |
| TPOT / ITL | Time Per Output Token / Inter-Token Latency | 相邻两个输出 token 的间隔 | 字「蹦」得顺不顺 |
| 吞吐量 | tokens / s（系统级） | 所有并发请求加总的产出 | 机器成本 |
| Goodput | 满足 SLO 的吞吐 | 把超时请求剔除后再计 | 真正能卖的容量 |
| P50 / P99 延迟 | 分位数 | 尾部请求有多糟 | 高峰期的口碑 |

不要只用「tokens per second」一个数去比较两篇博客。同一个 80 tok/s：

- 可以是 **单请求** Decode 很快（低 TPOT，产品爽）；
- 也可以是 **64 路并发** 摊出来的系统吞吐（单用户其实在排队）。

batch 把后者抬上去，几乎一定伤害前者。第 5 章的 continuous batching 是在这两极之间做调度，而不是「越大越快」。

<div class="callout insight">
<span class="label">要点</span>
讨论优化时先声明优化对象。量化权重通常降 TPOT、提并发容量；切块 Prefill 通常降 TTFT 的尾部；前缀缓存降的是重复 prompt 的 TTFT。没有对象的加速数字没有意义。
</div>

## Roofline 与算术强度

Roofline 把硬件画成两段屋顶：

1. **带宽屋顶**：可达 FLOP/s $= \text{算术强度} \times \text{带宽}$
2. **算力屋顶**：可达 FLOP/s $= \text{峰值算力}$

两条线的交点叫 **ridge point** $= \text{峰值算力} / \text{带宽}$。工作负载的算术强度在交点左边，就是 memory-bound；在右边，就是 compute-bound。

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{bytes moved}}
$$

H100 SXM 的粗画像（FP16 Tensor Core，不计稀疏）：

- 峰值 $\approx 989$ TFLOP/s
- HBM 带宽 $\approx 3.35$ TB/s
- ridge $\approx 295$ FLOP/byte

A100 80GB 大约 $312 / 2.0 \approx 156$ FLOP/byte。消费卡的脊点往往更靠左，因为带宽相对更差——所以「在 4090 上 decode 已经带宽受限」并不意外。

Horace He 的 [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) 把这件事讲成工程师的直觉：你看到的 kernel 时间，要么在等 ALU，要么在等总线。中间不存在第三种魔法。

## 为什么 Prefill 和 Decode 站在屋顶两侧

只保留主项：读一遍权重，做一遍与 token 数相关的 GEMM。令 $N$ 为参数量，$e$ 为每参数字节，$S$ 为 Prefill 长度，$B$ 为 batch。

**Decode（每步 1 token）**

$$
\mathrm{FLOPs} \approx 2N, \qquad \mathrm{bytes} \approx eN \implies \mathrm{AI} \approx \frac{2}{e}
$$

FP16 时 $\mathrm{AI} \approx 1$。离 295 差两个数量级。KV 会再加一项 $2 L H_{kv} d \cdot t \cdot e$ 的读取，把强度打得更低。

理论 Decode 速率（还没算 kernel 效率）是

$$
\text{tok/s} \approx \frac{\text{带宽}}{eN}
$$

8B、FP16、$eN \approx 16$ GB，H100 上 $\approx 3.35\ \mathrm{TB/s} / 16\ \mathrm{GB} \approx 210$ tok/s。这是 **batch=1 的天花板**，不是你在 notebook 里能立刻摸到的数；真实 kernel 还有启动开销、碎片访问、采样。但它告诉你：换一张「TFLOP 翻倍、带宽不变」的卡，这条上限几乎不动。

**Prefill**

$$
\mathrm{FLOPs} \approx 2N \cdot S \cdot B, \qquad \mathrm{bytes} \approx eN \implies \mathrm{AI} \approx \frac{2SB}{e}
$$

$S=2048, B=1, e=2$ 时 $\mathrm{AI} \approx 2048$，已经在脊点右侧。更长的 prompt、更大的 batch，Prefill 更偏 compute-bound。这也是为什么：

- Prefill 能从 FlashAttention、更好的 GEMM 融合里拿到接近峰值的利用率；
- Decode 从同样这些 kernel 里拿到的是 **更少的 HBM 往返**，不是更高的 TFLOP。

Attention 自身还有 $O(S^2)$ 的计算和（未融合时）多次中间结果往返。FlashAttention 的贡献正是把这条 $O(S^2)$ 的 IO 砍掉——第 3 章。对 Decode，$S$ 是历史长度，query 只有 1 行，Attention 往往不是大头，**权重 GEMM 才是**。

## 交互工具：Roofline 实验室

换 GPU、改精度、拉 Prefill 长度。看两个点：Decode 几乎钉在左下的斜线上；Prefill 随 $S$ 向右走，直到撞上水平的算力屋顶。

<RooflineLab />

建议操作：

1. 保持 8B FP16，把 Prefill 从 128 拖到 4096，观察点何时越过 ridge。
2. 把权重改成 INT4（$e=0.5$）。Decode AI 变成 4，仍然在斜线上，但理论上限 tok/s 大约翻 4 倍——这就是量化对 Decode 的主收益：**少搬字节**。
3. 换成 L4（带宽只有 ~0.3 TB/s）。同样 8B FP16，天花板会难看到让你重新考虑要不要上量化或更小的模型。

## 手算：H100 上 8B 能有多快

把数字写在纸上，不要打开实验室。

**已知：** Llama 3 8B $\approx 8 \times 10^9$ 参数；H100 带宽 $3.35 \times 10^{12}$ B/s；FP16。

1. 权重大小 $eN = 2 \times 8e9 = 16$ GB。
2. batch=1 Decode 上限 $3.35e12 / 16e9 \approx 209$ tok/s。对应 TPOT $\approx 4.8$ ms。
3. 若 KV 已有 8K token：第 1 章算过 KV $\approx 1$ GB。每步还要多读 1 GB，上限变成 $3.35e12 / 17e9 \approx 197$ tok/s。8K 时 KV 还不是主犯。
4. 若 context 拉到 128K：KV $\approx 16$ GB，和权重一样大，上限腰斩到 ~105 tok/s。
5. batch=32、仍假设权重只读一次、KV 按 32 路相加——权重项被 32 个 token 分摊，Decode 开始「像一点 Prefill」。这是吞吐上去、单请求 TPOT 下来的来源。Orca 的连续批处理，本质是让这张卡尽量不要在 batch=1 的斜线上空转。

把第 4 步再换 INT4 权重（约 4 GB）+ FP16 KV（16 GB）：总读取 20 GB，上限 $\approx 167$ tok/s。**只向量化权重、不管 KV，长上下文下收益会迅速消失。** 这就是为什么 KV 量化会在第 4 章作为独立主题出现。

<div class="callout math">
<span class="label">推导</span>
上限公式 <code>带宽 / 每步要读的字节</code> 假设 kernel 达到峰值带宽、且没有重复读取。真实利用率 60–80% 已经算健康。用它做「数量级判断」而不是 SLA 承诺。
</div>

## 代码实战：用 profiler 取证

理论说 Decode 在读权重。证据来自 profiler：GEMM kernel 的时间里，若算术强度低、单位时间字节高，就是 memory-bound。

下面的脚本在 CUDA 上对「假 8B 形状的一层 FFN」做一次 $S=1$ 和一次 $S=2048$ 的对比。没有 GPU 就跳过，保留上一节手算。

```python
import time
import torch
import torch.nn as nn

D, D_FF = 4096, 11008
ffn = nn.Sequential(
    nn.Linear(D, D_FF, bias=False),
    nn.SiLU(),
    nn.Linear(D_FF, D, bias=False),
).cuda().half()

def bench(S, n=50):
    x = torch.randn(1, S, D, device="cuda", dtype=torch.float16)
    for _ in range(10):
        ffn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        ffn(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"S={S:4d}  {ms:7.2f} ms")

bench(1)
bench(2048)
```

你应该看到：$S=2048$ 并不是 $S=1$ 的 2048 倍——因为 $S=1$ 已经被带宽按「读完全部权重」定价，多算一些 FLOP 几乎白送。这就是 Prefill 更「划算」的微观版本。

更进一步（可选）：

```bash
python -m torch.profiler.trace  # 或在代码里用 torch.profiler.profile
# nsys profile --stats=true python bench.py
```

在 nsys / PyTorch profiler 里找 `gemm` / `cutlass` kernel：Decode 形状下，内存吞吐接近峰值、SM 利用率却上不去，就是铁证。

HuggingFace 的 `generate` 默认还夹着采样、张量拷贝、Python 调度。测模型本身时用上面这种 **只含一层、固定形状** 的微基准；测服务时再用 vLLM 的 `benchmark_serving`。不要混。

<div class="callout lab">
<span class="label">实验完成标准</span>
你能向另一个人用「脊点、AI、16 GB / 3.35 TB/s」在白板上推出 8B 的 ~200 tok/s 上限，并指出 128K KV 何时变成和权重同样大的读取项。profiler 是加分，不是门槛。
</div>

## 参考资料

- **Horace He，[Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)**
- **kipply，[Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)**（和上一章是同一篇，这次重点读带宽与 batch 两节）
- NVIDIA，[Roofline 模型与 Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- 下一章预告：Dao et al.，[FlashAttention](https://arxiv.org/abs/2205.14135)——专门修理 Attention 的 IO

阶段 A 到这里结束。若公式和上限你能闭卷复述，可以进入阶段 B：第 3 章会把 Attention 从「一次写出 $S\times S$ 矩阵」改写成 tiling + online softmax。那是 kernel 视角的第一章。

## 自测

<Quiz :questions="[
  {
    prompt: '一个系统报 80 tok/s。为什么这还不够你判断它快不快？',
    options: [
      '因为没有公布用的是哪家模型',
      '因为没有区分这是单请求 TPOT 的倒数，还是多路并发摊出来的吞吐',
      '因为 tok/s 只能在 GPU 上测',
      '因为 80 已经低于任何硬件上限'
    ],
    answer: 1,
    explanation: '同一数字可能来自「一个用户 12.5 ms/token」或「四十个用户在排队」。必须拆成 TTFT / TPOT / 并发。'
  },
  {
    prompt: 'FP16、batch=1 的 Decode，AI ≈ 1 FLOP/byte。H100 ridge ≈ 295。结论是？',
    options: [
      'compute-bound，应换更强的 Tensor Core',
      'memory-bound，优先减少每步读取的字节（量化、GQA、更少的多余拷贝）',
      '已经位于脊点，两边优化收益相同',
      'AI 与屋顶无关，只看显存容量'
    ],
    answer: 1,
    explanation: '1 ≪ 295，落在斜线上。多给 FLOP 没有用；少给字节才有用。'
  },
  {
    prompt: '8B FP16 在 H100 上 batch=1 的 Decode 理论上限大约 200 tok/s。换一张峰值 TFLOP 翻倍、带宽相同的卡，上限如何变？',
    options: [
      '大约翻倍',
      '几乎不变',
      '变成原来的四分之一',
      '取决于词表大小'
    ],
    answer: 1,
    explanation: '上限由带宽/权重大小决定，与峰值算力无关。这正是 memory-bound 的定义。'
  },
  {
    prompt: '长上下文（例如 128K）时，只把权重量化到 INT4、KV 仍是 FP16，为什么收益可能很有限？',
    options: [
      'INT4 不能在 GPU 上跑',
      '此时每步读取可能已被 KV 主导，权重变小之后 KV 项仍在',
      '量化会改变 ridge point',
      'Prefill 会变慢，抵消 Decode 收益'
    ],
    answer: 1,
    explanation: '128K 时 8B 的 KV 约 16 GB，与 FP16 权重相当。权重缩到 4 GB 后，每步仍要读约 20 GB。必须同时处理 KV。'
  }
]" />
