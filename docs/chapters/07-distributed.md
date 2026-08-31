---
title: GPU 与推理瓶颈
description: 用 Roofline 的直觉区分算力受限与带宽受限，理解为什么 Prefill 和 Decode 需要不同优化。
---

# GPU 与推理瓶颈

## 为什么要关心它

“GPU 利用率低”不自动说明模型写得不好。生成一个 token 时，GPU 可能主要在等待从显存读取模型权重；处理很长 prompt 时，又可能主要在执行大量计算。对错的优化方向会白白增加复杂度：例如只追求更高 FLOPS，未必改善逐 token 延迟。

## 一个朴素心智模型

GPU 有两种重要能力：

- **计算能力**：每秒能做多少数学运算，像工厂的加工机器；
- **显存带宽**：每秒能把多少数据从显存搬给机器，像通往工厂的货运通道。

Roofline 模型问一个简单问题：每搬运一个字节，做了多少计算？这个比值叫算术强度。低算术强度时，货运通道先堵住，属于**带宽受限**；高到一定程度后，加工机器先满，属于**算力受限**。

唯一需要的公式是：

`可达到的性能 = min(峰值计算性能, 显存带宽 × 算术强度)`

它的两项含义完整如下：峰值计算性能是 GPU 在目标精度下的理论上限；显存带宽是每秒可搬运字节数；算术强度是该工作每搬运一个字节完成的运算量。取较小项，是因为计算和数据供给中先达到极限的一方决定上限。它是定位直觉，不是端到端延迟预测器。

## 图：把 Prefill 与 Decode 放到屋顶下

<RooflineLab />

组件的数值是简化示例：它仅近似考虑读取一次权重，省略 Attention 额外 I/O、内核开销、通信与调度。用它观察点相对屋顶的位置，而不要把“理论上限”当实测 benchmark。

## 跟着两类工作走

**Prefill**：完整 prompt 的许多位置可并行处理。矩阵计算规模大，数据复用机会多，常更接近算力受限。长 prompt 仍会使 Attention 与 KV 写入增加工作，所以不能简单地说“必定很快”。

**Decode**：每一步只推进一个或很少 token。为生成这个小结果，模型往往要读取大量权重，并读取不断增长的 KV Cache；可复用计算较少，常更接近带宽受限。提高 batch 或把多个请求连续调度，可能提高复用和总吞吐，但也会改变排队与 TTFT。

排查时按顺序做：

1. 用上一章的 TTFT、TPOT、吞吐和 Goodput 确定坏的是哪一段。
2. 分开测 Prefill 与 Decode，并固定模型、精度、输入/输出长度、并发和硬件。
3. 用 profiler 查看 kernel 时间、显存读写、计算利用率和通信时间；不要只看一个“GPU utilization”。
4. 再选择措施：带宽受限优先考虑权重/KV 精度、批处理与内存访问；算力受限再考虑更高效内核、并行度或模型计算量。

## 生产连接

- 权重量化可能减少从显存读取的字节数，因此常对带宽受限 Decode 有帮助；它也可能引入反量化成本和质量风险，必须在目标硬件实测。
- KV Cache 的容量和读取量随上下文增长。分页、前缀缓存和上下文预算同时影响容量、调度与延迟。
- 多 GPU 时，张量并行会加入通信；通信链路也有“带宽/延迟”限制。单卡 Roofline 不能替代多卡端到端剖析。
- 容量不足时 CPU/磁盘卸载能让请求完成，却可能把瓶颈转移到 PCIe 或存储；这是容量权衡，不是免费加速。

## 常见误解

- **“更多 TFLOPS 一定更快。”** 对带宽受限工作，额外计算峰值可能几乎无效。
- **“Decode 与 Prefill 可以用同一个最佳 batch。”** 两者并行形状、资源与用户目标不同。
- **“Roofline 给出真实 tok/s。”** 它给上界和分类直觉；真实值受内核、KV、通信、调度、网络等影响。
- **“GPU 利用率低就该加请求。”** 先判断是在等数据、等通信、受小 kernel 限制还是被队列策略限制。

## 小结

- Roofline 把性能问题先分为数据搬运与计算两类。
- Prefill 常有更多并行与复用；逐 token Decode 常更受内存读取制约。
- 用请求指标和 profiler 证据定位，再选择优化，且不要把简化模型当基准。

## 自测

<Quiz :questions="[
  { prompt: 'Roofline 直觉中的低算术强度通常意味着什么？', options: ['更可能受显存带宽限制', '一定没有 GPU', '模型不需要权重', '输出必定错误'], answer: 0, explanation: '每字节做的计算少时，数据搬运更可能先成为限制。' },
  { prompt: '为什么 Decode 常比 Prefill 更容易带宽受限？', options: ['Decode 从不读取权重', '它每步工作很小却仍要读取大量权重和历史 KV', '它能看未来 token', '它不使用显存'], answer: 1, explanation: '小量新计算难以摊薄大规模数据读取。' },
  { prompt: '准备优化 TPOT 时，第一步合适的做法是什么？', options: ['只比较广告中的 TFLOPS', '固定负载并用 profiler 与请求指标区分 Decode 瓶颈', '直接把请求全部卸载到磁盘', '忽略 KV Cache'], answer: 1, explanation: '先用证据判断是带宽、计算、通信还是调度问题。' }
]" />

## 参考资料

- [Williams 等：Roofline](https://doi.org/10.1145/1498765.1498785) — Roofline 模型原始论文，给出“计算与数据移动共同限制性能”的框架。
- [NVIDIA Nsight Compute Roofline 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-model) — 官方 profiler 中 Roofline 分析的解释与使用边界。
- [NVIDIA：Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) — 官方混合精度与 Transformer 内核资料；硬件支持和效果应以对应版本文档与实测为准。
