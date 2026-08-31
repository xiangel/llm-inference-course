---
title: KV Cache：避免重复计算
description: 理解生成时为何要保存旧 token 的 Key/Value，如何粗估容量，以及缓存为何不会替代上下文管理。
---

# KV Cache：避免重复计算

## 为什么要关心它

模型已经从“今天天气”生成了“不错”。要再生成下一个 token，如果每一步都重新处理整句，前面 token 的中间结果会被反复计算。回复越长，浪费越明显。

KV Cache 保存每层对历史 token 已算出的 Key 和 Value。新 token 只需产生自己的 K、V，并读取旧缓存来关注前文。这是几乎所有高效自回归推理服务的基本机制，也是长会话占显存的主要原因之一。

## 一个朴素心智模型

回到上一章的索引卡：Prompt 的每张卡已经做好了“索引线索”和“内容”。KV Cache 就是把它们按层收进档案盒。

生成下一词时，不重写旧卡；只制作新卡、放进盒子，然后用新 Query 查阅整盒卡。缓存保存的是模型内部数值，不是原始文本，也不能跨不同模型权重或不同请求随意复用。

## 图：先建档，后续只追加

<KvCacheFlow />

Prefill 是“完整 prompt 一次建档”；Decode 是“读旧档 + 写一张新卡”。聊天历史越长，Decode 每一步需要读取的档案越多。

## 跟着一次会话走

1. 用户发送 prompt。服务在 Prefill 中计算所有输入 token 的 K、V，并建立该请求专属的缓存。
2. 模型采样出第一个输出 token。它的 K、V 被追加到缓存。
3. 生成下一个 token 时，各层读取既有 K、V，避免重算整个历史。
4. 请求完成，服务可释放该请求的块；若用户带着历史发起新请求，是否命中前缀缓存取决于引擎、相同前缀和缓存策略，不能假设必然命中。

注意：KV Cache 避免的是历史 token 的 K/V **计算**，不是取消对历史的读取，也不会让长上下文免费。

## 容量：一个够用的估算式

单请求 KV Cache 的近似字节数：

`2 × 层数 × KV 头数 × 每头维度 × token 数 × 每元素字节数`

其中的 `2` 分别代表 Key 与 Value。该式假设没有额外对齐、元数据、分页碎片或实现特定布局。

| 变量 | 含义 | 应从哪里确认 |
|---|---|---|
| 层数 | Transformer block 数 | 模型配置 |
| KV 头数 | 保存 K/V 的头数 | 模型配置；GQA/MQA 常小于注意力头数 |
| 每头维度 | 单个头的隐藏维度 | 模型配置 |
| token 数 | prompt 加已生成 token | 请求与输出预算 |
| 每元素字节数 | KV 数据类型大小 | 推理引擎与 KV 精度设置 |

**例子（仅作容量算术，不是性能基准）**：32 层、8 个 KV 头、每头 128 维、4,096 token、FP16/BF16（2 字节）时，约为 `2 × 32 × 8 × 128 × 4,096 × 2 = 512 MiB`，即单个序列约半 GiB。并发序列会近似相加。

边界也很重要：真实占用还要留出模型权重、激活、CUDA/运行时缓冲、分页与调度余量；某些架构使用不同 KV 表示或压缩方案，上式不适用。容量规划应以目标模型和引擎实际遥测为准。

## 交互估算

<KvCalculator />

组件把权重的粗略占用也列出，便于建立量级感；它明确省略了运行时开销，不能作为“能否部署”的最终判定。

## 可选代码实验

无需 GPU 也能理解。可在 Colab 用一个小模型，分别以 `use_cache=False` 和 `use_cache=True` 生成短文本，打印每步传入模型的 token 长度与 `past_key_values` 的结构。只比较“旧 token 是否再次作为完整输入”，不要把 Colab 的墙钟时间当生产基准。

## 连接真实生产系统

- vLLM 的 [PagedAttention](https://docs.vllm.ai/en/latest/design/paged_attention.html) 将 KV 以块管理，类似虚拟内存页，减少不同长度请求造成的浪费。
- 调度器在并发、最大上下文和输出预算间分配 KV 空间；一次很长请求可能降低其他用户可用容量。
- 前缀缓存对大量共享系统提示或文档前缀很有价值，但正确性依赖 token 前缀完全一致以及引擎的隔离策略。

## 常见误解

- **“KV Cache 缓存了答案。”** 它缓存的是每层内部 K/V；输出仍要逐 token 计算与采样。
- **“开启缓存后长上下文不再慢。”** 每步仍需读取更多历史 KV，且缓存本身会占显存。
- **“注意力头数就是 KV 头数。”** GQA/MQA 可以让多个查询头共享更少 KV 头；必须读模型配置。

## 小结

- KV Cache 让 Decode 复用历史 K/V，而不是重复建立它们。
- 它用更多显存换取更少重复计算。
- 用简式先估量级，部署前用实际模型、精度、并发和运行时余量验证。

## 自测

<Quiz :questions="[
  { prompt: 'KV Cache 在 Decode 中直接避免了什么？', options: ['重新计算历史 token 的 K/V', '读取模型权重', '采样下一个 token', 'tokenize 用户输入'], answer: 0, explanation: '旧 K/V 被复用；仍要读权重、读取历史缓存并计算新 token。' },
  { prompt: '估算 KV Cache 时，哪个头数应使用？', options: ['词表大小', 'KV 头数', 'CPU 核数', 'batch 中最长输出的字符数'], answer: 1, explanation: 'GQA/MQA 下 KV 头数可能少于查询头数。' },
  { prompt: '为什么估算出的 512 MiB 不能视为全部显存需求？', options: ['因为 token 不存在', '真实部署还需权重、运行时缓冲与布局开销', '因为 KV 必定在 CPU', '因为 FP16 没有字节数'], answer: 1, explanation: '公式仅是 KV 的近似值，容量规划还要计算其他开销。' }
]" />

## 参考资料

- [Kwon 等：Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM PagedAttention 的原始论文，解释块化 KV 管理。
- [vLLM：Paged Attention 设计文档](https://docs.vllm.ai/en/latest/design/paged_attention.html) — 官方实现概览与术语。
- [Hugging Face：KV cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache) — 官方缓存类型、用途与限制。
