---
title: 第 10 章 · PagedAttention 与前缀缓存
description: 用小型块池模拟理解 KV Cache 如何分页分配、复用共享前缀。
---

# 第 10 章 · PagedAttention 与前缀缓存

**学时** 1–2 小时 · **需要** Python 与免费 Colab（只运行 CPU 块池模拟）· **本章不需要** GPU 或真实模型

一句话回答：**为什么 KV Cache 要按固定大小的块管理，而不是为每条请求预留一整段连续内存？**

<Checklist
  slug="10-paged-attention"
  :items="[
    { id: 'blocks', label: '能解释逻辑 token 位置与物理块的区别' },
    { id: 'pool', label: '在 Colab 运行块池模拟' },
    { id: 'prefix', label: '能说明前缀缓存的命中条件' },
    { id: 'quiz', label: '完成自测' }
  ]"
/>

## 为什么

每条生成请求的输出长度未知。若服务为它预留一块足以容纳最大长度的连续 KV Cache，短请求浪费大量空间；不同长度请求完成与加入后也会留下难以利用的碎片。KV Cache 是运行时状态，常常是并发上限的决定因素。

PagedAttention 将每条序列的 KV Cache 切为等大的逻辑块，再把逻辑块映射到块池中的任意空闲物理块。序列增长时只取新块，不要求与旧块相邻。

## 心智模型

这像操作系统的虚拟内存：程序以为自己拥有连续页号，内存管理器实际把页放到任意物理页。这里“页”存的是若干 token 的 K/V；block table 是序列逻辑位置到物理块 ID 的目录。注意：这是有用类比，不是 CPU 页表的逐字复刻。

## 数据流

<KvCacheFlow />

图已说明 Prefill 写入、Decode 复用并追加 KV。本章进一步把图中的连续 Cache 想成块表：`[逻辑块 0, 逻辑块 1] → [物理块 7, 物理块 2]`。

## 按步骤实验

### 1. 运行块池

Notebook 创建固定数量的整数块，不存真实张量。`allocate(sequence_id)` 给序列一个块；当 token 数跨过 `block_size`，再分配一个；`free` 将它归还。打印 block table 和空闲列表。

<a class="colab-link" href="https://colab.research.google.com/github/xiangel/llm-inference-course/blob/main/notebooks/10_paged_kv_blocks.ipynb" target="_blank" rel="noreferrer">在 Google Colab 打开块池模拟器 ↗</a>

### 2. 制造碎片

先让 A、B、C 依次占块，再完成 B。连续内存思路会在中间留下洞；块池让新请求 D 直接使用 B 归还的块。物理块 ID 不连续不是错误，只要 block table 正确。

### 3. 模拟共享前缀

让两个请求拥有完全相同的系统提示词和文档开头。第一个请求完成 Prefill 后，可将完整前缀块登记为可复用；第二个请求命中后引用这些块，并只为不同的后缀分配新块。真实实现还需要引用计数：共享块在最后一个引用离开前绝不能释放或覆盖。

### 4. 识别未命中

只改一个前缀 token，或让前缀只填满半块。模拟器应把它当作不同键或不可共享部分。前缀缓存不是“语义相似缓存”，它依赖相同模型配置和相同 token 前缀；模板、采样参数或多模态输入的影响要以实现规则为准。

## 生产连接

阅读 [vLLM PagedAttention 论文](https://arxiv.org/abs/2309.06180) 了解动机，并在第 13 章对照 vLLM V1 的 `block_pool.py` 与 KV cache manager。生产系统还要处理块哈希、引用计数、LRU 淘汰、并发安全、缓存命中指标和 OOM 恢复。块大小是实现和配置选择，不要把 Notebook 的值当作 vLLM 的固定事实。

## 常见误解

- PagedAttention 不是把权重分页；它主要管理每个请求不断增长的 KV Cache。
- 前缀缓存不能让不同文字“智能地复用”，必须满足精确的可缓存前缀条件。
- 命中缓存仍要做后续 Decode，也不自动消除网络、排队和输出时间。
- CPU 的整数块模拟验证生命周期，不测量 Attention kernel。

## 小结

- 固定大小 KV 块避免为未知长度预留整段连续空间。
- block table 将逻辑顺序映射到离散物理块。
- 前缀缓存以共享完整前缀块换取更少的 Prefill 和分配。

## 自测

<Quiz :questions="[
  { prompt: 'block table 的核心作用是？', options: ['保存模型权重', '将序列的逻辑块映射到物理 KV 块', '把 HTTP 转为 JSON', '训练 Tokenizer'], answer: 1, explanation: '序列可逻辑连续，而物理块无需相邻。' },
  { prompt: '两请求何时最可能复用前缀缓存？', options: ['问题意思相近', '模型和 token 化后的前缀满足相同缓存条件', '输出长度相同', '来自同一 IP'], answer: 1, explanation: '前缀缓存依赖精确、可验证的输入状态。' },
  { prompt: '共享 KV 块为何需要引用计数？', options: ['让块更大', '防止仍被另一请求使用的块被释放或覆盖', '减少 Tokenizer 词典', '停止 Decode'], answer: 1, explanation: '共享资源必须等最后一个使用者离开后才能回收。' }
]" />

## 参考资料

- [vLLM 论文：Efficient Memory Management](https://arxiv.org/abs/2309.06180) — PagedAttention 的原始主资料。
- [vLLM 文档](https://docs.vllm.ai/en/latest/) — 按当前版本查看前缀缓存和运行配置。
- [vLLM GitHub 仓库](https://github.com/vllm-project/vllm) — 第 13 章源码映射的主仓库。
