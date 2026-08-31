---
title: 推理性能的语言
description: 用 TTFT、TPOT、throughput 与 Goodput 描述用户体验和服务能力，避免用单一“速度”掩盖问题。
---

# 推理性能的语言

## 为什么要关心它

“模型每秒 100 token”听起来很快，却不能回答用户等多久才看到第一个字、多人同时请求时是否排队、或超时请求算不算有效工作。推理优化必须先用共同语言描述目标，否则可能把离线吞吐做高，却让交互体验变差。

本章只建立指标与诊断框架；不需要 Colab，也不把任意单机数字外推到你的服务。

## 一个朴素心智模型

把一次流式回答看成餐厅出餐：

- **TTFT（time to first token）**：从服务收到请求到用户看见第一个可用 token 的等待；像第一道菜何时上桌。
- **TPOT（time per output token）**：第一个 token 后，相邻输出 token 的平均间隔；像后续菜的出餐节奏。
- **Throughput（吞吐量）**：系统单位时间生成或处理的 token 数；像厨房总出菜量。
- **Goodput（有效吞吐）**：真正满足服务目标的吞吐，例如在约定延迟内完成、未取消、格式/质量通过的请求所贡献的 token 或请求。

吞吐高不必然 TTFT 低：把请求攒成大批次可能让 GPU 更忙，却让新用户等更久。

## 图：一次请求的时间线

<PrefillDecodeDiagram />

用户的等待可粗略分成排队、Prefill、首 token 发送，以及随后多个 Decode 间隔。图描述计算阶段；实际 TTFT 还可能包含网关、tokenization、调度和网络传输。指标定义应明确计时起止点。

## 跟着一次测量走

假设要评估一个聊天端点：

1. 先写合同：TTFT 从“网关收到完整请求”还是“客户端发出请求”开始？首 token 是服务内部产生、写入 socket，还是浏览器渲染？选一个并固定。
2. 为每个请求记录到达、开始执行、首个流式片段、最后片段、取消/错误，以及输入和输出 token 数。
3. 从同一批记录分别计算延迟分位数（如 p50/p95）、每请求 TPOT、总 token throughput。
4. 定义 Goodput 门槛，例如“未出错且 TTFT、完成时间均在 SLO 内”的完成请求；被客户端取消或超时的工作不能悄悄计为成功。
5. 按输入长度、输出长度、并发、模型、硬件和采样设置分组。平均值会掩盖长 prompt 与排队造成的尾部延迟。

报告结果时附带负载形状、计时边界、成功率和资源配置。没有这些上下文，两个“tok/s”不可比较。

## 生产连接

- **交互聊天**：TTFT 和尾部 TTFT 往往直接影响体感；流式传输能让用户尽早看到内容，但不能减少已经发生的排队。
- **离线批处理**：更关心总吞吐和单位成本，允许较大批量与较长等待。
- **多租户服务**：Goodput 是防止“GPU 很忙但用户超时”的护栏。队列长度、取消率和拒绝率应与 GPU 利用率一起看。
- **优化定位**：长输入导致 TTFT 高，先检查 Prefill、排队与 prompt；TPOT 高，则检查 Decode、KV 读取、权重读取和批处理策略。下一章解释这些硬件瓶颈。

## 常见误解

- **“吞吐等于用户速度。”** 吞吐是整体产能；用户看到的是 TTFT、TPOT 和尾部延迟。
- **“TTFT 只等于模型 Prefill。”** 计时范围内还可能有排队、序列化、tokenization 和网络。
- **“完成的 token 都是 Goodput。”** 超过 SLO、被取消或不符合业务成功条件的工作未必有效。
- **“只看平均值足够。”** 交互服务的长尾常由排队与长请求决定，需看分位数和分组。

## 小结

- TTFT 描述首字等待，TPOT 描述后续输出节奏。
- Throughput 描述总产能，Goodput 把产能与业务成功条件连起来。
- 先约定测量边界与负载，再讨论优化是否有效。

## 自测

<Quiz :questions="[
  { prompt: 'TTFT 最直接反映什么？', options: ['模型参数量', '用户看到第一个 token 前的等待', '输出总 token 数', 'GPU 温度'], answer: 1, explanation: '它衡量首个可用流式输出之前的端到端等待，具体边界需明确。' },
  { prompt: '系统吞吐升高但 TTFT 变差，哪种情况可能发生？', options: ['更大批次提高利用率但增加排队', 'tokenizer 被删除', '模型停止生成', '所有请求变成空 prompt'], answer: 0, explanation: '批量策略可以提高总产能，同时推迟个别请求开始。' },
  { prompt: 'Goodput 为什么需要业务定义？', options: ['它固定等于 GPU 利用率', '有效工作取决于 SLO、取消和成功条件', '它不需要请求日志', '它只测网络带宽'], answer: 1, explanation: '没有成功条件和延迟目标，无法区分有用输出与无效工作。' }
]" />

## 参考资料

- [NVIDIA GenAI-Perf 指标说明](https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/) — 官方博客介绍 TTFT、TPOT、吞吐等常用定义；使用时仍需声明自己的计时边界。
- [NVIDIA GenAI-Perf 文档](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — 请求级与汇总指标的官方参考。
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 生成式 AI 遥测属性规范，可用于统一请求记录字段。
