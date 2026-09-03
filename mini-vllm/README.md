# Mini-vLLM

教学用最小 LLM 推理引擎。它会出现在 **第五篇**，用来证明：

Engine → Request → Scheduler → BlockManager → ModelRunner → Sampler

这一条链可以在不复制 vLLM 源码的前提下被实现出来。

当前目录是占位。在第五篇之前请使用 `examples/` 中的 NumPy 教学脚本，不要把空目录当成可服务的引擎。

身份约定：

- `examples/`：章节手算与算法脚本
- `mini-vllm/`：教学引擎（尚未实现）
- vLLM / SGLang / TensorRT-LLM：生产框架；讨论时必须标注版本与日期
