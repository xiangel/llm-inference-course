# Mini-vLLM

教学用最小 LLM 推理引擎，对应 **第五篇**。

Engine → Request → Scheduler → BlockManager → ModelRunner → Sampler

用 NumPy 跑通 Continuous Batching、Chunked Prefill 和分页 KV。它 **不是** vLLM 源码，也不是可上线的服务。

```bash
PYTHONPATH=mini-vllm python3 -m mini_vllm.engine
python3 -m unittest tests.test_mini_vllm -v
```

身份约定：

- `examples/`：章节手算与算法脚本
- `mini-vllm/`：教学引擎（本目录）
- vLLM / SGLang / TensorRT-LLM：生产框架；讨论时必须标注版本与日期

已知缺口（故意不实现，见 5.7）：Prefix Cache、抢占、换出、CUDA、真实词表、RoPE、混合 Prefill/Decode kernel。
