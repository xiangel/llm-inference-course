# examples

与书籍章节对应的可运行教学代码。依赖：Python 3.10+、NumPy。

```bash
python3 -m unittest tests.test_ch00_ch01 -v
python3 examples/ch00/kv_cache_memory.py
python3 examples/ch01/attention.py
python3 examples/ch01/rope.py
python3 examples/ch01/sampling.py
python3 examples/ch01/generate_one_token.py
```

这些实现用 NumPy 表达公式，方便在没有 GPU 的环境验证数字。它们 **不是** PyTorch 生产模型，也 **不是** vLLM 源码。
