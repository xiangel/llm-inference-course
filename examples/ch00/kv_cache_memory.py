"""KV Cache memory estimator used by Part 0 numerical examples.

This is a teaching implementation. It matches the standard dense-attention
formula used in later chapters:

    bytes = 2 * num_layers * num_kv_heads * head_dim
            * sequence_length * batch_size * dtype_bytes

The leading 2 counts Key and Value. It does not include model weights,
activation workspace, allocator fragmentation, or CUDA graphs.
"""

from __future__ import annotations

from dataclasses import dataclass


GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    hidden_size: int
    dtype_bytes: int = 2  # BF16 / FP16

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_q_heads


# Official Llama 3.1 hyper-parameters from Grattafiori et al., 2024,
# "The Llama 3 Herd of Models", Table 3. Last verified: 2026-09-03.
LLAMA31_8B = ModelSpec("Llama 3.1 8B", 32, 32, 8, 4096)
LLAMA31_70B = ModelSpec("Llama 3.1 70B", 80, 64, 8, 8192)

# Pedagogical dense MHA model used in Part 1 worked examples.
TEACHING_7B = ModelSpec("Teaching 7B-like MHA", 32, 32, 32, 4096)


def kv_cache_bytes(
    spec: ModelSpec,
    sequence_length: int,
    batch_size: int = 1,
) -> int:
    return (
        2
        * spec.num_layers
        * spec.num_kv_heads
        * spec.head_dim
        * sequence_length
        * batch_size
        * spec.dtype_bytes
    )


def kv_cache_gib(
    spec: ModelSpec,
    sequence_length: int,
    batch_size: int = 1,
) -> float:
    return kv_cache_bytes(spec, sequence_length, batch_size) / GIB


def weight_bytes_approx(num_params: float, dtype_bytes: int = 2) -> int:
    """Approximate weight footprint. 70B BF16 ≈ 140 GiB, not an exact param count."""
    return int(num_params * dtype_bytes)


def flop_matmul(m: int, n: int, k: int) -> int:
    """FLOPs for one (m, k) @ (k, n) GEMM using the 2*m*n*k convention."""
    return 2 * m * n * k


def attention_qkv_flops(batch: int, seq: int, hidden: int) -> int:
    return 3 * flop_matmul(batch * seq, hidden, hidden)


def attention_scores_flops(batch: int, heads: int, q_len: int, kv_len: int, head_dim: int) -> int:
    return flop_matmul(batch * heads * q_len, kv_len, head_dim)


def attention_av_flops(batch: int, heads: int, q_len: int, kv_len: int, head_dim: int) -> int:
    return flop_matmul(batch * heads * q_len, head_dim, kv_len)


def attention_out_proj_flops(batch: int, seq: int, hidden: int) -> int:
    return flop_matmul(batch * seq, hidden, hidden)


def print_worked_examples() -> None:
    print("=== KV Cache ===")
    for spec, seq, batch in (
        (LLAMA31_8B, 8192, 1),
        (LLAMA31_70B, 8192, 1),
        (LLAMA31_70B, 8192, 8),
        (LLAMA31_70B, 131072, 1),
        (TEACHING_7B, 1024, 4),
    ):
        gib = kv_cache_gib(spec, seq, batch)
        print(
            f"{spec.name}: seq={seq} batch={batch} "
            f"kv={kv_cache_bytes(spec, seq, batch)} B ({gib:.4f} GiB)"
        )

    print("=== Weights (approx) ===")
    print("70B BF16 weights ≈", weight_bytes_approx(70e9) / GIB, "GiB")
    print("8B BF16 weights ≈", weight_bytes_approx(8e9) / GIB, "GiB")

    batch, seq, hidden, heads = 4, 1024, 4096, 32
    head_dim = hidden // heads
    print("=== Teaching attention FLOPs (one layer) ===")
    qkv = attention_qkv_flops(batch, seq, hidden)
    scores = attention_scores_flops(batch, heads, seq, seq, head_dim)
    av = attention_av_flops(batch, heads, seq, seq, head_dim)
    out = attention_out_proj_flops(batch, seq, hidden)
    print("QKV", qkv)
    print("QK^T", scores)
    print("AV", av)
    print("out", out)
    print("total", qkv + scores + av + out)

    print("=== Decode FLOPs vs Prefill (one layer, last token only) ===")
    dec_scores = attention_scores_flops(batch, heads, 1, seq, head_dim)
    dec_av = attention_av_flops(batch, heads, 1, seq, head_dim)
    print("decode QK^T", dec_scores)
    print("decode AV", dec_av)


if __name__ == "__main__":
    print_worked_examples()
