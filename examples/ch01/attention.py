"""Causal scaled dot-product attention. Teaching implementation in NumPy.

This is NOT Hugging Face or vLLM source. It implements the formula from
Vaswani et al., 2017, section 3.2.1:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + causal_mask) V
"""

from __future__ import annotations

import math

import numpy as np


def causal_mask(q_len: int, kv_len: int, dtype=np.float64) -> np.ndarray:
    """Return an additive mask: 0 for allowed positions, -inf otherwise.

    When q_len == kv_len this is the usual lower-triangular mask. When
    q_len == 1 and kv_len == T (decode), every cached key is visible.
    """
    q_pos = np.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = np.arange(kv_len)[None, :]
    allowed = k_pos <= q_pos
    mask = np.zeros((q_len, kv_len), dtype=dtype)
    mask[~allowed] = -np.inf
    return mask


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    causal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """q, k, v: (batch, heads, seq, head_dim) or (seq, head_dim).

    Returns (output, attention_weights).
    """
    squeezed = q.ndim == 2
    if squeezed:
        q = q[None, None]
        k = k[None, None]
        v = v[None, None]

    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / math.sqrt(d_k)
    if causal:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        scores = scores + causal_mask(q_len, kv_len, dtype=scores.dtype)

    weights = softmax(scores, axis=-1)
    out = weights @ v
    if squeezed:
        return out[0, 0], weights[0, 0]
    return out, weights


def project_qkv(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x: (seq, hidden). Weights: (hidden, hidden). Teaching MHA-ready linear maps."""
    return x @ w_q, x @ w_k, x @ w_v


def split_heads(x: np.ndarray, num_heads: int) -> np.ndarray:
    seq, hidden = x.shape
    head_dim = hidden // num_heads
    return x.reshape(seq, num_heads, head_dim).transpose(1, 0, 2)


def merge_heads(x: np.ndarray) -> np.ndarray:
    heads, seq, head_dim = x.shape
    return x.transpose(1, 0, 2).reshape(seq, heads * head_dim)


def multi_head_attention(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    q, k, v = project_qkv(x, w_q, w_k, w_v)
    q_h, k_h, v_h = split_heads(q, num_heads), split_heads(k, num_heads), split_heads(v, num_heads)
    # (heads, seq, dim)
    out_h, _ = scaled_dot_product_attention(q_h[None], k_h[None], v_h[None])
    return merge_heads(out_h[0]) @ w_o


def tiny_worked_example() -> dict[str, np.ndarray]:
    """Hand-checkable 4-token, 1-head example used in chapter 1.3."""
    rng = np.random.default_rng(0)
    q = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )
    k = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    v = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    out, weights = scaled_dot_product_attention(q, k, v, causal=True)
    return {
        "q": q,
        "k": k,
        "v": v,
        "weights": weights,
        "out": out,
        "raw_scores": q @ k.T / math.sqrt(q.shape[-1]),
        "rng_probe": rng.normal(size=1),
    }


if __name__ == "__main__":
    data = tiny_worked_example()
    np.set_printoptions(precision=4, suppress=True)
    print("raw scores\n", data["raw_scores"])
    print("weights\n", data["weights"])
    print("out\n", data["out"])
