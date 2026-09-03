"""Rotary Position Embedding (RoPE). Teaching implementation.

Two layouts exist in the wild:

1. Paper / interleaved pairs (RoFormer, Su et al. 2021): rotate (x0, x1),
   (x2, x3), ...
2. Llama / Hugging Face `rotate_half`: rotate the first half of the vector
   against the second half.

They are the same transform up to a fixed dimension permutation. This file
implements both and never claims the teaching layout is Llama source.
"""

from __future__ import annotations

import numpy as np


def inv_freq(dim: int, theta: float = 10000.0) -> np.ndarray:
    """θ_i = theta ** (-2i / dim) for i = 0, 2, 4, ..."""
    return 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))


def rotary_angles(seq_len: int, dim: int, theta: float = 10000.0) -> np.ndarray:
    """Return (seq, dim/2) angles: position * θ_i."""
    positions = np.arange(seq_len, dtype=np.float64)
    return np.outer(positions, inv_freq(dim, theta))


def rotate_interleaved(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """RoFormer-style: treat adjacent even/odd dims as a 2D plane.

    x: (..., dim)  cos/sin: (..., dim/2) broadcastable to x's even/odd slice.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    rotated = np.empty_like(x)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Hugging Face Llama helper: (-x_second, x_first)."""
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return np.concatenate((-second, first), axis=-1)


def apply_rope_llama(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Llama / HF layout: cos/sin are duplicated to full dim, then
    x * cos + rotate_half(x) * sin.

    This matches the *algorithm* of `apply_rotary_pos_emb` in recent
    Hugging Face Transformers. It is a teaching reimplementation, not a
    copy of their source.
    """
    return x * cos + rotate_half(x) * sin


def expand_cos_sin_for_llama(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """angles: (seq, dim/2) -> cos/sin of shape (seq, dim) by repeating halves."""
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
    return cos, sin


def expand_cos_sin_for_interleaved(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """angles: (seq, dim/2) used directly on even/odd pairs."""
    return np.cos(angles), np.sin(angles)


def tiny_worked_example() -> dict[str, np.ndarray]:
    """head_dim=4, positions 0..2, theta=10000. Hand-checkable."""
    dim = 4
    theta = 10000.0
    angles = rotary_angles(3, dim, theta)
    q = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    cos_i, sin_i = expand_cos_sin_for_interleaved(angles)
    q_paper = rotate_interleaved(q, cos_i, sin_i)

    cos_l, sin_l = expand_cos_sin_for_llama(angles)
    q_llama = apply_rope_llama(q, cos_l, sin_l)

    # Relative-position check: q_m · k_n depends on (n-m) after RoPE.
    k = q.copy()
    k_paper = rotate_interleaved(k, cos_i, sin_i)
    dots = q_paper @ k_paper.T
    return {
        "inv_freq": inv_freq(dim, theta),
        "angles": angles,
        "q_paper": q_paper,
        "q_llama": q_llama,
        "dots": dots,
    }


if __name__ == "__main__":
    data = tiny_worked_example()
    np.set_printoptions(precision=6, suppress=True)
    for key, value in data.items():
        print(key)
        print(value)
        print()
