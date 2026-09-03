"""Logits → probabilities → next token. Teaching implementation.

Algorithms follow Hugging Face generation strategies (greedy, temperature,
top-k, nucleus / top-p). This is not a copy of Transformers source.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy() for argmax")
    return logits / temperature


def top_k_mask(logits: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= logits.size:
        return logits
    threshold = np.partition(logits, -k)[-k]
    masked = logits.copy()
    masked[masked < threshold] = -np.inf
    return masked


def top_p_mask(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling: keep the smallest prefix whose mass >= p.

    Hugging Face sorts by probability, then keeps tokens until the
    cumulative probability reaches `p`. Tokens after the cutoff are
    dropped. Ties are broken by the stable argsort used here.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError("p must be in (0, 1]")
    probs = softmax(logits)
    order = np.argsort(-probs, kind="mergesort")
    sorted_probs = probs[order]
    cdf = np.cumsum(sorted_probs)
    keep_sorted = cdf - sorted_probs <= p
    keep_sorted[0] = True
    keep = np.zeros_like(logits, dtype=bool)
    keep[order[keep_sorted]] = True
    masked = logits.copy()
    masked[~keep] = -np.inf
    return masked


def greedy(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def sample(
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    processed = apply_temperature(logits, temperature)
    if top_k is not None:
        processed = top_k_mask(processed, top_k)
    if top_p is not None:
        processed = top_p_mask(processed, top_p)
    probs = softmax(processed)
    rng = rng or np.random.default_rng(0)
    return int(rng.choice(len(probs), p=probs))


def tiny_worked_example() -> dict[str, np.ndarray | int | list[int]]:
    logits = np.array([4.0, 2.0, 1.0, 0.0], dtype=np.float64)
    tokens = ["是", "很", "的", "猫"]
    rng = np.random.default_rng(0)
    draws = [sample(logits, temperature=0.7, top_k=3, rng=np.random.default_rng(i)) for i in range(8)]
    return {
        "tokens": tokens,
        "logits": logits,
        "probs_t1": softmax(logits),
        "probs_t07": softmax(apply_temperature(logits, 0.7)),
        "probs_t02": softmax(apply_temperature(logits, 0.2)),
        "topk2_logits": top_k_mask(logits, 2),
        "topp09_logits": top_p_mask(logits, 0.9),
        "greedy": greedy(logits),
        "draws_t07_k3": draws,
    }


if __name__ == "__main__":
    data = tiny_worked_example()
    np.set_printoptions(precision=4, suppress=True)
    for key, value in data.items():
        print(f"{key}: {value}")
