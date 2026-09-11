"""One-layer toy LM used by ModelRunner. Not a checkpoint loader.

Identity: ``mini_vllm.model.ToyLM``. Distinct from ``examples/ch01`` ToyLM
(no RoPE here — paging and scheduling are the lesson, not position encoding).
"""

from __future__ import annotations

import numpy as np

from mini_vllm.sampler import greedy


class ToyLM:
    def __init__(self, vocab_size: int = 8, hidden: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.tok_emb = rng.normal(scale=0.3, size=(vocab_size, hidden))
        self.w_q = rng.normal(scale=0.3, size=(hidden, hidden))
        self.w_k = rng.normal(scale=0.3, size=(hidden, hidden))
        self.w_v = rng.normal(scale=0.3, size=(hidden, hidden))
        self.w_o = rng.normal(scale=0.3, size=(hidden, hidden))
        self.w_ff1 = rng.normal(scale=0.3, size=(hidden, hidden * 2))
        self.w_ff2 = rng.normal(scale=0.3, size=(hidden * 2, hidden))
        self.lm_head = rng.normal(scale=0.3, size=(hidden, vocab_size))

    def rmsnorm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        return x / rms

    def embed(self, ids: np.ndarray) -> np.ndarray:
        return self.tok_emb[ids]

    def qkv(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self.embed(np.asarray(token_ids, dtype=np.int64))
        xn = self.rmsnorm(x)
        return x, xn @ self.w_q, xn @ self.w_k, xn @ self.w_v

    def finish(self, x: np.ndarray, attn: np.ndarray) -> np.ndarray:
        h = x + attn @ self.w_o
        ff = np.maximum(self.rmsnorm(h) @ self.w_ff1, 0.0)
        h = h + ff @ self.w_ff2
        return self.rmsnorm(h) @ self.lm_head

    def forward(
        self,
        x: np.ndarray,
        past_kv: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """x: [B, T, H]. Returns logits [B, T, V] and new K/V [B, T, H]."""
        xn = self.rmsnorm(x)
        q = xn @ self.w_q
        k = xn @ self.w_k
        v = xn @ self.w_v
        if past_kv is None:
            k_full, v_full = k, v
        else:
            pk, pv = past_kv
            k_full = np.concatenate([pk, k], axis=1)
            v_full = np.concatenate([pv, v], axis=1)
        attn = np.stack(
            [causal_attn(q[b], k_full[b], v_full[b]) for b in range(x.shape[0])],
            axis=0,
        )
        logits = self.finish(x, attn)
        return logits, k, v


def causal_attn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = q.shape[-1]
    q_len, kv_len = q.shape[0], k.shape[0]
    scores = q @ k.T / np.sqrt(d)
    q_pos = np.arange(kv_len - q_len, kv_len)[:, None]
    k_pos = np.arange(kv_len)[None, :]
    scores = np.where(k_pos <= q_pos, scores, -np.inf)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ v


def pick_token(logits: np.ndarray) -> int:
    return greedy(logits)
