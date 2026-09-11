"""Prefill vs Decode shapes, FLOPs, and a tiny KV-cached loop.

Teaching code. Not vLLM. The toy cache stores K and V for one sequence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ch00"))
sys.path.insert(0, str(ROOT / "ch01"))

from attention import scaled_dot_product_attention  # noqa: E402
from generate_one_token import ToyLM  # noqa: E402
from kv_cache_memory import (  # noqa: E402
    LLAMA31_70B,
    TEACHING_7B,
    ModelSpec,
    attention_av_flops,
    attention_scores_flops,
    kv_cache_gib,
)
from rope import apply_rope_llama, expand_cos_sin_for_llama, rotary_angles  # noqa: E402
from sampling import greedy  # noqa: E402


def prefill_decode_score_flops(seq: int = 1024, batch: int = 4, heads: int = 32, head_dim: int = 128):
    prefill = attention_scores_flops(batch, heads, seq, seq, head_dim)
    decode = attention_scores_flops(batch, heads, 1, seq, head_dim)
    return prefill, decode, prefill / decode


class CachedToyLM:
    """ToyLM plus a single-sequence K/V cache. Still one layer, not an engine."""

    def __init__(self, seed: int = 0):
        self.model = ToyLM(seed=seed)
        self.k: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def _qkv(self, token_ids: list[int], start_pos: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.model.embed[np.array(token_ids)]
        x_norm = self.model.rmsnorm(x)
        q = x_norm @ self.model.w_q
        k = x_norm @ self.model.w_k
        v = x_norm @ self.model.w_v
        angles = rotary_angles(start_pos + len(token_ids), self.model.hidden)[start_pos:]
        cos, sin = expand_cos_sin_for_llama(angles)
        return apply_rope_llama(q, cos, sin), apply_rope_llama(k, cos, sin), v

    def prefill(self, prompt_ids: list[int]) -> np.ndarray:
        q, k, v = self._qkv(prompt_ids, start_pos=0)
        self.k, self.v = k, v
        attn, _ = scaled_dot_product_attention(q, k, v, causal=True)
        h = self.model.embed[np.array(prompt_ids)] + attn @ self.model.w_o
        ff = np.maximum(self.model.rmsnorm(h) @ self.model.w_ff1, 0.0)
        h = h + ff @ self.model.w_ff2
        return (self.model.rmsnorm(h) @ self.model.lm_head)[-1]

    def decode_one(self, new_id: int) -> np.ndarray:
        assert self.k is not None and self.v is not None
        q, k_new, v_new = self._qkv([new_id], start_pos=self.k.shape[0])
        self.k = np.concatenate([self.k, k_new], axis=0)
        self.v = np.concatenate([self.v, v_new], axis=0)
        attn, _ = scaled_dot_product_attention(q, self.k, self.v, causal=True)
        x = self.model.embed[np.array([new_id])]
        h = x + attn @ self.model.w_o
        ff = np.maximum(self.model.rmsnorm(h) @ self.model.w_ff1, 0.0)
        h = h + ff @ self.model.w_ff2
        return (self.model.rmsnorm(h) @ self.model.lm_head)[-1]


def generate_with_cache(prompt_ids: list[int], max_new_tokens: int, seed: int = 0) -> list[int]:
    cached = CachedToyLM(seed=seed)
    logits = cached.prefill(prompt_ids)
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        nxt = greedy(logits)
        ids.append(nxt)
        logits = cached.decode_one(nxt)
    return ids


def gqa_vs_mha_kv_gib() -> tuple[float, float]:
    gqa = kv_cache_gib(LLAMA31_70B, 8192, 1)
    pretend_mha = kv_cache_gib(
        ModelSpec(
            "70B pretend MHA",
            LLAMA31_70B.num_layers,
            LLAMA31_70B.num_q_heads,
            LLAMA31_70B.num_q_heads,
            LLAMA31_70B.hidden_size,
        ),
        8192,
        1,
    )
    return gqa, pretend_mha


if __name__ == "__main__":
    prefill, decode, ratio = prefill_decode_score_flops()
    print("teaching score FLOPs prefill", prefill)
    print("teaching score FLOPs decode", decode)
    print("ratio", ratio)
    print("70B 8K GQA GiB", kv_cache_gib(LLAMA31_70B, 8192, 1))
    print("7B-like 4x1024 GiB", kv_cache_gib(TEACHING_7B, 1024, 4))
    gqa, mha = gqa_vs_mha_kv_gib()
    print("70B 8K if 64 KV heads GiB", mha)
    naive = ToyLM(seed=0).generate([1, 3, 4], 3)
    cached = generate_with_cache([1, 3, 4], 3)
    print("naive", naive)
    print("cached", cached)
