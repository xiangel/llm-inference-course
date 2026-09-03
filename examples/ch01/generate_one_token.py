"""Minimal autoregressive loop: a 1-layer toy language model in NumPy.

The network is intentionally tiny so a reader can print every tensor.
It is a teaching model, not a Transformer checkpoint loader.
"""

from __future__ import annotations

import numpy as np

from attention import scaled_dot_product_attention
from rope import apply_rope_llama, expand_cos_sin_for_llama, rotary_angles
from sampling import greedy, softmax


class ToyLM:
    def __init__(self, vocab_size: int = 8, hidden: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.embed = rng.normal(scale=0.3, size=(vocab_size, hidden))
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

    def forward(self, token_ids: list[int]) -> np.ndarray:
        x = self.embed[np.array(token_ids)]
        x_norm = self.rmsnorm(x)
        q, k, v = x_norm @ self.w_q, x_norm @ self.w_k, x_norm @ self.w_v
        angles = rotary_angles(len(token_ids), self.hidden)
        cos, sin = expand_cos_sin_for_llama(angles)
        q = apply_rope_llama(q, cos, sin)
        k = apply_rope_llama(k, cos, sin)
        attn, _ = scaled_dot_product_attention(q, k, v, causal=True)
        h = x + attn @ self.w_o
        ff = self.rmsnorm(h) @ self.w_ff1
        ff = np.maximum(ff, 0.0)  # ReLU stand-in; Llama uses SwiGLU
        h = h + ff @ self.w_ff2
        logits = self.rmsnorm(h) @ self.lm_head
        return logits

    def next_token_logits(self, token_ids: list[int]) -> np.ndarray:
        return self.forward(token_ids)[-1]

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            logits = self.next_token_logits(ids)
            ids.append(greedy(logits))
        return ids


def run_demo() -> dict[str, object]:
    model = ToyLM()
    prompt = [1, 3, 4]
    logits = model.next_token_logits(prompt)
    generated = model.generate(prompt, max_new_tokens=4)
    return {
        "prompt": prompt,
        "next_logits": logits,
        "next_probs": softmax(logits),
        "next_id": greedy(logits),
        "generated": generated,
    }


if __name__ == "__main__":
    demo = run_demo()
    np.set_printoptions(precision=4, suppress=True)
    for key, value in demo.items():
        print(f"{key}: {value}")
