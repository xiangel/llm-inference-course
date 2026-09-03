from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "ch00"))
sys.path.insert(0, str(ROOT / "examples" / "ch01"))

from kv_cache_memory import (  # noqa: E402
    GIB,
    LLAMA31_8B,
    LLAMA31_70B,
    TEACHING_7B,
    attention_av_flops,
    attention_out_proj_flops,
    attention_qkv_flops,
    attention_scores_flops,
    kv_cache_bytes,
    kv_cache_gib,
    weight_bytes_approx,
)
from attention import scaled_dot_product_attention, tiny_worked_example  # noqa: E402
from generate_one_token import ToyLM  # noqa: E402
from rope import inv_freq, tiny_worked_example as rope_example  # noqa: E402
from sampling import greedy, softmax, tiny_worked_example as sampling_example, top_k_mask, top_p_mask  # noqa: E402


class KvMemoryTests(unittest.TestCase):
    def test_llama31_70b_8k_is_exactly_2_5_gib(self):
        self.assertEqual(kv_cache_bytes(LLAMA31_70B, 8192, 1), 2_684_354_560)
        self.assertAlmostEqual(kv_cache_gib(LLAMA31_70B, 8192, 1), 2.5)

    def test_llama31_8b_8k_is_exactly_1_gib(self):
        self.assertEqual(kv_cache_bytes(LLAMA31_8B, 8192, 1), GIB)
        self.assertAlmostEqual(kv_cache_gib(LLAMA31_8B, 8192, 1), 1.0)

    def test_70b_128k_is_40_gib(self):
        self.assertAlmostEqual(kv_cache_gib(LLAMA31_70B, 131072, 1), 40.0)

    def test_teaching_batch4_seq1024(self):
        # 2 * 32 * 32 * 128 * 1024 * 4 * 2 = 2_147_483_648 B = 2 GiB
        self.assertEqual(kv_cache_bytes(TEACHING_7B, 1024, 4), 2 * GIB)

    def test_weight_approx(self):
        # 70e9 params * 2 bytes / 1024^3 ≈ 130.43 GiB; 140e9 bytes = 140 GB (10^9).
        self.assertAlmostEqual(weight_bytes_approx(70e9) / GIB, 130.385160446167, places=6)
        self.assertEqual(weight_bytes_approx(70e9), 140_000_000_000)

    def test_attention_flops_prefill(self):
        batch, seq, hidden, heads, head_dim = 4, 1024, 4096, 32, 128
        self.assertEqual(attention_qkv_flops(batch, seq, hidden), 412_316_860_416)
        self.assertEqual(
            attention_scores_flops(batch, heads, seq, seq, head_dim),
            34_359_738_368,
        )
        self.assertEqual(
            attention_av_flops(batch, heads, seq, seq, head_dim),
            34_359_738_368,
        )
        self.assertEqual(attention_out_proj_flops(batch, seq, hidden), 137_438_953_472)


class AttentionTests(unittest.TestCase):
    def test_causal_mask_hides_future(self):
        data = tiny_worked_example()
        weights = data["weights"]
        self.assertTrue((weights[0, 1:] == 0).all())
        self.assertTrue((weights[1, 2:] == 0).all())
        self.assertAlmostEqual(weights.sum(axis=-1).min(), 1.0, places=12)

    def test_decode_sees_all_cached_keys(self):
        import numpy as np

        q = np.array([[0.5, 0.5]])
        k = tiny_worked_example()["k"]
        v = tiny_worked_example()["v"]
        out, weights = scaled_dot_product_attention(q, k, v, causal=True)
        self.assertEqual(weights.shape, (1, 4))
        self.assertGreater(weights[0, 3], 0.0)
        self.assertEqual(out.shape, (1, 2))


class RopeTests(unittest.TestCase):
    def test_inv_freq(self):
        freq = inv_freq(4, 10000.0)
        self.assertAlmostEqual(freq[0], 1.0)
        self.assertAlmostEqual(freq[1], 0.01)

    def test_relative_inner_product_depends_on_offset(self):
        dots = rope_example()["dots"]
        # After RoPE, q_i · k_j equals q_{i+d} · k_{j+d} for the same offset.
        self.assertAlmostEqual(dots[0, 1], dots[1, 2], places=12)


class SamplingTests(unittest.TestCase):
    def test_greedy_picks_max(self):
        data = sampling_example()
        self.assertEqual(data["greedy"], 0)
        self.assertGreater(data["probs_t02"][0], data["probs_t1"][0])

    def test_top_k_keeps_two(self):
        import numpy as np

        logits = np.array([4.0, 2.0, 1.0, 0.0])
        masked = top_k_mask(logits, 2)
        self.assertTrue(np.isneginf(masked[2]))
        self.assertTrue(np.isneginf(masked[3]))

    def test_top_p_keeps_prefix(self):
        import numpy as np

        logits = np.array([4.0, 2.0, 1.0, 0.0])
        masked = top_p_mask(logits, 0.9)
        kept = ~np.isneginf(masked)
        self.assertTrue(kept[0])
        self.assertFalse(kept[3])


class GenerateTests(unittest.TestCase):
    def test_generate_grows_by_requested_tokens(self):
        model = ToyLM(seed=0)
        prompt = [1, 3, 4]
        out = model.generate(prompt, max_new_tokens=4)
        self.assertEqual(len(out), 7)
        self.assertEqual(out[:3], prompt)
        logits = model.next_token_logits(prompt)
        self.assertEqual(logits.shape, (8,))


if __name__ == "__main__":
    unittest.main()
