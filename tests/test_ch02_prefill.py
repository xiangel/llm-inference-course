from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "ch00"))
sys.path.insert(0, str(ROOT / "examples" / "ch01"))
sys.path.insert(0, str(ROOT / "examples" / "ch02"))

from generate_one_token import ToyLM  # noqa: E402
from kv_cache_memory import LLAMA31_70B, attention_scores_flops, kv_cache_gib  # noqa: E402
from prefill_decode import (  # noqa: E402
    generate_with_cache,
    gqa_vs_mha_kv_gib,
    prefill_decode_score_flops,
)


class PrefillDecodeTests(unittest.TestCase):
    def test_score_flops_ratio_equals_seq(self):
        prefill, decode, ratio = prefill_decode_score_flops(seq=1024, batch=4)
        self.assertEqual(prefill, attention_scores_flops(4, 32, 1024, 1024, 128))
        self.assertEqual(decode, attention_scores_flops(4, 32, 1, 1024, 128))
        self.assertEqual(ratio, 1024)

    def test_cached_matches_naive_recompute(self):
        prompt = [1, 3, 4]
        naive = ToyLM(seed=0).generate(prompt, 3)
        cached = generate_with_cache(prompt, 3, seed=0)
        self.assertEqual(cached, naive)

    def test_70b_gqa_is_2_5_and_mha_is_20(self):
        gqa, mha = gqa_vs_mha_kv_gib()
        self.assertAlmostEqual(gqa, 2.5)
        self.assertAlmostEqual(mha, 20.0)
        self.assertAlmostEqual(kv_cache_gib(LLAMA31_70B, 8192, 1), 2.5)


if __name__ == "__main__":
    unittest.main()
