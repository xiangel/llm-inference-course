from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "ch04"))

from static_vs_continuous import (  # noqa: E402
    chunk_plan,
    continuous_slot_steps,
    static_pad_tokens,
    static_slot_steps,
    static_wait_steps,
)


class StaticVsContinuousTests(unittest.TestCase):
    def test_static_pad_tokens(self):
        self.assertEqual(static_pad_tokens([8, 32, 4]), 24 + 0 + 28)

    def test_short_request_waits_for_longest(self):
        self.assertEqual(static_wait_steps([4, 16, 8]), [12, 0, 8])

    def test_continuous_saves_slot_steps(self):
        decode = [4, 16, 8]
        self.assertEqual(static_slot_steps(decode), 48)
        self.assertEqual(continuous_slot_steps(decode), 28)

    def test_chunk_plan(self):
        self.assertEqual(chunk_plan(20, 8), [8, 8, 4])
        self.assertEqual(chunk_plan(0, 8), [])


if __name__ == "__main__":
    unittest.main()
