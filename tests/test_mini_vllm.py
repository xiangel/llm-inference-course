from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mini-vllm"))

import numpy as np  # noqa: E402

from mini_vllm import Engine, RequestStatus  # noqa: E402
from mini_vllm.block_manager import BlockManager  # noqa: E402
from mini_vllm.engine import demo_tpot_steps  # noqa: E402
from mini_vllm.request import Request  # noqa: E402
from mini_vllm.scheduler import Scheduler  # noqa: E402


class BlockManagerTests(unittest.TestCase):
    def test_write_gather_roundtrip(self):
        bm = BlockManager(num_blocks=8, block_size=4, hidden=3)
        req = Request("r", [1, 2, 3, 4, 5], max_new_tokens=1)
        bm.ensure(req, 5)
        k = np.arange(15, dtype=np.float64).reshape(5, 3)
        v = k + 100
        bm.write_kv(req, k, v, 0)
        gk, gv = bm.gather_kv(req, 5)
        np.testing.assert_array_equal(gk, k)
        np.testing.assert_array_equal(gv, v)
        bm.free(req)
        self.assertEqual(len(bm.free_ids), 8)
        self.assertEqual(req.block_ids, [])


class EngineTests(unittest.TestCase):
    def test_generate_length(self):
        out = Engine(seed=0).generate([1, 3, 4], max_new_tokens=5)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(0 <= t < 64 for t in out))

    def test_stream_matches_generate(self):
        a = Engine(seed=1).generate([2, 5, 7], max_new_tokens=4)
        b = list(Engine(seed=1).generate_stream([2, 5, 7], max_new_tokens=4))
        self.assertEqual(a, b)

    def test_batch_continuous_finishes_all(self):
        outs = Engine(seed=2).generate_batch(
            [[1, 2], [3, 4, 5, 6], [7]],
            max_new_tokens=3,
        )
        self.assertEqual([len(x) for x in outs], [3, 3, 3])

    def test_chunked_prefill_uses_several_steps(self):
        engine = Engine(seed=0, block_size=4)
        engine.scheduler.max_prefill_tokens = 2
        stats = demo_tpot_steps(engine, [1, 2, 3, 4, 5], max_new_tokens=3)
        self.assertEqual(stats["prompt_tokens"], 5)
        self.assertGreaterEqual(stats["prefill_steps"], 3)
        self.assertEqual(stats["decode_steps"], 2)
        self.assertEqual(stats["output_tokens"], 3)

    def test_finished_request_releases_blocks(self):
        engine = Engine(seed=0, num_blocks=16, block_size=4)
        free_before = len(engine.block_manager.free_ids)
        engine.generate([1, 2, 3], max_new_tokens=2)
        self.assertEqual(len(engine.block_manager.free_ids), free_before)

    def test_scheduler_prefers_waiting_prefill(self):
        bm = BlockManager(num_blocks=32, block_size=4, hidden=8)
        sched = Scheduler(bm, max_prefill_tokens=8)
        running = Request("old", [1], max_new_tokens=4)
        running.status = RequestStatus.DECODE
        running.prompt_computed = 1
        running.append_token(2)
        bm.ensure(running, 2)
        sched.running.append(running)
        sched.add_request(Request("new", [3, 4, 5], max_new_tokens=2))
        batch = sched.schedule()
        self.assertIsNotNone(batch)
        self.assertEqual(batch.kind, "prefill")
        self.assertEqual(batch.requests[0].request_id, "new")


if __name__ == "__main__":
    unittest.main()
