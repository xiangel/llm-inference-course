"""Run Prefill or Decode against the paged KV pool.

Teaching stand-in for a GPU model runner. It gathers pages into a dense
tensor, runs ``ToyLM``, and writes new K/V back. Not vLLM source.
"""

from __future__ import annotations

import numpy as np

from mini_vllm.block_manager import BlockManager
from mini_vllm.model import ToyLM
from mini_vllm.request import Request
from mini_vllm.scheduler import ScheduledBatch


class ModelRunner:
    def __init__(self, model: ToyLM, block_manager: BlockManager) -> None:
        self.model = model
        self.block_manager = block_manager

    def execute(self, batch: ScheduledBatch) -> np.ndarray:
        if batch.kind == "prefill":
            return self._prefill(batch.requests)
        return self._decode(batch.requests)

    def _prefill(self, requests: list[Request]) -> np.ndarray:
        logits_out: list[np.ndarray] = []
        for req in requests:
            start = req.prompt_computed
            end = start + req.prompt_chunk
            ids = np.asarray(req.prompt_ids[start:end], dtype=np.int64)
            x = self.model.embed(ids[None, :])
            past = None
            if start > 0:
                past_k, past_v = self.block_manager.gather_kv(req, start)
                past = (past_k[None], past_v[None])
            logits, new_k, new_v = self.model.forward(x, past_kv=past)
            self.block_manager.write_kv(req, new_k[0], new_v[0], start)
            logits_out.append(logits[0, -1])
        return np.stack(logits_out, axis=0)

    def _decode(self, requests: list[Request]) -> np.ndarray:
        logits_out: list[np.ndarray] = []
        for req in requests:
            last_id = req.token_ids[-1]
            x = self.model.embed(np.asarray([[last_id]], dtype=np.int64))
            written = req.kv_written
            past_k, past_v = self.block_manager.gather_kv(req, written)
            logits, new_k, new_v = self.model.forward(
                x, past_kv=(past_k[None], past_v[None])
            )
            self.block_manager.write_kv(req, new_k[0], new_v[0], written)
            logits_out.append(logits[0, -1])
        return np.stack(logits_out, axis=0)
