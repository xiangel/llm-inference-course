"""Mini-vLLM Engine: add → schedule → execute → sample → stream.

Teaching loop. Not vLLM source. Identity: ``mini_vllm.Engine``.
"""

from __future__ import annotations

from collections.abc import Iterator

from mini_vllm.block_manager import BlockManager
from mini_vllm.model import ToyLM
from mini_vllm.model_runner import ModelRunner
from mini_vllm.request import Request
from mini_vllm.sampler import greedy
from mini_vllm.scheduler import Scheduler


class Engine:
    def __init__(
        self,
        vocab_size: int = 64,
        hidden: int = 32,
        num_blocks: int = 64,
        block_size: int = 8,
        max_num_seqs: int = 8,
        seed: int = 0,
    ) -> None:
        self.model = ToyLM(vocab_size=vocab_size, hidden=hidden, seed=seed)
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
            hidden=hidden,
        )
        self.scheduler = Scheduler(
            self.block_manager,
            max_num_seqs=max_num_seqs,
        )
        self.runner = ModelRunner(self.model, self.block_manager)
        self._seq = 0

    def add_request(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int = 8,
        request_id: str | None = None,
    ) -> Request:
        self._seq += 1
        req = Request(
            request_id=request_id or f"req-{self._seq}",
            prompt_ids=list(prompt_ids),
            max_new_tokens=max_new_tokens,
        )
        self.scheduler.add_request(req)
        return req

    def step(self) -> list[Request]:
        """Run one schedule → execute → sample cycle.

        Returns requests that finished in this step.
        """
        batch = self.scheduler.schedule()
        if batch is None or not batch.requests:
            return []
        logits = self.runner.execute(batch)
        tokens = greedy(logits)
        if isinstance(tokens, int):
            tokens = [tokens]
        return self.scheduler.on_step_finished(batch, tokens)

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 8) -> list[int]:
        req = self.add_request(prompt_ids, max_new_tokens=max_new_tokens)
        while self.scheduler.has_unfinished():
            self.step()
        return list(req.output_ids)

    def generate_stream(
        self, prompt_ids: list[int], max_new_tokens: int = 8
    ) -> Iterator[int]:
        """Yield new tokens as they are sampled (after Prefill completes)."""
        req = self.add_request(prompt_ids, max_new_tokens=max_new_tokens)
        emitted = 0
        while self.scheduler.has_unfinished():
            self.step()
            while emitted < len(req.output_ids):
                yield req.output_ids[emitted]
                emitted += 1

    def generate_batch(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 8,
    ) -> list[list[int]]:
        reqs = [
            self.add_request(p, max_new_tokens=max_new_tokens, request_id=f"b-{i}")
            for i, p in enumerate(prompts)
        ]
        while self.scheduler.has_unfinished():
            self.step()
        return [list(r.output_ids) for r in reqs]


def demo_tpot_steps(engine: Engine, prompt_ids: list[int], max_new_tokens: int) -> dict[str, int]:
    """Count teaching steps: first output token ≈ TTFT boundary, rest ≈ TPOT steps."""
    req = engine.add_request(prompt_ids, max_new_tokens=max_new_tokens)
    prefill_steps = 0
    decode_steps = 0
    while engine.scheduler.has_unfinished():
        batch = engine.scheduler.schedule()
        if batch is None:
            break
        if batch.kind == "prefill":
            prefill_steps += 1
        else:
            decode_steps += 1
        logits = engine.runner.execute(batch)
        tokens = greedy(logits)
        if isinstance(tokens, int):
            tokens = [tokens]
        engine.scheduler.on_step_finished(batch, tokens)
    return {
        "prefill_steps": prefill_steps,
        "decode_steps": decode_steps,
        "output_tokens": len(req.output_ids),
        "prompt_tokens": len(prompt_ids),
    }


if __name__ == "__main__":
    engine = Engine(seed=0)
    print("single", engine.generate([1, 3, 4], max_new_tokens=4))
    engine = Engine(seed=0)
    print("batch", engine.generate_batch([[1, 3, 4], [2, 5]], max_new_tokens=3))
    engine = Engine(seed=0)
    print("steps", demo_tpot_steps(engine, [1, 3, 4, 6, 7], 4))
