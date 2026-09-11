"""Teaching scheduler: FCFS + continuous batching + chunked prefill.

Not vLLM's Scheduler. Prefill and Decode are separate batches so the two
phases stay visible. Production engines may mix them in one kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mini_vllm.block_manager import BlockManager
from mini_vllm.request import Request, RequestStatus


@dataclass
class ScheduledBatch:
    kind: str  # "prefill" | "decode"
    requests: list[Request] = field(default_factory=list)
    token_budget: int = 0


class Scheduler:
    def __init__(
        self,
        block_manager: BlockManager,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 256,
        max_prefill_tokens: int = 64,
    ) -> None:
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_prefill_tokens = max_prefill_tokens
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def in_flight(self) -> int:
        ids = {r.request_id for r in self.running}
        ids.update(r.request_id for r in self.waiting if r.block_ids)
        return len(ids)

    def schedule(self) -> ScheduledBatch | None:
        if self.waiting:
            batch = self._schedule_prefill()
            if batch.requests:
                return batch
        if self.running:
            return self._schedule_decode()
        return None

    def _schedule_prefill(self) -> ScheduledBatch:
        picked: list[Request] = []
        leftover: list[Request] = []
        tokens_left = min(self.max_num_batched_tokens, self.max_prefill_tokens)

        for req in self.waiting:
            if tokens_left <= 0 or req.prompt_remaining <= 0:
                leftover.append(req)
                continue
            take = min(req.prompt_remaining, tokens_left)
            if not req.block_ids and self.in_flight() + len(picked) >= self.max_num_seqs:
                leftover.append(req)
                continue
            target_len = req.prompt_computed + take
            if not self.block_manager.can_cover(req, target_len):
                leftover.append(req)
                continue
            self.block_manager.ensure(req, target_len)
            req.prompt_chunk = take
            req.status = RequestStatus.PREFILL
            picked.append(req)
            tokens_left -= take

        self.waiting = leftover
        return ScheduledBatch(
            kind="prefill",
            requests=picked,
            token_budget=sum(r.prompt_chunk for r in picked),
        )

    def _schedule_decode(self) -> ScheduledBatch:
        picked: list[Request] = []
        leftover: list[Request] = []
        for req in self.running:
            if req.status == RequestStatus.FINISHED:
                continue
            target_len = req.prompt_computed + len(req.output_ids)
            if not self.block_manager.can_cover(req, target_len):
                leftover.append(req)
                continue
            self.block_manager.ensure(req, target_len)
            req.status = RequestStatus.DECODE
            picked.append(req)
        self.running = leftover + picked
        return ScheduledBatch(kind="decode", requests=picked, token_budget=len(picked))

    def on_step_finished(self, batch: ScheduledBatch, new_tokens: list[int]) -> list[Request]:
        finished: list[Request] = []
        for req, tok in zip(batch.requests, new_tokens, strict=True):
            if batch.kind == "prefill":
                req.prompt_computed += req.prompt_chunk
                req.prompt_chunk = 0
                if req.prompt_remaining > 0:
                    req.status = RequestStatus.WAITING
                    if req not in self.waiting:
                        self.waiting.append(req)
                    continue
                req.append_token(tok)
                if req.is_finished:
                    self._finish(req)
                    finished.append(req)
                else:
                    req.status = RequestStatus.DECODE
                    if req not in self.running:
                        self.running.append(req)
            else:
                req.append_token(tok)
                if req.is_finished:
                    self._finish(req)
                    finished.append(req)
        return finished

    def _finish(self, req: Request) -> None:
        req.status = RequestStatus.FINISHED
        self.block_manager.free(req)
        if req in self.running:
            self.running.remove(req)
        if req in self.waiting:
            self.waiting.remove(req)
