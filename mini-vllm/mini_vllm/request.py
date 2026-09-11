"""One generation request and its teaching-engine status.

Not vLLM's Request. Status names are for the book, not a production state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    output_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING
    block_ids: list[int] = field(default_factory=list)
    prompt_computed: int = 0
    prompt_chunk: int = 0

    @property
    def seq_len(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def prompt_remaining(self) -> int:
        return len(self.prompt_ids) - self.prompt_computed

    # Alias used in early drafts / chapter prose.
    prompt_left = prompt_remaining

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

    @property
    def is_finished(self) -> bool:
        return len(self.output_ids) >= self.max_new_tokens

    @property
    def kv_written(self) -> int:
        """Tokens whose K/V are already in the paged pool.

        After Prefill of the full prompt, this equals ``prompt_computed``.
        The first sampled token is appended before its KV is written; each
        Decode step writes that pending token, then appends the next one.
        """
        pending_unwritten = 1 if self.output_ids else 0
        return self.prompt_computed + len(self.output_ids) - pending_unwritten

    def append_token(self, tok: int) -> None:
        self.output_ids.append(tok)
