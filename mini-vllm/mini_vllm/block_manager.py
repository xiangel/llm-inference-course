"""Paged KV pool. Teaching implementation, not vLLM BlockPool.

Physical storage is a fixed array of blocks. Each request holds a list of
block ids (the teaching Block Table). Attention still gathers pages back
into a dense ``[S, H]`` tensor — paging here is the allocator, not a kernel.
"""

from __future__ import annotations

import numpy as np

from mini_vllm.request import Request


class BlockManager:
    def __init__(self, num_blocks: int = 64, block_size: int = 4, hidden: int = 8):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.hidden = hidden
        self.free_ids: list[int] = list(range(num_blocks))
        self.k = np.zeros((num_blocks, block_size, hidden))
        self.v = np.zeros((num_blocks, block_size, hidden))

    def blocks_for(self, seq_len: int) -> int:
        if seq_len <= 0:
            return 0
        return (seq_len + self.block_size - 1) // self.block_size

    def allocate(self, n: int) -> list[int] | None:
        if n <= 0:
            return []
        if len(self.free_ids) < n:
            return None
        return [self.free_ids.pop() for _ in range(n)]

    def can_cover(self, req: Request, target_len: int) -> bool:
        extra = self.blocks_for(target_len) - len(req.block_ids)
        return extra <= 0 or extra <= len(self.free_ids)

    def ensure(self, req: Request, target_len: int) -> None:
        extra = self.blocks_for(target_len) - len(req.block_ids)
        if extra <= 0:
            return
        added = self.allocate(extra)
        if added is None:
            raise RuntimeError("out of KV blocks")
        req.block_ids.extend(added)

    def grow(self, block_ids: list[int], new_seq_len: int) -> list[int] | None:
        need = self.blocks_for(new_seq_len)
        extra = need - len(block_ids)
        if extra <= 0:
            return block_ids
        added = self.allocate(extra)
        if added is None:
            return None
        return block_ids + added

    def free_blocks(self, block_ids: list[int]) -> None:
        self.free_ids.extend(block_ids)
        for bid in block_ids:
            self.k[bid] = 0
            self.v[bid] = 0

    def free(self, req: Request) -> None:
        self.free_blocks(req.block_ids)
        req.block_ids = []

    def write(self, block_ids: list[int], index: int, k_vec: np.ndarray, v_vec: np.ndarray) -> None:
        block_i, offset = divmod(index, self.block_size)
        bid = block_ids[block_i]
        self.k[bid, offset] = k_vec
        self.v[bid, offset] = v_vec

    def write_kv(self, req: Request, k: np.ndarray, v: np.ndarray, start_index: int) -> None:
        for i in range(k.shape[0]):
            self.write(req.block_ids, start_index + i, k[i], v[i])

    def gather(self, block_ids: list[int], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        if seq_len == 0:
            hidden = self.hidden
            return np.zeros((0, hidden)), np.zeros((0, hidden))
        ks, vs = [], []
        for i in range(seq_len):
            block_i, offset = divmod(i, self.block_size)
            bid = block_ids[block_i]
            ks.append(self.k[bid, offset])
            vs.append(self.v[bid, offset])
        return np.stack(ks, axis=0), np.stack(vs, axis=0)

    def gather_kv(self, req: Request, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        return self.gather(req.block_ids, seq_len)
