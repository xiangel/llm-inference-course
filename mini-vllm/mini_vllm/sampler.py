"""Greedy sampler. Teaching stand-in, not vLLM's Sampler."""

from __future__ import annotations

import numpy as np


def greedy(logits: np.ndarray) -> int | list[int]:
    """Argmax over the last axis.

    1-D logits → one token id. 2-D [B, V] → a list of ids, one per row.
    """
    if logits.ndim == 1:
        return int(np.argmax(logits))
    return [int(x) for x in np.argmax(logits, axis=-1)]
