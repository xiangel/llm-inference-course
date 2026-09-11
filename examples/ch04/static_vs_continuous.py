"""Static batch waste vs iteration-level (continuous) batching.

Teaching counters, not a GPU scheduler. Mini-vLLM implements the continuous
path; this file only makes the wasted-slot arithmetic visible.
"""

from __future__ import annotations


def static_pad_tokens(lengths: list[int]) -> int:
    """Tokens of padding if every sequence is padded to the max length."""
    if not lengths:
        return 0
    longest = max(lengths)
    return sum(longest - length for length in lengths)


def static_wait_steps(decode_lens: list[int]) -> list[int]:
    """How many extra Decode steps a short request waits in a static batch."""
    if not decode_lens:
        return []
    longest = max(decode_lens)
    return [longest - length for length in decode_lens]


def continuous_slot_steps(decode_lens: list[int]) -> int:
    """Occupied slot-steps if a request leaves as soon as it finishes."""
    return sum(decode_lens)


def static_slot_steps(decode_lens: list[int]) -> int:
    """Occupied slot-steps if the whole batch waits for the longest request."""
    if not decode_lens:
        return 0
    return max(decode_lens) * len(decode_lens)


def chunk_plan(prompt_len: int, max_prefill_tokens: int) -> list[int]:
    """How a long prompt is split under a token budget."""
    if prompt_len <= 0:
        return []
    sizes = []
    left = prompt_len
    while left > 0:
        take = min(left, max_prefill_tokens)
        sizes.append(take)
        left -= take
    return sizes


def print_worked_examples() -> None:
    decode = [4, 16, 8]
    print("static pad tokens (prompts 8, 32, 4)", static_pad_tokens([8, 32, 4]))
    print("static wait steps", static_wait_steps(decode))
    print("static slot-steps", static_slot_steps(decode))
    print("continuous slot-steps", continuous_slot_steps(decode))
    print("chunk plan 20 / 8", chunk_plan(20, 8))


if __name__ == "__main__":
    print_worked_examples()
