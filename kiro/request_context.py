# -*- coding: utf-8 -*-

"""
Per-request values that are not part of the API payloads.

The gateway measures the outgoing prompt in two ways: what Kiro reports through
``contextUsagePercentage`` (a percentage of the model window) and what the
serialized payload actually weighs. The second number is what upstream really
processes, and it is also what the *client* needs in order to decide when to
compact: Claude Code compares the usage it receives against the model window,
so reporting an understated prompt keeps its auto-compaction from ever firing.

Each measure answers a different question and they must not be conflated:

``payload_estimate_tokens`` is what the gateway actually sends upstream *after*
it has pruned old tool results and summarised older turns. It is used for cost
analysis and upstream diagnostics.

``client_prompt_tokens`` is how large the conversation is while it still lives
in the client. It is recorded *before* any shrinking happens, so it stays honest
even when the gateway rewrites the prompt. This is the number Claude Code must
see: it decides when to auto-compact by comparing received usage against the
model window, so reporting a shrunken payload keeps compaction from ever firing
and the session grows until the gateway has to rewrite history on every turn.

Both values are carried in context variables instead of being threaded through
every streaming function, because the streaming generators run in the same
asyncio context as the request handler.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

payload_estimate_tokens: ContextVar[Optional[int]] = ContextVar(
    "payload_estimate_tokens", default=None
)

client_prompt_tokens: ContextVar[Optional[int]] = ContextVar(
    "client_prompt_tokens", default=None
)


def set_payload_estimate(tokens: Optional[int]) -> None:
    """
    Record the outgoing prompt size for the current request.

    Args:
        tokens: Estimated prompt tokens, or None to clear it
    """
    payload_estimate_tokens.set(tokens)


def get_payload_estimate() -> Optional[int]:
    """
    Read the outgoing prompt size for the current request.

    Returns:
        Estimated prompt tokens, or None when nothing was measured
    """
    return payload_estimate_tokens.get()


def set_client_prompt_tokens(tokens: Optional[int]) -> None:
    """
    Record the conversation size the client currently holds.

    Measurements arrive from several layers (converter pruning, route-level
    compaction, the route's own tokenizer pass) and a later one is always *after*
    some shrinking. The value is therefore monotonic within a request: a smaller
    later measurement means "the gateway shrank the prompt", which must never
    lower what the client is told. Recording the maximum makes the result
    independent of call order, which is exactly the class of bug this guards
    against.

    Args:
        tokens: Estimated prompt tokens, or None to clear the value
    """
    if tokens is None:
        client_prompt_tokens.set(None)
        return
    measured = int(tokens)
    if measured <= 0:
        return
    current = client_prompt_tokens.get()
    if current is None or measured > current:
        client_prompt_tokens.set(measured)


def get_client_prompt_tokens() -> Optional[int]:
    """
    Read the largest conversation size seen for the current request.

    Returns:
        Estimated prompt tokens for the untouched conversation, or None when
        nothing was measured
    """
    return client_prompt_tokens.get()
