# -*- coding: utf-8 -*-

"""
Automatic, client-transparent conversation compaction.

This is the gateway-side equivalent of Anthropic's server-side compaction:
when a conversation is still too large after old tool results have been
cleared, the gateway asks Kiro for a dense summary of the older turns and
replaces them with that summary. The client keeps its full history; only the
outbound Kiro payload is compacted, so the next request looks normal to
Claude Code / any Anthropic client.

Design choices:
- keep the most recent messages untouched (active working set);
- summarize only older messages, preserving user goals, decisions, files,
  errors and pending tasks;
- cache the summary per conversation for a short TTL so a 30-tool-call loop
  does not summarize on every single turn;
- on any failure, fall back silently to the original request.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import (
    AUTO_COMPACT_FALLBACK_TOKENS,
    AUTO_COMPACT_RATIO,
    model_max_input_tokens,
)

from kiro.converters_anthropic import (
    convert_anthropic_content_to_text,
    extract_tool_results_from_anthropic_content,
    extract_tool_uses_from_anthropic_content,
)
from kiro.converters_core import ThinkingConfig, UnifiedMessage, build_kiro_payload
from kiro.http_client import KiroHttpClient
from kiro.streaming_core import collect_stream_to_result
from kiro.tokenizer import estimate_request_tokens


# Auto-compaction is a last resort and must stay well above the pruning
# threshold: it costs an upstream summarisation call AND replaces the prefix
# (dropping the prompt cache). 0 means "follow the model window".
AUTO_COMPACT_TOKENS = int(os.getenv("AUTO_COMPACT_TOKENS", "0") or 0)
AUTO_COMPACT_KEEP_MESSAGES = 8


def auto_compact_threshold(model_id: Optional[str]) -> int:
    """
    Token estimate at which auto-compaction is allowed to run.

    Args:
        model_id: Model id for the request (may carry an effort suffix)

    Returns:
        Threshold in estimated tokens: an explicit ``AUTO_COMPACT_TOKENS``
        override when set, otherwise 70% of the model's input window
    """
    if AUTO_COMPACT_TOKENS > 0:
        return AUTO_COMPACT_TOKENS
    limit = model_max_input_tokens(model_id)
    if not limit:
        return AUTO_COMPACT_FALLBACK_TOKENS
    return max(1, int(limit * AUTO_COMPACT_RATIO))


SUMMARY_MODEL_ID = "claude-haiku-5.5"
SUMMARY_MAX_CHARS = 12_000
SUMMARY_CACHE_TTL_SECONDS = 600
SUMMARY_CACHE_MAX_ENTRIES = 64

_SUMMARY_INSTRUCTION = """You are compacting a coding-agent conversation.
Write a dense handoff summary for the next assistant turn.

Preserve:
- the user's goal, constraints and acceptance criteria;
- decisions already made and why;
- files/paths, symbols, commands, error messages and identifiers;
- what has been completed, what failed, and what is still pending.

Also keep an explicit "already verified" checklist, because the next turn must
not repeat work:
- every command that was run and its outcome (exit code, pass/fail, test counts);
- every fact confirmed by reading code, docs or the device;
- the current state of the working tree (files changed, created, deleted,
  staged, uncommitted).

Omit:
- raw tool output, repeated file contents and chit-chat;
- anything that is no longer needed to continue the task.

Return only the summary, in plain text, at most 1500 words."""


# key -> (created_at, summary, marker of the last message the summary covers)
_CACHE: Dict[str, Tuple[float, str, str]] = {}


def _message_marker(message: Any) -> str:
    """
    Stable fingerprint of one message, used to locate the summary boundary.

    Args:
        message: Anthropic message object or dict

    Returns:
        Short hex digest of role plus content
    """
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role is None and isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
    encoded = json.dumps(
        [role, content], ensure_ascii=False, default=str, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _cache_get(key: str) -> Optional[Tuple[str, str]]:
    """
    Read a cached summary.

    Args:
        key: Conversation id

    Returns:
        Tuple of (summary, covered_marker), or None when absent/expired
    """
    entry = _CACHE.get(key)
    if not entry:
        return None
    created, summary, covered = entry
    if time.time() - created > SUMMARY_CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return summary, covered


def _cache_set(key: str, summary: str, covered: str) -> None:
    if len(_CACHE) >= SUMMARY_CACHE_MAX_ENTRIES:
        oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
        _CACHE.pop(oldest, None)
    _CACHE[key] = (time.time(), summary, covered)


def _summarizer_excerpt(text: str, limit: int) -> str:
    """
    Shorten text for the summarizer while keeping its conclusion.

    The summarizer is asked to record command outcomes and exit codes, which sit
    at the end of a tool result. A head-only excerpt therefore removes exactly
    the information the summary is supposed to preserve, and the next turn has no
    record that the check already passed.

    Args:
        text: Original text
        limit: Maximum characters to keep

    Returns:
        The original text when short, otherwise its start and end joined by a gap
        marker
    """
    if len(text) <= limit:
        return text
    head = max(1, int(limit * 2 / 3))
    tail = max(1, limit - head)
    omitted = len(text) - head - tail
    return f"{text[:head]}\n[... {omitted} chars omitted ...]\n{text[-tail:]}"


def _render_transcript(messages: List[Any], tool_result_max_chars: int = 1500) -> str:
    """Render Anthropic messages as a plain transcript for the summarizer."""
    parts: List[str] = []
    for message in messages:
        role = getattr(message, "role", "user")
        content = getattr(message, "content", "")
        text = convert_anthropic_content_to_text(content).strip()
        if text:
            parts.append(f"{role.upper()}:\n{text}")

        for call in extract_tool_uses_from_anthropic_content(content):
            function = call.get("function", {})
            name = function.get("name", "tool")
            arguments = function.get("arguments", {})
            parts.append(
                f"ASSISTANT TOOL CALL {name}:\n{_summarizer_excerpt(str(arguments), 800)}"
            )

        for result in extract_tool_results_from_anthropic_content(content):
            result_text = _summarizer_excerpt(
                str(result.get("content", "")), tool_result_max_chars
            )
            status = " [FAILED]" if result.get("is_error") else ""
            parts.append(
                f"TOOL RESULT {result.get('tool_use_id', '')}{status}:\n{result_text}"
            )

    return "\n\n".join(parts)


def _split_old_and_recent(
    messages: List[Any], keep: int = AUTO_COMPACT_KEEP_MESSAGES
) -> Optional[Tuple[List[Any], List[Any]]]:
    """Split history so the recent working set stays verbatim."""
    if len(messages) <= keep + 2:
        return None

    cut = len(messages) - keep
    # Prefer a plain user message as the boundary so the recent block does not
    # start with an orphaned tool result.
    while cut < len(messages) - 1:
        candidate = messages[cut]
        if getattr(candidate, "role", "") == "user":
            text = convert_anthropic_content_to_text(getattr(candidate, "content", "")).strip()
            if text:
                break
        cut += 1

    old = messages[:cut]
    recent = messages[cut:]
    if len(old) < 2 or not recent:
        return None
    return old, recent


def _append_summary_to_system(system: Any, summary: str) -> Any:
    block = f"\n\n[Earlier conversation summary]\n{summary.strip()}"
    if isinstance(system, list):
        return list(system) + [{"type": "text", "text": block}]
    if isinstance(system, str):
        return system + block
    return block


async def _summarize(
    old_messages: List[Any],
    auth_manager: Any,
    conversation_id: str,
    profile_arn: str,
    shared_client: Any,
    fallback_model_id: str,
) -> Optional[str]:
    """Call Kiro once and return the summary text, or None on failure."""
    transcript = _render_transcript(old_messages)
    if not transcript:
        return None

    prompt = f"{_SUMMARY_INSTRUCTION}\n\n--- CONVERSATION ---\n{transcript}"
    models_to_try = [SUMMARY_MODEL_ID, fallback_model_id]

    for model_id in models_to_try:
        client = None
        try:
            payload = build_kiro_payload(
                messages=[UnifiedMessage(role="user", content=prompt)],
                system_prompt="",
                model_id=model_id,
                tools=None,
                conversation_id=f"{conversation_id}-summary-{hashlib.sha256(transcript.encode()).hexdigest()[:8]}",
                profile_arn=profile_arn,
                thinking_config=ThinkingConfig(enabled=False),
                effort="low",
            ).payload

            client = KiroHttpClient(auth_manager, shared_client=shared_client)
            response = await client.request_with_retry(
                "POST",
                f"{auth_manager.api_host}/generateAssistantResponse",
                payload,
                stream=True,
            )
            try:
                if response.status_code != 200:
                    continue
                result = await collect_stream_to_result(
                    response,
                    first_token_timeout=90.0,
                    enable_thinking_parser=False,
                )
                summary = (result.content or "").strip()
                if summary:
                    return summary[:SUMMARY_MAX_CHARS]
            finally:
                await response.aclose()
        except Exception as exc:
            logger.warning(f"[AutoCompact] summary via {model_id} failed: {exc}")
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass

    return None


async def maybe_compact_request(
    request_data: Any,
    auth_manager: Any,
    conversation_id: str,
    profile_arn: str,
    shared_client: Any,
    estimated_tokens: Optional[int] = None,
    threshold_tokens: Optional[int] = None,
) -> Optional[Any]:
    """
    Return a compacted copy of the request, or None when no action is needed.

    The original request is never mutated. On any failure the caller should
    keep using the original request.
    """
    messages = list(getattr(request_data, "messages", []) or [])
    if not messages:
        return None

    if estimated_tokens is None:
        tools = getattr(request_data, "tools", None)
        system = getattr(request_data, "system", None)
        estimated_tokens = estimate_request_tokens(
            messages=[m.model_dump() for m in messages],
            tools=[t.model_dump() for t in tools] if tools else None,
            system_prompt=system,
            apply_claude_correction=False,
        )["total_tokens"]

    if threshold_tokens is None:
        threshold_tokens = auto_compact_threshold(getattr(request_data, "model", None))

    if estimated_tokens <= threshold_tokens:
        return None

    split = _split_old_and_recent(messages)
    if not split:
        return None
    old_messages, recent_messages = split

    summary: Optional[str] = None
    carried: List[Any] = []
    cached = _cache_get(conversation_id)
    if cached:
        cached_summary, covered = cached
        # The cached summary covers everything up to `covered`. Messages that
        # arrived after it are new "old" messages: they are no longer in the
        # recent window, and the summary cannot describe them. Dropping them
        # silently hid up to ten minutes of work from the model, which then
        # repeated it. Carry them verbatim between the summary and the recent
        # block instead of paying for another summarisation call.
        boundary = None
        for index in range(len(old_messages) - 1, -1, -1):
            if _message_marker(old_messages[index]) == covered:
                boundary = index
                break
        if boundary is not None:
            summary = cached_summary
            carried = old_messages[boundary + 1 :]
            if carried:
                logger.debug(
                    f"[AutoCompact] summary covers {boundary + 1}/{len(old_messages)} "
                    f"old message(s); carrying {len(carried)} unsummarised"
                )
        else:
            logger.debug(
                "[AutoCompact] cached summary no longer aligns with the history; "
                "summarising again"
            )

    if not summary:
        summary = await _summarize(
            old_messages,
            auth_manager,
            conversation_id,
            profile_arn,
            shared_client,
            fallback_model_id=getattr(request_data, "model", "claude-sonnet-5"),
        )
        if not summary:
            return None
        _cache_set(conversation_id, summary, _message_marker(old_messages[-1]))
        logger.info(
            f"[AutoCompact] summarized {len(old_messages)} old message(s) "
            f"for conversation {conversation_id[:16]}..."
        )

    compacted = request_data.model_copy(deep=True)
    compacted.messages = list(carried) + recent_messages
    compacted.system = _append_summary_to_system(
        getattr(request_data, "system", None), summary
    )
    return compacted
