# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Payload size guard for Kiro API requests.

The Kiro API rejects payloads exceeding ~615KB with a misleading
"Improperly formed request." (reason: null) error. This module provides:
- Pre-flight size checking
- Auto-trimming of oldest history entries to fit under the limit

Ported from sametakofficial's payload_guards.py, simplified.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class PayloadTrimStats:
    """Statistics from a payload trim operation."""
    original_bytes: int
    final_bytes: int
    original_entries: int
    final_entries: int
    trimmed: bool


def check_payload_size(payload: Dict[str, Any]) -> int:
    """Return the serialized byte size of the payload as UTF-8 JSON."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


# A screenshot costs roughly this many tokens on Claude models. The exact value
# depends on resolution, but the order of magnitude is what matters: counting the
# base64 text instead overstates one 288 KB screenshot as ~84,000 tokens.
IMAGE_TOKEN_ESTIMATE = 1_500

# Bytes of JSON per token for ordinary text. Measured against Kiro's own count on
# long sessions; see the module docstring in kiro/request_context.py.
BYTES_PER_TOKEN = 3.5


def _iter_images(payload: Any, _seen: Optional[set] = None):
    """
    Yield every image dict found in a Kiro or Anthropic payload.

    Images appear in two shapes:

    * Kiro: ``userInputMessage.images = [{"format": "jpeg", "source": {"bytes": ...}}]``
    * Anthropic: a content block ``{"type": "image", "source": {"data": ...}}``

    Both are detected so the same estimate works before and after conversion. An
    image list entry is never counted twice, even when it also carries a ``type``.

    Args:
        payload: Payload, or any nested value inside one
        _seen: Identity set used to avoid double counting during recursion

    Yields:
        Image dictionaries
    """
    if _seen is None:
        _seen = set()

    def _emit(image: Dict[str, Any]):
        marker = id(image)
        if marker not in _seen:
            _seen.add(marker)
            return image
        return None

    if isinstance(payload, dict):
        if payload.get("type") == "image":
            emitted = _emit(payload)
            if emitted is not None:
                yield emitted
        for key, value in payload.items():
            if key == "images" and isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        emitted = _emit(item)
                        if emitted is not None:
                            yield emitted
            else:
                yield from _iter_images(value, _seen)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_images(item, _seen)


def _image_base64(image: Dict[str, Any]) -> str:
    """
    Return the base64 payload of one image, whichever shape it uses.

    Args:
        image: Image dictionary in Kiro or Anthropic shape

    Returns:
        Base64 string, or "" when the image carries no data
    """
    source = image.get("source")
    if isinstance(source, dict):
        for key in ("bytes", "data"):
            value = source.get(key)
            if isinstance(value, str):
                return value
    for key in ("bytes", "data"):
        value = image.get(key)
        if isinstance(value, str):
            return value
    return ""


def payload_image_stats(payload: Any) -> Tuple[int, int]:
    """
    Count images and their base64 size in a payload.

    Args:
        payload: Kiro payload

    Returns:
        Tuple of (image count, total base64 characters)
    """
    count = 0
    total = 0
    for image in _iter_images(payload):
        count += 1
        total += len(_image_base64(image))
    return count, total


def estimate_payload_tokens(
    payload: Dict[str, Any],
    bytes_per_token: float = BYTES_PER_TOKEN,
    image_tokens: int = IMAGE_TOKEN_ESTIMATE,
) -> int:
    """
    Estimate prompt tokens without letting inline images dominate the number.

    Serialized-bytes / 3.5 is accurate for text and catastrophically wrong for
    base64: one 288 KB screenshot measures ~84,000 tokens instead of ~1,500, so a
    session with a handful of screenshots looked like a 1.4M-token conversation
    and was pruned and summarised on every turn. Image bytes are therefore
    excluded from the text estimate and each image is charged a fixed cost.

    Args:
        payload: Kiro payload about to be sent upstream
        bytes_per_token: Text density of the serialized payload
        image_tokens: Token cost charged per image

    Returns:
        Estimated prompt tokens
    """
    total_bytes = check_payload_size(payload)
    image_count, image_bytes = payload_image_stats(payload)
    text_bytes = max(0, total_bytes - image_bytes)
    return int(text_bytes / bytes_per_token) + image_count * image_tokens


def exceeds_input_window(payload_tokens: int, max_input_tokens: int, reserve_tokens: int = 1024) -> bool:
    """
    Decide whether a prompt is too large to send upstream.

    Used after compaction as a last line of defence. Returning Anthropic's
    standard "prompt is too long" error makes the client shrink the
    conversation it holds; sending an oversized prompt instead lets the upstream
    truncate the tail silently, which looks like the model forgot the task.

    Args:
        payload_tokens: Measured prompt size in tokens
        max_input_tokens: Model input window in tokens
        reserve_tokens: Headroom kept for the response and protocol overhead

    Returns:
        True when the prompt no longer fits
    """
    if max_input_tokens <= 0:
        return False
    return payload_tokens > max_input_tokens - reserve_tokens


def _strip_empty_tool_uses(history: list) -> None:
    """Remove empty toolUses arrays in-place (Kiro quirk)."""
    for entry in history:
        assistant = entry.get("assistantResponseMessage")
        if assistant and "toolUses" in assistant and assistant["toolUses"] == []:
            del assistant["toolUses"]


def _align_to_user_message(history: list) -> list:
    """Ensure history starts with a userInputMessage entry."""
    while history and "userInputMessage" not in history[0]:
        history.pop(0)
    return history


def _repair_orphaned_tool_results(history: list) -> None:
    """
    Remove orphaned toolResults that reference toolUseIds not present
    in the preceding assistant message. Preserve orphaned text content
    inline with a marker.
    """
    for i, entry in enumerate(history):
        user_msg = entry.get("userInputMessage")
        if not user_msg:
            continue

        ctx = user_msg.get("userInputMessageContext")
        if not ctx or "toolResults" not in ctx:
            continue

        # Collect toolUseIds from the preceding assistant message
        valid_ids = set()
        if i > 0:
            prev_assistant = history[i - 1].get("assistantResponseMessage")
            if prev_assistant:
                for tu in prev_assistant.get("toolUses", []):
                    tool_use_id = tu.get("toolUseId")
                    if tool_use_id:
                        valid_ids.add(tool_use_id)

        kept = []
        orphaned_text_parts = []
        for tr in ctx["toolResults"]:
            if tr.get("toolUseId") in valid_ids:
                kept.append(tr)
            else:
                # Preserve text content from orphaned results
                content = tr.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            orphaned_text_parts.append(part["text"])
                elif isinstance(content, str) and content:
                    orphaned_text_parts.append(content)

        if len(kept) != len(ctx["toolResults"]):
            if kept:
                ctx["toolResults"] = kept
            else:
                del ctx["toolResults"]
                if not ctx:
                    del user_msg["userInputMessageContext"]

            # Append orphaned text to user message content
            if orphaned_text_parts:
                marker = "\n[trimmed tool result] " + "; ".join(orphaned_text_parts)
                current_content = user_msg.get("content", "")
                user_msg["content"] = current_content + marker


def trim_payload_to_limit(payload: Dict[str, Any], max_bytes: int) -> PayloadTrimStats:
    """
    Trim oldest history entries so the serialized payload fits under max_bytes.

    Trims in user/assistant pairs (2 entries at a time), aligns start to
    userInputMessage, and repairs orphaned toolResults after trimming.
    """
    original_bytes = check_payload_size(payload)
    history = payload.get("conversationState", {}).get("history")

    if not history:
        return PayloadTrimStats(
            original_bytes=original_bytes,
            final_bytes=original_bytes,
            original_entries=0,
            final_entries=0,
            trimmed=False,
        )

    original_entries = len(history)

    # Strip empty toolUses before measuring
    _strip_empty_tool_uses(history)

    # Trim pairs from the beginning until under limit (keep at least 2 entries)
    while len(history) > 2 and check_payload_size(payload) > max_bytes:
        # Remove 2 entries (a user/assistant pair)
        history.pop(0)
        history.pop(0)

    # Align to userInputMessage boundary
    _align_to_user_message(history)

    # Repair orphaned tool results after trimming
    _repair_orphaned_tool_results(history)

    final_bytes = check_payload_size(payload)
    return PayloadTrimStats(
        original_bytes=original_bytes,
        final_bytes=final_bytes,
        original_entries=original_entries,
        final_entries=len(history),
        trimmed=original_entries != len(history),
    )
