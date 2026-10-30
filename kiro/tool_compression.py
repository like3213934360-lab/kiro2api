# -*- coding: utf-8 -*-

"""
Deterministic compression of large tool outputs (cache-safe).

Agent sessions spend most of their prompt budget on tool output: git diffs,
grep hits, directory listings and build logs. They are sent again on every
turn, so shrinking them once shrinks every later turn too.

Cache safety is the hard requirement. Kiro's prefix cache compares request
bytes, so a transform may only depend on the *content* of a message, never on
its position or age. Everything here is a pure function of the text:

    compress(compress(x)) == compress(x)

so a tool result transformed on the turn it appears keeps the exact same bytes
when it is re-sent as history on later turns.

What is compressed (only when it is genuinely large):

* unified diffs — keep headers and the first lines of each hunk
* grep output — cap the number of match lines
* directory listings — cap the number of entries
* build/test logs — collapse consecutive duplicate lines, cap total lines
* anything above the hard cap — keep head and tail

Error results are never touched, and anything already carrying a marker is
left alone (idempotency).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# Text shorter than this is never compressed: small outputs are cheap and the
# model may need them verbatim.
MIN_COMPRESS_BYTES = 800

# Do not replace duplicates smaller than this (short identical outputs such as
# "OK" or a version string are normal and cheap).
MIN_DEDUPE_BYTES = 500

# Last-resort ceiling for a single tool result.
HARD_CAP_BYTES = 50_000
HARD_CAP_HEAD = 24_000
HARD_CAP_TAIL = 16_000

# Per-format ceilings.
MAX_DIFF_HUNK_LINES = 120
MAX_GREP_LINES = 200
MAX_LISTING_LINES = 200
MAX_LOG_LINES = 400

MARKER = "\n[...kiro-gateway: output shortened automatically; re-run the tool for the full result ...]"
DEDUPE_MARKER = "[kiro-gateway: identical to an earlier tool output; omitted]"


@dataclass
class CompressionStats:
    """Byte accounting for one payload."""

    bytes_before: int = 0
    bytes_after: int = 0
    tool_bytes: int = 0
    hits: Dict[str, int] = field(default_factory=dict)

    @property
    def saved(self) -> int:
        """Bytes removed."""
        return max(0, self.bytes_before - self.bytes_after)

    def record(self, kind: str) -> None:
        """Count one transformed tool result."""
        self.hits[kind] = self.hits.get(kind, 0) + 1

    def format(self) -> Optional[str]:
        """
        Render a log line, or None when nothing changed.

        Returns:
            Compact summary such as ``saved 12.3 KB / 240.0 KB (5.1%) · diff=2``
        """
        if self.tool_bytes <= 0:
            return None
        kinds = ", ".join(f"{name}={count}" for name, count in sorted(self.hits.items()))
        tool_part = f"tool_output={self.tool_bytes / 1024:.1f} KB"
        if not self.saved:
            return tool_part + " · nothing to compress"
        pct = self.saved * 100.0 / self.bytes_before
        return (
            f"saved {self.saved / 1024:.1f} KB / {self.bytes_before / 1024:.1f} KB "
            f"({pct:.1f}%) · {tool_part}" + (f" · {kinds}" if kinds else "")
        )


def compress_tool_outputs(payload: Dict[str, Any]) -> CompressionStats:
    """
    Compress large tool results in a Kiro payload, in place.

    Applies to history and to the current message, because the current message
    becomes history on the next turn: transforming both keeps the bytes stable
    across turns, which is what keeps Kiro's prefix cache warm.

    Args:
        payload: Kiro payload produced by build_kiro_payload()

    Returns:
        Byte accounting for the transform
    """
    stats = CompressionStats()
    state = payload.get("conversationState") or {}
    entries: List[Dict[str, Any]] = list(state.get("history") or [])
    current = state.get("currentMessage")
    if isinstance(current, dict):
        entries.append(current)

    seen: Dict[str, bool] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        user_message = entry.get("userInputMessage") or {}
        context = user_message.get("userInputMessageContext") or {}
        tool_results = context.get("toolResults") or []
        if not isinstance(tool_results, list):
            continue

        for tool_result in tool_results:
            if not isinstance(tool_result, dict):
                continue
            if str(tool_result.get("status") or "").lower() == "error":
                continue

            for part in tool_result.get("content") or []:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if not isinstance(text, str) or not text:
                    continue
                stats.tool_bytes += len(text)
                if len(text) < MIN_COMPRESS_BYTES:
                    continue
                if MARKER in text or DEDUPE_MARKER in text:
                    continue  # already transformed on an earlier turn

                stats.bytes_before += len(text)
                compressed, kind = compress_text(text)

                digest = hashlib.sha256(compressed.encode("utf-8")).hexdigest()
                if len(compressed) >= MIN_DEDUPE_BYTES and digest in seen:
                    compressed = DEDUPE_MARKER
                    kind = "dedupe"
                else:
                    seen[digest] = True

                part["text"] = compressed
                stats.bytes_after += len(compressed)
                if kind:
                    stats.record(kind)

    line = stats.format()
    if line:
        logger.info(f"[ToolCompress] {line}")
    return stats


def compress_text(text: str) -> Tuple[str, Optional[str]]:
    """
    Compress one tool output using the first matching format rule.

    The result is only returned when it is actually smaller than the input, so
    calling this twice is a no-op and the bytes stay stable across turns.

    Args:
        text: Tool output

    Returns:
        Tuple of (possibly compressed text, rule name or None)
    """
    if MARKER in text or DEDUPE_MARKER in text:
        return text, None  # already transformed on an earlier turn

    if len(text) >= HARD_CAP_BYTES:
        candidate, kind = _hard_cap(text), "hard_cap"
    elif _looks_like_diff(text):
        candidate, kind = _compress_diff(text), "diff"
    elif _looks_like_grep(text):
        candidate, kind = _compress_lines(text, MAX_GREP_LINES, keep="head"), "grep"
    elif _looks_like_listing(text):
        candidate, kind = _compress_lines(text, MAX_LISTING_LINES, keep="head"), "listing"
    elif _looks_like_log(text):
        candidate, kind = _compress_log(text), "log"
    else:
        return text, None

    if len(candidate) >= len(text):
        return text, None
    return candidate, kind


def _hard_cap(text: str) -> str:
    """Keep the head and tail of a pathologically large output."""
    head = text[:HARD_CAP_HEAD]
    tail = text[-HARD_CAP_TAIL:]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[... {omitted} bytes omitted ...]{MARKER}\n{tail}"


def _looks_like_diff(text: str) -> bool:
    """True for unified/git diffs."""
    head = text[:4000]
    return ("diff --git " in head or head.startswith("diff --git ")) and "\n@@" in text


def _looks_like_grep(text: str) -> bool:
    """True when most lines look like ``path:line:content``."""
    lines = text.splitlines()
    if len(lines) < 20:
        return False
    hits = 0
    for line in lines[:200]:
        first, sep, rest = line.partition(":")
        if sep and first and rest[:1].isdigit():
            hits += 1
    return hits >= min(len(lines[:200]), 20) * 0.8


def _looks_like_listing(text: str) -> bool:
    """True when most lines look like file paths (no spaces, contains a slash)."""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 40:
        return False
    sample = lines[:200]
    hits = sum(1 for line in sample if "/" in line and " " not in line.strip())
    return hits >= len(sample) * 0.8


def _looks_like_log(text: str) -> bool:
    """True for multi-line output long enough to benefit from line capping."""
    return text.count("\n") >= 60


def _compress_lines(text: str, limit: int, keep: str = "head") -> str:
    """Cap the number of lines, keeping the head (or head and tail)."""
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    if keep == "head":
        kept = lines[:limit]
        omitted = len(lines) - limit
    else:
        head = limit // 2
        tail = limit - head
        return "\n".join(lines[:head]) + f"\n[... {len(lines) - limit} lines omitted ...]{MARKER}\n" + "\n".join(lines[-tail:])
    return "\n".join(kept) + f"\n[... {omitted} lines omitted ...]{MARKER}"


def _compress_diff(text: str) -> str:
    """
    Keep diff headers and the first lines of every hunk.

    Long hunks (generated files, lock files, huge edits) dominate the prompt
    while the header already tells the model what changed.
    """
    output: List[str] = []
    hunk_lines = 0
    omitted_in_hunk = 0

    for line in text.splitlines():
        if line.startswith("@@"):
            if omitted_in_hunk:
                output.append(f"[... {omitted_in_hunk} lines omitted ...]")
                omitted_in_hunk = 0
            hunk_lines = 0
            output.append(line)
            continue

        if line.startswith(("diff --git ", "index ", "--- ", "+++ ", "new file", "deleted file", "rename ")):
            output.append(line)
            continue

        if hunk_lines < MAX_DIFF_HUNK_LINES:
            output.append(line)
            hunk_lines += 1
        else:
            omitted_in_hunk += 1

    if omitted_in_hunk:
        output.append(f"[... {omitted_in_hunk} lines omitted ...]")

    if len(output) >= len(text.splitlines()):
        return text
    return "\n".join(output) + MARKER


def _compress_log(text: str) -> str:
    """Collapse consecutive duplicate lines, then cap the total."""
    lines = text.splitlines()
    deduped: List[str] = []
    previous = None
    repeats = 0
    for line in lines:
        if line == previous:
            repeats += 1
            continue
        if repeats:
            deduped.append(f"  ... repeated {repeats} time(s) ...")
            repeats = 0
        deduped.append(line)
        previous = line
    if repeats:
        deduped.append(f"  ... repeated {repeats} time(s) ...")

    if len(deduped) <= MAX_LOG_LINES:
        return "\n".join(deduped) if len(deduped) != len(lines) else text

    head = deduped[: MAX_LOG_LINES // 2]
    tail = deduped[-(MAX_LOG_LINES // 2):]
    omitted = len(deduped) - len(head) - len(tail)
    return (
        "\n".join(head)
        + f"\n[... {omitted} lines omitted ...]{MARKER}\n"
        + "\n".join(tail)
    )
