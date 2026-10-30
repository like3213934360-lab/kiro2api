# -*- coding: utf-8 -*-

"""
Deterministic compression of large tool outputs (cache-safe).

Agent sessions spend most of their prompt budget on tool output: git diffs,
grep hits, directory trees and build logs. They are sent again on every turn,
so shrinking them once shrinks every later turn too.

Cache safety is the hard requirement. Kiro's prefix cache compares request
bytes, so a transform may only depend on the *content* of a message, never on
its position or age. Everything here is a pure function of the text:

    compress(compress(x)) == compress(x)

so a tool result transformed on the turn it appears keeps the exact same bytes
when it is re-sent as history on later turns.

Rules, in order (first match wins):

* anything above the hard cap — keep head and tail
* unified diffs — keep headers and the first lines of each hunk
* ``git status`` — collapse to counts plus the first files per section
* ``git log`` — keep the newest commits, cap the list
* ``tree`` — drop the summary line, cap the listing
* ``grep`` — at most 10 matches per file, with a match count header
* ``find``/``ls`` listings — cap the number of entries
* build/test logs — collapse repeated lines, then cap
* any other long text (file reads) — keep head and tail, only above 2000 lines

Error results are never touched, and anything already carrying a marker is left
alone (idempotency).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# Text shorter than this is never compressed: small outputs are cheap and the
# model may need them verbatim.
MIN_COMPRESS_BYTES = 800

# Do not replace duplicates smaller than this (short identical outputs such as
# "OK" or a version string are normal and cheap).
MIN_DEDUPE_BYTES = 500

# Last-resort ceiling for a single tool result. Only pathological outputs (a
# whole generated file, a giant log dump) should ever reach this: truncating
# ordinary file reads makes the model re-read them, which costs far more than
# the tokens saved.
HARD_CAP_BYTES = 200_000
HARD_CAP_HEAD = 100_000
HARD_CAP_TAIL = 60_000

# Per-format ceilings (mirrors 9router's RTK defaults, raised where the old
# value was measured to cause a re-run).
#
# A truncated answer is far more expensive than the tokens it saves: the model
# cannot conclude anything from a partial list, so it runs the same command
# again — a full round trip plus the credits for re-sending the whole prompt.
# Measured on real sessions: 23% of Bash calls were byte-identical repeats, and
# `git status` alone ran 20 times because the changed-file list was capped at 10.
MAX_DIFF_HUNK_LINES = 200
MAX_GIT_LOG_LINES = 200
GREP_PER_FILE_MAX = 100
MAX_GREP_LINES = 1000
MAX_LISTING_LINES = 200
MAX_TREE_LINES = 200
MAX_LOG_LINES = 400
# Failure lines rescued from the omitted middle of a long build/test log.
MAX_LOG_KEPT_ERRORS = 60

# Generic long-text fallback (file reads and anything unclassified). Kept very
# high on purpose: a truncated read is worse than a large prompt, because the
# model answers by reading the file again (measured: 100 tool calls in one turn
# when reads were cut at 250 lines).
SMART_TRUNCATE_MIN_LINES = 2000
SMART_TRUNCATE_HEAD = 1200
SMART_TRUNCATE_TAIL = 600

# A short status is cheap (about 40 bytes per file) and the model needs the whole
# list to decide what to do; capping it at 10 cost a full re-run of `git status`
# for every repository with more changes than that.
STATUS_MAX_FILES = 100

MARKER = "\n[...kiro-gateway: output shortened automatically; re-run the tool for the full result ...]"
DEDUPE_MARKER = "[kiro-gateway: identical to an earlier tool output; omitted]"

_GREP_LINE_RE = re.compile(r"^([^:\n]+):(\d+):(.*)$")
# A line that opens with a date or a clock time is log output, never a grep hit.
_TIMESTAMP_LINE_RE = re.compile(
    r"^\s*\[?\d{2,4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|^\s*\[?\d{1,2}:\d{2}:\d{2}"
)
_TREE_CHARS = ("├", "└", "│", "|--", "`--")
_LOG_HINT_RE = re.compile(
    r"^\s*(\[?\d{2}:\d{2}:\d{2}|\[?\d{4}-\d{2}-\d{2}|INFO|WARN|ERROR|DEBUG|TRACE|"
    r"Compiling|Finished|Running|test .* \.\.\.|PASS|FAIL|OK\b)",
    re.IGNORECASE,
)
# Lines that carry a verdict the model must not lose when a log is capped.
_LOG_ERROR_RE = re.compile(
    r"(error|failed|failure|exception|traceback|fatal|panic|assert|"
    r"\bFAILED\b|\bERROR\b|✗|\bnot found\b|non-zero exit)",
    re.IGNORECASE,
)
_STATUS_FILE_PREFIXES = (
    "modified:",
    "new file:",
    "deleted:",
    "renamed:",
    "copied:",
    "typechange:",
    "both modified:",
    "both added:",
)


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
        Render a log line, or None when there was no tool output at all.

        Returns:
            Compact summary such as
            ``saved 12.3 KB / 240.0 KB (5.1%) · tool_output=512.0 KB · diff=2``
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
    elif _looks_like_git_status(text):
        candidate, kind = _compress_git_status(text), "git_status"
    elif _looks_like_git_log(text):
        candidate, kind = _compress_git_log(text), "git_log"
    elif _looks_like_tree(text):
        candidate, kind = _compress_tree(text), "tree"
    elif _looks_like_grep(text):
        candidate, kind = _compress_grep(text), "grep"
    elif _looks_like_listing(text):
        candidate, kind = _compress_lines(text, MAX_LISTING_LINES), "listing"
    elif _looks_like_log(text):
        candidate, kind = _compress_log(text), "log"
    elif len(text.splitlines()) >= SMART_TRUNCATE_MIN_LINES:
        candidate, kind = _smart_truncate(text), "truncate"
    else:
        return text, None

    if len(candidate) >= len(text):
        return text, None
    return candidate, kind


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _looks_like_diff(text: str) -> bool:
    """True for unified/git diffs."""
    head = text[:4000]
    return ("diff --git " in head or head.startswith("diff --git ")) and "\n@@" in text


def _looks_like_git_status(text: str) -> bool:
    """
    True for long-form or porcelain ``git status`` output.

    A single "On branch" line is not enough: source files and docs quote it, and
    treating such a file as status output rewrote 300 lines of code into two
    lines of section counts, forcing the model to read the file again.
    """
    head = text[:4000]
    lines = head.splitlines()
    prefixed = sum(1 for line in lines if line.strip().startswith(_STATUS_FILE_PREFIXES))
    if prefixed >= 3:
        return True
    if "Changes to be committed" in head or "Changes not staged for commit" in head:
        return True
    has_branch = any(line.startswith("On branch ") for line in lines)
    return has_branch and ("Untracked files:" in head or "nothing to commit" in head)


def _looks_like_git_log(text: str) -> bool:
    """True when the text is a ``git log`` listing."""
    head = text[:4000]
    commits = sum(1 for line in head.splitlines() if line.startswith("commit "))
    return commits >= 2 and ("Author:" in head or "Date:" in head)


def _looks_like_tree(text: str) -> bool:
    """
    True for ``tree``/directory diagrams.

    Directory trees mark almost every line; a source file may contain one or two
    box-drawing characters in a comment. Requiring most lines to be marked keeps
    such a file from being capped as if it were a listing.
    """
    lines = [line for line in text[:4000].splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    marked = sum(1 for line in lines if any(marker in line for marker in _TREE_CHARS))
    return marked >= 5 and marked >= len(lines) * 0.5


def _looks_like_grep(text: str) -> bool:
    """
    True when most lines look like ``path:line:content``.

    Timestamps satisfy the same shape (``2026-10-31 12:34:56 [ERROR] ...`` parses
    as path ``2026-10-31 12``, line ``34``, content ``56 [ERROR] ...``), so a
    build log would be treated as grep output, regrouped by hour and truncated —
    dropping the actual error line. Lines that start with a date or clock time
    therefore disqualify the text from being grep output.
    """
    lines = text.splitlines()
    if len(lines) < 20:
        return False
    sample = lines[:200]

    # A traceback is a stack, not a file listing: "  File \"/tmp/x.py\", line 51"
    # satisfies the path:line:content shape. Regrouping it hid the exception text.
    if "Traceback (most recent call last)" in text:
        return False

    stamped = sum(1 for line in sample if _TIMESTAMP_LINE_RE.match(line))
    if stamped >= max(3, len(sample) // 10):
        return False

    hits = sum(1 for line in sample if _GREP_LINE_RE.match(line))
    return hits >= min(len(sample), 20) * 0.8


def _looks_like_listing(text: str) -> bool:
    """
    True when most lines look like file paths.

    Code is excluded explicitly: a minified bundle or a block of imports also has
    space-free lines containing slashes, but it is a file read and must not be
    capped as though it were a directory listing.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 40:
        return False
    sample = lines[:200]

    def _is_path(line: str) -> bool:
        stripped = line.strip()
        return (
            "/" in stripped
            and " " not in stripped
            and not any(mark in stripped for mark in ("{", "}", "(", ")", "=", ";", "->", "=>"))
        )

    hits = sum(1 for line in sample if _is_path(line))
    return hits >= len(sample) * 0.8


def _looks_like_log(text: str) -> bool:
    """True for build/test logs: long output with repeats or level prefixes."""
    lines = text.splitlines()
    if len(lines) < 60:
        return False
    sample = lines[:400]
    repeated = any(sample[i] == sample[i - 1] for i in range(1, len(sample)))
    hinted = sum(1 for line in sample if _LOG_HINT_RE.match(line))
    return repeated or hinted >= len(sample) * 0.3


# ---------------------------------------------------------------------------
# Compressors
# ---------------------------------------------------------------------------

def _hard_cap(text: str) -> str:
    """Keep the head and tail of a pathologically large output."""
    head = text[:HARD_CAP_HEAD]
    tail = text[-HARD_CAP_TAIL:]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[... {omitted} bytes omitted ...]{MARKER}\n{tail}"


def _smart_truncate(text: str) -> str:
    """Keep the head and tail of a long text (file reads)."""
    lines = text.splitlines()
    head = lines[:SMART_TRUNCATE_HEAD]
    tail = lines[-SMART_TRUNCATE_TAIL:]
    omitted = len(lines) - len(head) - len(tail)
    return (
        "\n".join(head)
        + f"\n[... {omitted} lines omitted ...]{MARKER}\n"
        + "\n".join(tail)
    )


def _compress_lines(text: str, limit: int) -> str:
    """Cap the number of lines, keeping the head."""
    lines = text.splitlines()
    if len(lines) <= limit:
        return text
    omitted = len(lines) - limit
    return "\n".join(lines[:limit]) + f"\n[... {omitted} lines omitted ...]{MARKER}"


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


def _compress_git_status(text: str) -> str:
    """
    Collapse ``git status`` to a branch line plus the first files per section.

    The full status is mostly a long file list; what matters is which section
    things are in and roughly how many.
    """
    lines = text.splitlines()
    branch = ""
    sections: Dict[str, List[str]] = {"staged": [], "modified": [], "untracked": []}
    current: Optional[str] = None

    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("On branch "):
            branch = stripped[len("On branch "):]
            continue
        if stripped.startswith("Changes to be committed"):
            current = "staged"
            continue
        if stripped.startswith("Changes not staged"):
            current = "modified"
            continue
        if stripped.startswith("Untracked files"):
            current = "untracked"
            continue
        if stripped.startswith(("Your branch", "nothing to commit", "(")) or stripped.startswith('"'):
            continue
        if current is None:
            continue

        name = stripped
        for prefix in _STATUS_FILE_PREFIXES:
            if stripped.startswith(prefix):
                name = stripped[len(prefix):].strip()
                break
        sections[current].append(name)

    out: List[str] = []
    if branch:
        out.append(f"* {branch}")
    labels = {"staged": "+ Staged", "modified": "~ Modified", "untracked": "? Untracked"}
    total = 0
    for key in ("staged", "modified", "untracked"):
        items = sections[key]
        total += len(items)
        if not items:
            continue
        out.append(f"{labels[key]}: {len(items)} file(s)")
        for name in items[:STATUS_MAX_FILES]:
            out.append(f"   {name}")
        if len(items) > STATUS_MAX_FILES:
            out.append(f"   ... +{len(items) - STATUS_MAX_FILES} more")

    if total == 0:
        out.append("clean — nothing to commit")

    return "\n".join(out) + MARKER


def _compress_git_log(text: str) -> str:
    """Keep the newest commits and cap the listing."""
    lines = text.splitlines()
    if len(lines) <= MAX_GIT_LOG_LINES:
        return text
    omitted = len(lines) - MAX_GIT_LOG_LINES
    return (
        "\n".join(lines[:MAX_GIT_LOG_LINES])
        + f"\n[... {omitted} lines omitted ...]{MARKER}"
    )


def _compress_tree(text: str) -> str:
    """Drop the summary line and cap the tree."""
    kept: List[str] = []
    for line in text.splitlines():
        if "director" in line and "file" in line:
            continue
        if not line.strip() and not kept:
            continue
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()

    if len(kept) > MAX_TREE_LINES:
        omitted = len(kept) - MAX_TREE_LINES
        return "\n".join(kept[:MAX_TREE_LINES]) + f"\n[... {omitted} more entries ...]{MARKER}"
    if len(kept) == len(text.splitlines()):
        return text
    return "\n".join(kept)


def _compress_grep(text: str) -> str:
    """
    Group grep hits per file and keep the first hits of each.

    Mirrors RTK's ``grep`` filter: a header with the match count, then up to
    ``GREP_PER_FILE_MAX`` hits per file and a ``+N`` marker for the rest.

    The rewrite only happens when the text is almost entirely hits. A mixed
    output (hits plus prose, a table or a stack trace) cannot be regrouped
    without dropping the unmatched lines, and dropping content is worse than a
    large prompt: the model re-runs the command to see what was removed.
    """
    lines = text.splitlines()
    matched_lines = sum(1 for line in lines if _GREP_LINE_RE.match(line))
    if matched_lines < len(lines) * 0.8:
        return text

    by_file: Dict[str, List[Tuple[str, str]]] = {}
    total = 0
    for line in lines:
        match = _GREP_LINE_RE.match(line)
        if not match:
            continue
        path, number, content = match.groups()
        by_file.setdefault(path, []).append((number, content))
        total += 1

    if total == 0:
        return text

    files = sorted(by_file)
    out: List[str] = [f"{total} matches in {len(files)} file(s):", ""]
    line_budget = MAX_GREP_LINES

    for path in files:
        matches = by_file[path]
        block = [f"[file] {path} ({len(matches)}):"]
        for number, content in matches[:GREP_PER_FILE_MAX]:
            block.append(f"  {number:>4}: {content.strip()}")
        if len(matches) > GREP_PER_FILE_MAX:
            block.append(f"  +{len(matches) - GREP_PER_FILE_MAX} more")
        block.append("")

        if line_budget - len(block) < 0:
            out.append(f"[... {len(files) - files.index(path)} file(s) omitted ...]")
            break
        out.extend(block)
        line_budget -= len(block)

    return "\n".join(out).rstrip() + MARKER


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
    middle = deduped[MAX_LOG_LINES // 2 : -(MAX_LOG_LINES // 2)]

    # Failures are what the model needs from a build log, and they are rarely in
    # the first or last 200 lines of a 2000-line output. Dropping them made the
    # model re-run the build to see the error again — the single most expensive
    # possible reaction (a 40s build plus a full prefilled prompt).
    error_lines = [line for line in middle if _LOG_ERROR_RE.search(line)]
    omitted = len(middle)
    parts: List[str] = list(head)
    if error_lines and len(error_lines) <= MAX_LOG_KEPT_ERRORS:
        parts.append(
            f"[... {omitted} lines omitted; {len(error_lines)} error/failure "
            f"line(s) from the omitted part are kept below ...]"
        )
        parts.extend(error_lines)
    elif error_lines:
        kept = error_lines[:MAX_LOG_KEPT_ERRORS]
        parts.append(
            f"[... {omitted} lines omitted; first {len(kept)} of "
            f"{len(error_lines)} error/failure line(s) kept below ...]"
        )
        parts.extend(kept)
    else:
        parts.append(f"[... {omitted} lines omitted ...]{MARKER}")
    parts.extend(tail)
    return "\n".join(parts)
