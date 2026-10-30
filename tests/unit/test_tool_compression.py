# -*- coding: utf-8 -*-

"""
Unit tests for kiro/tool_compression.py.

The transform must be deterministic and idempotent: Kiro's prefix cache compares
request bytes, so compressing the same tool output on a later turn has to produce
exactly the same text it produced when the output first appeared.
"""

import pytest

from kiro.tool_compression import (
    DEDUPE_MARKER,
    HARD_CAP_BYTES,
    MARKER,
    MIN_COMPRESS_BYTES,
    CompressionStats,
    compress_text,
    compress_tool_outputs,
)


def tool_result(text: str, status: str = "success", tool_id: str = "t1"):
    """Kiro-shaped tool result."""
    return {"toolUseId": tool_id, "status": status, "content": [{"text": text}]}


def payload_with(results, history_results=None):
    """Build a minimal Kiro payload."""
    history = [
        {"userInputMessage": {"content": "go", "userInputMessageContext": {"toolResults": history_results}}}
    ] if history_results is not None else []
    return {
        "conversationState": {
            "history": history,
            "currentMessage": {
                "userInputMessage": {
                    "content": "continue",
                    "userInputMessageContext": {"toolResults": results},
                }
            },
        }
    }


def make_diff(hunks: int = 1, lines_per_hunk: int = 400) -> str:
    """Large unified diff."""
    out = ["diff --git a/app.py b/app.py", "index 123..456 100644", "--- a/app.py", "+++ b/app.py"]
    for h in range(hunks):
        out.append(f"@@ -{h * 10 + 1},3 +{h * 10 + 1},3 @@")
        for i in range(lines_per_hunk):
            out.append(f"+line {h}-{i} " + "x" * 20)
    return "\n".join(out)


def make_grep(count: int = 800) -> str:
    """Grep-like output."""
    return "\n".join(f"src/file{i % 40}.py:{i}:match text {i}" for i in range(count))


def make_listing(count: int = 600) -> str:
    """find/ls-like output."""
    return "\n".join(f"./src/pkg{i % 30}/module{i}.py" for i in range(count))


def make_log(count: int = 1200) -> str:
    """Build log with consecutive repeated lines."""
    lines = []
    for i in range(count):
        lines.extend([f"[build] compiling unit {i}"] * 4)
        lines.append("[build] waiting...")
    return "\n".join(lines)


class TestSmallInputs:
    """Small or already-marked text must be left untouched."""

    def test_short_text_is_unchanged(self):
        """
        What it does: Compresses a short output.
        Purpose: The model may need small results verbatim.
        """
        text = "short output"

        compressed, kind = compress_text(text)

        assert compressed == text
        assert kind is None

    def test_text_at_threshold_is_unchanged(self):
        """
        What it does: Compresses text just below the minimum size.
        Purpose: The threshold must be a hard floor.
        """
        text = "x" * (MIN_COMPRESS_BYTES - 1)

        compressed, kind = compress_text(text)

        assert compressed == text
        assert kind is None


class TestDiffCompression:
    """Unified diffs keep headers and the start of each hunk."""

    def test_headers_survive_and_hunk_is_capped(self):
        """
        What it does: Compresses a 400-line hunk.
        Purpose: The model still needs to know which file and hunk changed.
        """
        text = make_diff(lines_per_hunk=400)

        compressed, kind = compress_text(text)

        assert kind == "diff"
        assert "diff --git a/app.py b/app.py" in compressed
        assert "@@ -1,3 +1,3 @@" in compressed
        assert len(compressed) < len(text)
        assert MARKER in compressed

    def test_compression_is_idempotent(self):
        """
        What it does: Compresses the same diff twice.
        Purpose: A second pass must not change bytes (cache safety).
        """
        text = make_diff(lines_per_hunk=400)

        once, _ = compress_text(text)
        twice, kind = compress_text(once)

        assert twice == once
        assert kind is None

    def test_short_diff_is_untouched(self):
        """
        What it does: Compresses a small diff.
        Purpose: Small diffs are worth sending in full.
        """
        text = make_diff(hunks=1, lines_per_hunk=5)

        compressed, kind = compress_text(text)

        assert compressed == text
        assert kind is None


class TestGrepAndListing:
    """Line-oriented outputs are capped."""

    def test_grep_lines_are_capped(self):
        """
        What it does: Compresses 800 grep hits.
        Purpose: Match spam dominates the prompt without adding much.
        """
        compressed, kind = compress_text(make_grep(800))

        assert kind == "grep"
        assert compressed.count("\n") < 800
        assert "src/file0.py:0:" in compressed

    def test_listing_entries_are_capped(self):
        """
        What it does: Compresses a 600-entry directory listing.
        Purpose: Path lists are repetitive and compressible.
        """
        compressed, kind = compress_text(make_listing(600))

        assert kind == "listing"
        assert compressed.count("\n") < 600

    def test_caps_are_idempotent(self):
        """
        What it does: Compresses capped output again.
        Purpose: Repeated turns must produce identical bytes.
        """
        once, _ = compress_text(make_grep(800))
        twice, kind = compress_text(once)

        assert twice == once
        assert kind is None


class TestLogsAndCap:
    """Build logs are deduplicated and capped; huge blobs keep head and tail."""

    def test_repeated_lines_are_collapsed(self):
        """
        What it does: Compresses a log with alternating repeated lines.
        Purpose: Repeated progress lines add no information.
        """
        compressed, kind = compress_text(make_log(200))

        assert kind == "log"
        assert "repeated" in compressed
        assert len(compressed) < len(make_log(200))

    def test_hard_cap_keeps_head_and_tail(self):
        """
        What it does: Compresses a 200 KB blob.
        Purpose: One pathological output must not blow the whole context.
        """
        text = "A" * 100_000 + "MIDDLE" + "B" * 100_000

        compressed, kind = compress_text(text)

        assert kind == "hard_cap"
        assert compressed.startswith("A" * 1000)
        assert compressed.endswith("B" * 1000)
        assert "omitted" in compressed
        assert len(compressed) <= HARD_CAP_BYTES


class TestPayloadTransform:
    """End-to-end transform over a Kiro payload."""

    def test_errors_are_never_compressed(self):
        """
        What it does: Puts a large diff in a tool result marked error.
        Purpose: Error traces are the most valuable thing to keep verbatim.
        """
        text = make_diff(lines_per_hunk=400)
        payload = payload_with([tool_result(text, status="error")])

        compress_tool_outputs(payload)

        assert payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]["toolResults"][0]["content"][0]["text"] == text

    def test_identical_outputs_are_deduplicated(self):
        """
        What it does: Sends the same large output twice (history + current).
        Purpose: The second copy carries no information.
        """
        text = make_grep(400)
        payload = payload_with([tool_result(text)], history_results=[tool_result(text, tool_id="t0")])

        stats = compress_tool_outputs(payload)

        current = payload["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]["toolResults"][0]["content"][0]["text"]
        history = payload["conversationState"]["history"][0]["userInputMessage"][
            "userInputMessageContext"
        ]["toolResults"][0]["content"][0]["text"]

        assert history != DEDUPE_MARKER
        assert current == DEDUPE_MARKER
        assert stats.hits.get("dedupe") == 1

    def test_bytes_are_stable_when_output_moves_from_current_to_history(self):
        """
        What it does: Compresses a result as current, then as history.
        Purpose: This is the cache-safety guarantee — the bytes sent on turn N
                 must equal the bytes sent on turn N+1.
        """
        original = make_grep(500)

        first = payload_with([tool_result(original)])
        compress_tool_outputs(first)
        sent_as_current = first["conversationState"]["currentMessage"]["userInputMessage"][
            "userInputMessageContext"
        ]["toolResults"][0]["content"][0]["text"]

        # Next turn: the client sends the original bytes again, now as history.
        second = payload_with([], history_results=[tool_result(original)])
        compress_tool_outputs(second)
        sent_as_history = second["conversationState"]["history"][0]["userInputMessage"][
            "userInputMessageContext"
        ]["toolResults"][0]["content"][0]["text"]

        assert sent_as_history == sent_as_current

    def test_stats_only_count_changed_outputs(self):
        """
        What it does: Mixes one compressible and one tiny output.
        Purpose: The log line must reflect real savings.
        """
        payload = payload_with([
            tool_result(make_grep(500), tool_id="big"),
            tool_result("tiny", tool_id="small"),
        ])

        stats = compress_tool_outputs(payload)

        assert stats.saved > 0
        assert stats.hits.get("grep") == 1
        assert "saved" in stats.format()

    def test_no_changes_returns_empty_stats(self):
        """
        What it does: Compresses a payload of small outputs.
        Purpose: Nothing to report means no log line.
        """
        payload = payload_with([tool_result("small output")])

        stats = compress_tool_outputs(payload)

        assert stats.saved == 0
        assert "nothing to compress" in stats.format()


class TestStatsFormatting:
    """The log line is read by operators on every compressed turn."""

    def test_percentages_and_counts(self):
        """
        What it does: Builds a stats object by hand.
        Purpose: The format must stay compact and informative.
        """
        stats = CompressionStats(bytes_before=10_000, bytes_after=6_000, tool_bytes=12_000)
        stats.record("diff")
        stats.record("diff")
        stats.record("grep")

        line = stats.format()

        assert "3.9 KB" in line
        assert "40.0%" in line
        assert "diff=2" in line
        assert "grep=1" in line
