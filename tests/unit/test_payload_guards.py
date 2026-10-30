# -*- coding: utf-8 -*-

"""
Unit tests for payload size guard logic.
Tests check_payload_size() and trim_payload_to_limit() functions.
"""

import json

import pytest

from kiro.payload_guards import (
    PayloadTrimStats,
    check_payload_size,
    trim_payload_to_limit,
)


def _make_payload(num_pairs=5, content_size=100):
    """Helper: build a minimal Kiro-shaped payload with N user/assistant pairs."""
    history = []
    for i in range(num_pairs):
        history.append({
            "userInputMessage": {"content": f"user message {i} " + "x" * content_size}
        })
        history.append({
            "assistantResponseMessage": {"content": f"assistant message {i} " + "y" * content_size}
        })
    return {
        "conversationState": {
            "chatTriggerType": "MANUAL",
            "conversationId": "test-conv",
            "currentMessage": {
                "userInputMessage": {"content": "current message", "modelId": "test"}
            },
            "history": history,
        },
        "profileArn": "arn:aws:test",
    }


class TestCheckPayloadSize:

    def test_check_payload_size_returns_bytes(self):
        """Correct byte count for a simple payload."""
        payload = {"key": "value"}
        expected = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
        assert check_payload_size(payload) == expected

    def test_check_payload_size_utf8(self):
        """Non-ASCII characters counted correctly."""
        payload = {"emoji": "\U0001f600", "chinese": "\u4f60\u597d"}
        size = check_payload_size(payload)
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        assert size == len(raw)
        # json.dumps escapes non-ASCII by default, so byte count matches
        # The important thing is consistency with what gets sent over the wire
        assert size > 0


class TestTrimPayloadToLimit:

    def test_trim_does_nothing_when_under_limit(self):
        """No-op when payload is small."""
        payload = _make_payload(num_pairs=2, content_size=10)
        original_size = check_payload_size(payload)
        stats = trim_payload_to_limit(payload, max_bytes=original_size + 1000)
        assert not stats.trimmed
        assert stats.original_entries == stats.final_entries
        assert stats.original_bytes == stats.final_bytes

    def test_trim_removes_oldest_history_pairs(self):
        """Removes pairs from the beginning of history."""
        payload = _make_payload(num_pairs=10, content_size=500)
        original_size = check_payload_size(payload)
        # Set limit to ~half the original size
        limit = original_size // 2
        stats = trim_payload_to_limit(payload, max_bytes=limit)

        assert stats.trimmed
        assert stats.final_entries < stats.original_entries
        assert stats.final_bytes <= limit
        # History should still exist and be shorter
        history = payload["conversationState"]["history"]
        assert len(history) == stats.final_entries
        assert len(history) < 20  # original was 20 entries (10 pairs)

    def test_trim_aligns_to_user_message(self):
        """Start index always lands on userInputMessage entry."""
        payload = _make_payload(num_pairs=10, content_size=500)
        limit = check_payload_size(payload) // 3
        trim_payload_to_limit(payload, max_bytes=limit)

        history = payload["conversationState"]["history"]
        assert len(history) > 0
        assert "userInputMessage" in history[0]

    def test_trim_preserves_minimum_history(self):
        """Never trims below 2 entries."""
        payload = _make_payload(num_pairs=5, content_size=1000)
        # Set an impossibly low limit
        stats = trim_payload_to_limit(payload, max_bytes=100)

        history = payload["conversationState"]["history"]
        assert len(history) >= 2
        assert stats.final_entries >= 2

    def test_trim_repairs_orphaned_tool_results(self):
        """Orphaned toolResults removed, text preserved inline."""
        history = [
            {"userInputMessage": {"content": "msg0"}},
            {
                "assistantResponseMessage": {
                    "content": "resp0",
                    "toolUses": [{"toolUseId": "tool-A", "name": "read", "input": "{}"}],
                }
            },
            {
                "userInputMessage": {
                    "content": "msg1",
                    "userInputMessageContext": {
                        "toolResults": [
                            {
                                "toolUseId": "tool-A",
                                "content": [{"text": "result from tool-A"}],
                            },
                            {
                                "toolUseId": "tool-ORPHAN",
                                "content": [{"text": "orphaned data"}],
                            },
                        ]
                    },
                }
            },
            {"assistantResponseMessage": {"content": "resp1"}},
        ]
        payload = {
            "conversationState": {
                "conversationId": "test",
                "chatTriggerType": "MANUAL",
                "currentMessage": {"userInputMessage": {"content": "now", "modelId": "m"}},
                "history": history,
            }
        }
        # Trim with a generous limit so only repair logic runs (no pair removal)
        big_limit = check_payload_size(payload) + 10000
        trim_payload_to_limit(payload, max_bytes=big_limit)

        # tool-A should remain, tool-ORPHAN should be removed with text preserved
        ctx = history[2]["userInputMessage"]["userInputMessageContext"]
        assert len(ctx["toolResults"]) == 1
        assert ctx["toolResults"][0]["toolUseId"] == "tool-A"
        # Orphaned text preserved in content
        assert "orphaned data" in history[2]["userInputMessage"]["content"]
        assert "[trimmed tool result]" in history[2]["userInputMessage"]["content"]

    def test_trim_strips_empty_tool_uses(self):
        """Empty toolUses: [] arrays cleaned before size measurement."""
        history = [
            {"userInputMessage": {"content": "msg"}},
            {"assistantResponseMessage": {"content": "resp", "toolUses": []}},
        ]
        payload = {
            "conversationState": {
                "conversationId": "test",
                "chatTriggerType": "MANUAL",
                "currentMessage": {"userInputMessage": {"content": "now", "modelId": "m"}},
                "history": history,
            }
        }
        big_limit = check_payload_size(payload) + 10000
        trim_payload_to_limit(payload, max_bytes=big_limit)

        # Empty toolUses should be stripped
        assert "toolUses" not in history[1]["assistantResponseMessage"]

    def test_trim_stats_accurate(self):
        """Stats reflect actual changes."""
        payload = _make_payload(num_pairs=8, content_size=500)
        original_size = check_payload_size(payload)
        limit = original_size // 2

        stats = trim_payload_to_limit(payload, max_bytes=limit)

        assert stats.original_bytes == original_size
        assert stats.original_entries == 16  # 8 pairs * 2
        assert stats.final_bytes == check_payload_size(payload)
        assert stats.final_entries == len(payload["conversationState"]["history"])
        assert stats.trimmed is True

    def test_trim_no_history(self):
        """Payload with no history returns no-op stats."""
        payload = {
            "conversationState": {
                "conversationId": "test",
                "chatTriggerType": "MANUAL",
                "currentMessage": {"userInputMessage": {"content": "hi", "modelId": "m"}},
            }
        }
        stats = trim_payload_to_limit(payload, max_bytes=100)
        assert not stats.trimmed
        assert stats.original_entries == 0
        assert stats.final_entries == 0


class TestExceedsInputWindow:
    """The post-compaction guard that asks the client to shrink instead of us."""

    def test_prompt_above_the_window_is_rejected(self):
        """
        What it does: Checks a prompt larger than the model window.
        Purpose: Sending it anyway lets the upstream truncate the tail silently,
                 which looks like the model forgot the task.
        """
        from kiro.payload_guards import exceeds_input_window

        assert exceeds_input_window(1_200_000, 1_000_000) is True

    def test_prompt_just_below_the_window_passes(self):
        """
        What it does: Checks a prompt that fits.
        Purpose: Normal requests must never be rejected by this guard.
        """
        from kiro.payload_guards import exceeds_input_window

        assert exceeds_input_window(900_000, 1_000_000) is False

    def test_keeps_headroom_for_the_response(self):
        """
        What it does: Checks a prompt exactly at the window.
        Purpose: A prompt that fills the window leaves no room to answer, so the
                 reserve must still reject it.
        """
        from kiro.payload_guards import exceeds_input_window

        assert exceeds_input_window(1_000_000, 1_000_000) is True
        assert exceeds_input_window(999_500, 1_000_000) is True   # inside the reserve
        assert exceeds_input_window(998_000, 1_000_000) is False

    def test_unknown_window_never_rejects(self):
        """
        What it does: Passes a zero window (model metadata missing).
        Purpose: An unknown limit must not block traffic; the client's own
                 compaction is the fallback.
        """
        from kiro.payload_guards import exceeds_input_window

        assert exceeds_input_window(10_000_000, 0) is False


class TestImageAwareTokenEstimate:
    """
    Inline images must not be measured as text.

    A 288 KB screenshot is ~1,500 tokens but ~84,000 when its base64 is counted
    as ordinary JSON. That inflated number drove pruning, compaction and the
    client's own auto-compaction, hiding history the model needed.
    """

    @staticmethod
    def _image(size: int = 220_000) -> dict:
        import base64
        import os

        return {"format": "jpeg", "source": {"bytes": base64.b64encode(os.urandom(size)).decode()}}

    def test_one_screenshot_costs_about_one_image(self):
        """
        What it does: Estimates a payload carrying a 288 KB screenshot.
        Purpose: The estimate must stay within the image's real cost, not grow
                 with its base64 length.
        """
        from kiro.payload_guards import IMAGE_TOKEN_ESTIMATE, estimate_payload_tokens

        payload = {"conversationState": {"currentMessage": {"userInputMessage": {
            "content": "x" * 400, "images": [self._image()]}}}}
        estimate = estimate_payload_tokens(payload)

        print(f"estimate={estimate}")
        assert estimate < IMAGE_TOKEN_ESTIMATE + 1_000

    def test_text_estimate_matches_the_byte_rule(self):
        """
        What it does: Estimates a payload with no images.
        Purpose: Plain text must keep the previous accuracy.
        """
        from kiro.payload_guards import check_payload_size, estimate_payload_tokens

        payload = {"conversationState": {"currentMessage": {"userInputMessage": {
            "content": "hello world " * 500, "images": []}}}}

        assert estimate_payload_tokens(payload) == int(check_payload_size(payload) / 3.5)

    def test_each_image_is_counted(self):
        """
        What it does: Estimates a payload with five screenshots.
        Purpose: A session accumulates images across turns; the charge is per
                 image, not per payload.
        """
        from kiro.payload_guards import IMAGE_TOKEN_ESTIMATE, estimate_payload_tokens

        payload = {"conversationState": {"history": [
            {"userInputMessage": {"content": "", "images": [self._image(50_000)]}}
            for _ in range(5)
        ]}}

        # Per-image charge, plus the small JSON overhead around each block.
        assert 5 * IMAGE_TOKEN_ESTIMATE <= estimate_payload_tokens(payload) < 5 * IMAGE_TOKEN_ESTIMATE + 500

    def test_anthropic_shaped_images_are_detected(self):
        """
        What it does: Estimates an Anthropic-style image block.
        Purpose: The same helper is used before conversion; both shapes must be
                 recognised or the fix silently stops working.
        """
        from kiro.payload_guards import IMAGE_TOKEN_ESTIMATE, estimate_payload_tokens

        payload = {"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "A" * 100_000}},
        ]}]}

        # Image charge plus the tiny JSON wrapper around it.
        assert IMAGE_TOKEN_ESTIMATE <= estimate_payload_tokens(payload) < IMAGE_TOKEN_ESTIMATE + 200

    def test_real_session_shape_stays_small(self):
        """
        What it does: Reproduces the failing session: 6 screenshots plus 200 KB
                 of tool output.
        Purpose: That payload used to read as >300K tokens; it must now read as
                 the text it actually contains.
        """
        from kiro.payload_guards import estimate_payload_tokens

        payload = {"conversationState": {
            "history": [{"userInputMessage": {"content": "tool output " * 400,
                                              "images": [self._image(180_000)]}} for _ in range(6)],
            "currentMessage": {"userInputMessage": {"content": "continue"}},
        }}

        estimate = estimate_payload_tokens(payload)
        print(f"estimate={estimate:,}")
        assert estimate < 120_000
