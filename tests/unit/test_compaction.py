# -*- coding: utf-8 -*-

"""Tests for automatic, client-transparent conversation compaction."""

import pytest

from kiro.models_anthropic import AnthropicMessage, AnthropicMessagesRequest
import kiro.compaction as compaction


def make_request(n_messages=14):
    messages = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(AnthropicMessage(role=role, content=f"message {i}"))
    return AnthropicMessagesRequest(
        model="claude-opus-5.5",
        messages=messages,
        max_tokens=1024,
        system="Original system prompt",
    )


@pytest.fixture(autouse=True)
def clear_summary_cache():
    compaction._CACHE.clear()
    yield
    compaction._CACHE.clear()


class TestSplit:
    def test_keeps_recent_messages_and_a_user_boundary(self):
        request = make_request(14)
        old, recent = compaction._split_old_and_recent(request.messages, keep=6)

        assert len(recent) == 6
        assert len(old) == 8
        assert recent[0].role == "user"
        assert recent[-1].content == "message 13"

    def test_short_conversation_is_not_compacted(self):
        assert compaction._split_old_and_recent(make_request(4).messages, keep=8) is None


class TestCompaction:
    @pytest.mark.asyncio
    async def test_replaces_old_messages_with_summary(self, monkeypatch):
        calls = {"count": 0}

        async def fake_summarize(*args, **kwargs):
            calls["count"] += 1
            return "SUMMARY OF OLD TURNS"

        monkeypatch.setattr(compaction, "_summarize", fake_summarize)

        request = make_request(14)
        compacted = await compaction.maybe_compact_request(
            request,
            auth_manager=object(),
            conversation_id="conv-1",
            profile_arn="arn:aws:test",
            shared_client=None,
            estimated_tokens=500_000,
            threshold_tokens=300_000,
        )

        assert compacted is not None
        assert len(compacted.messages) == 8
        assert compacted.messages[-1].content == "message 13"
        assert "SUMMARY OF OLD TURNS" in compacted.system
        assert "Original system prompt" in compacted.system
        # Original request is untouched.
        assert len(request.messages) == 14

    @pytest.mark.asyncio
    async def test_summary_is_cached_across_turns(self, monkeypatch):
        calls = {"count": 0}

        async def fake_summarize(*args, **kwargs):
            calls["count"] += 1
            return "CACHED SUMMARY"

        monkeypatch.setattr(compaction, "_summarize", fake_summarize)

        request = make_request(14)
        for _ in range(2):
            compacted = await compaction.maybe_compact_request(
                request,
                auth_manager=object(),
                conversation_id="conv-cache",
                profile_arn="arn:aws:test",
                shared_client=None,
                estimated_tokens=500_000,
                threshold_tokens=300_000,
            )
            assert compacted is not None

        assert calls["count"] == 1

    @pytest.mark.asyncio
    async def test_no_action_below_threshold(self, monkeypatch):
        async def fake_summarize(*args, **kwargs):
            raise AssertionError("should not summarize")

        monkeypatch.setattr(compaction, "_summarize", fake_summarize)

        compacted = await compaction.maybe_compact_request(
            make_request(14),
            auth_manager=object(),
            conversation_id="conv-small",
            profile_arn="",
            shared_client=None,
            estimated_tokens=100_000,
            threshold_tokens=300_000,
        )
        assert compacted is None


class TestAutoCompactThreshold:
    """Compaction must stay a last resort, well above the prune threshold."""

    def test_follows_the_model_window(self):
        """
        What it does: Looks up the threshold for small and large windows.
        Purpose: One fixed 300k threshold compacted 1M-context sessions while
                 pruning was still working on them.
        """
        assert compaction.auto_compact_threshold("claude-opus-5.5") == 700_000
        assert compaction.auto_compact_threshold("claude-sonnet-4.5") == 140_000
        assert compaction.auto_compact_threshold("unknown-model") == 300_000

    def test_suffix_and_alias_are_handled(self):
        """
        What it does: Uses an effort suffix and an alias.
        Purpose: The model name in the payload is not always the bare id.
        """
        assert compaction.auto_compact_threshold("claude-opus-5.5-xhigh") == 700_000
        assert compaction.auto_compact_threshold("auto-kiro") == 700_000

    def test_stays_above_the_prune_threshold(self, monkeypatch):
        """
        What it does: Compares both thresholds for several windows.
        Purpose: Compaction during a pruned session would pay for a summary and
                 drop the prompt cache for no benefit.
        """
        import kiro.converters_core as cc

        for model in ("claude-opus-5.5", "claude-sonnet-4.5", "some-unknown"):
            assert compaction.auto_compact_threshold(model) > cc.auto_prune_threshold(model)

    def test_explicit_override_wins(self, monkeypatch):
        """
        What it does: Sets the legacy env override.
        Purpose: Operators who pinned the old value keep control.
        """
        monkeypatch.setattr(compaction, "AUTO_COMPACT_TOKENS", 123_456)

        assert compaction.auto_compact_threshold("claude-opus-5.5") == 123_456

    @pytest.mark.asyncio
    async def test_pruned_payload_is_not_compacted(self, monkeypatch):
        """
        What it does: Runs compaction on a payload just under the threshold.
        Purpose: After pruning, the payload must fall below compaction and stay
                 untouched.
        """
        request = make_request()

        result = await compaction.maybe_compact_request(
            request,
            auth_manager=None,
            conversation_id="c1",
            profile_arn="arn:test",
            shared_client=None,
            estimated_tokens=compaction.auto_compact_threshold(request.model) - 1,
        )

        assert result is None
