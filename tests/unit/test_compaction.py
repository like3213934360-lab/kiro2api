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
        assert compaction.auto_compact_threshold("claude-opus-5.5") == 950_000
        assert compaction.auto_compact_threshold("claude-sonnet-4.5") == 190_000
        assert compaction.auto_compact_threshold("unknown-model") == 300_000

    def test_suffix_and_alias_are_handled(self):
        """
        What it does: Uses an effort suffix and an alias.
        Purpose: The model name in the payload is not always the bare id.
        """
        assert compaction.auto_compact_threshold("claude-opus-5.5-xhigh") == 950_000
        assert compaction.auto_compact_threshold("auto-kiro") == 950_000

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


class TestSummaryInstructions:
    """
    The handoff summary must stop the next turn from redoing finished work.

    A summary that only lists "what is pending" makes the model re-derive facts
    it already verified; that is how one session ran the same grep 71 times.
    """

    def test_requires_an_already_verified_checklist(self):
        """
        What it does: Inspects the summarisation instruction.
        Purpose: Confirmed facts and command outcomes must be preserved verbatim,
                 not summarised away as chit-chat.
        """
        from kiro.compaction import _SUMMARY_INSTRUCTION

        instruction = _SUMMARY_INSTRUCTION.lower()
        assert "already verified" in instruction
        assert "exit code" in instruction
        assert "must\nnot repeat work" in instruction or "not repeat work" in instruction

    def test_requires_the_working_tree_state(self):
        """
        What it does: Checks that the summary must capture git state.
        Purpose: Without it the next turn re-runs `git status` to rediscover what
                 is modified, staged or deleted.
        """
        from kiro.compaction import _SUMMARY_INSTRUCTION

        assert "working tree" in _SUMMARY_INSTRUCTION
        assert "staged" in _SUMMARY_INSTRUCTION

    def test_is_not_an_instruction_to_the_model_being_summarised(self):
        """
        What it does: Verifies the prompt stays compact.
        Purpose: It is prepended to every summarisation call, so bloat costs
                 tokens on the most expensive request in the session.
        """
        from kiro.compaction import _SUMMARY_INSTRUCTION

        assert len(_SUMMARY_INSTRUCTION) < 3000


class TestSummarizerTranscript:
    """
    The summarizer must see the same verdicts the model saw.

    It is asked to record command outcomes, so a head-only excerpt of each tool
    result removes exactly that: exit codes and failure summaries live at the end.
    """

    def test_excerpt_keeps_the_end_of_a_result(self):
        """
        What it does: Shortens a long tool result.
        Purpose: "exit=1" must survive into the summary prompt.
        """
        from kiro.compaction import _summarizer_excerpt

        text = "grep output\n" + ("match\n" * 2000) + "exit=1"
        excerpt = _summarizer_excerpt(text, 1500)

        assert excerpt.startswith("grep output")
        assert excerpt.endswith("exit=1")
        assert len(excerpt) <= 1600

    def test_short_text_is_unchanged(self):
        """
        What it does: Passes text below the limit.
        Purpose: Small results must reach the summarizer verbatim.
        """
        from kiro.compaction import _summarizer_excerpt

        assert _summarizer_excerpt("ok", 1500) == "ok"

    def test_failed_results_are_labelled(self):
        """
        What it does: Renders a transcript containing an error result.
        Purpose: The summarizer must be able to write "this command failed" into
                 the handoff checklist instead of guessing from text.
        """
        from types import SimpleNamespace

        from kiro.compaction import _render_transcript

        message = SimpleNamespace(role="user", content=[{
            "type": "tool_result", "tool_use_id": "call_1",
            "content": "build failed", "is_error": True,
        }])

        transcript = _render_transcript([message])

        assert "[FAILED]" in transcript
        assert "build failed" in transcript


class TestUnsummarisedMessagesAreCarried:
    """
    A cached summary must not swallow the turns that arrived after it.

    The summary covers everything older than the recent window. While it is
    cached, newly aged-out messages belong to neither the summary nor the recent
    block, so they disappeared from the model's view for up to the cache TTL
    (ten minutes) — and the model repeated that work.
    """

    @staticmethod
    async def _compact(request, conversation_id, monkeypatch):
        return await compaction.maybe_compact_request(
            request,
            auth_manager=object(),
            conversation_id=conversation_id,
            profile_arn="",
            shared_client=None,
            estimated_tokens=500_000,
            threshold_tokens=300_000,
        )

    @pytest.mark.asyncio
    async def test_newly_aged_out_messages_are_kept(self, monkeypatch):
        """
        What it does: Compacts a 14-message request, then an 18-message one.
        Purpose: Messages 6-9 aged out between the two calls; they must still
                 reach the model, without a second summarisation call.
        """
        calls = {"count": 0}

        async def fake_summarize(*args, **kwargs):
            calls["count"] += 1
            return "SUMMARY"

        monkeypatch.setattr(compaction, "_summarize", fake_summarize)

        first = await self._compact(make_request(14), "conv-carry", monkeypatch)
        assert first is not None
        assert len(first.messages) == 8          # summary + last 8

        second = await self._compact(make_request(18), "conv-carry", monkeypatch)
        assert second is not None
        contents = [m.content for m in second.messages]
        print(f"carried block starts at: {contents[0]!r}, total={len(contents)}")

        assert calls["count"] == 1, "the cached summary should be reused"
        assert contents[0] == "message 6"        # first message the summary missed
        assert contents[:4] == ["message 6", "message 7", "message 8", "message 9"]
        assert contents[-1] == "message 17"
        assert "SUMMARY" in second.system

    @pytest.mark.asyncio
    async def test_summarises_again_when_the_history_no_longer_aligns(self, monkeypatch):
        """
        What it does: Compacts, then compacts a request with rewritten content.
        Purpose: A client-side compaction or edit removes the covered message; the
                 stale summary must not be applied to a different history.
        """
        calls = {"count": 0}

        async def fake_summarize(*args, **kwargs):
            calls["count"] += 1
            return f"SUMMARY-{calls['count']}"

        monkeypatch.setattr(compaction, "_summarize", fake_summarize)

        await self._compact(make_request(14), "conv-realigned", monkeypatch)

        rewritten = make_request(18)
        for i, message in enumerate(rewritten.messages):
            message.content = f"rewritten {i}"

        second = await self._compact(rewritten, "conv-realigned", monkeypatch)

        assert second is not None
        assert calls["count"] == 2
        assert "SUMMARY-2" in second.system
        # Nothing is carried when the boundary cannot be located.
        assert len(second.messages) == 8

    @pytest.mark.asyncio
    async def test_unchanged_history_still_reuses_the_summary(self, monkeypatch):
        """
        What it does: Compacts the same request twice.
        Purpose: The boundary check must not defeat the cache it relies on.
        """
        calls = {"count": 0}

        async def fake_summarize(*args, **kwargs):
            calls["count"] += 1
            return "SUMMARY"

        monkeypatch.setattr(compaction, "_summarize", fake_summarize)
        request = make_request(14)

        await self._compact(request, "conv-same", monkeypatch)
        second = await self._compact(request, "conv-same", monkeypatch)

        assert calls["count"] == 1
        assert len(second.messages) == 8
