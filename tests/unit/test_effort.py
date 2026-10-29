# -*- coding: utf-8 -*-

"""
Tests for native Kiro reasoning-effort resolution.
"""

import pytest

from kiro.effort import (
    build_effort_fields,
    budget_to_effort,
    normalize_effort,
    resolve_effort,
    split_effort_suffix,
    supports_effort,
)


class TestEffortSuffix:
    def test_supported_suffix_is_stripped(self):
        assert split_effort_suffix("claude-opus-5.5-high") == (
            "claude-opus-5.5",
            "high",
            "claude-opus-5.5",
        )
        assert split_effort_suffix("claude-sonnet-5.5-xhigh")[:2] == (
            "claude-sonnet-5.5",
            "xhigh",
        )

    def test_unsupported_suffix_is_kept(self):
        base, effort, _ = split_effort_suffix("claude-haiku-4.5-high")
        assert base == "claude-haiku-4.5-high"
        assert effort is None

    def test_plain_model_is_unchanged(self):
        assert split_effort_suffix("claude-opus-5.5") == (
            "claude-opus-5.5",
            None,
            "claude-opus-5.5",
        )


class TestEffortResolution:
    def test_suffix_wins(self):
        assert resolve_effort(
            "claude-opus-5.5-xhigh",
            "claude-opus-5.5",
            output_config={"effort": "low"},
        ) == "xhigh"

    def test_output_config_effort(self):
        assert resolve_effort(
            "claude-opus-5.5",
            "claude-opus-5.5",
            output_config={"effort": "medium"},
        ) == "medium"

    def test_budget_mapping(self):
        assert budget_to_effort(8_000) == "low"
        assert budget_to_effort(16_000) == "medium"
        assert budget_to_effort(24_000) == "high"
        assert budget_to_effort(32_000) == "max"
        assert resolve_effort(
            "claude-opus-5.5",
            "claude-opus-5.5",
            thinking={"type": "enabled", "budget_tokens": 16_000},
        ) == "medium"

    def test_disabled_thinking_wins(self):
        assert resolve_effort(
            "claude-opus-5.5",
            "claude-opus-5.5",
            output_config={"effort": "high"},
            thinking={"type": "disabled"},
        ) is None

    def test_haiku_does_not_support_effort(self):
        assert not supports_effort("claude-haiku-4.5")
        assert resolve_effort(
            "claude-haiku-4.5",
            "claude-haiku-4.5",
            output_config={"effort": "high"},
        ) is None

    def test_xhigh_is_clamped_on_older_models(self):
        assert resolve_effort(
            "claude-opus-4.5",
            "claude-opus-4.5",
            output_config={"effort": "xhigh"},
        ) == "max"

    def test_gpt_uses_reasoning_field(self):
        assert build_effort_fields("gpt-5.6-sol", "high") == {
            "reasoning": {"effort": "high"}
        }
        assert build_effort_fields("claude-opus-5.5", "high") == {
            "output_config": {"effort": "high"}
        }

    def test_normalize_effort(self):
        assert normalize_effort("HIGH") == "high"
        assert normalize_effort("none") is None
