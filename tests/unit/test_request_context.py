# -*- coding: utf-8 -*-

"""Tests for the per-request measurements shared with the streaming layer."""

import pytest

from kiro.request_context import (
    get_client_prompt_tokens,
    get_payload_estimate,
    set_client_prompt_tokens,
    set_payload_estimate,
)


@pytest.fixture(autouse=True)
def _clear_context_values():
    """Keep context variables from leaking between tests."""
    set_client_prompt_tokens(None)
    set_payload_estimate(None)
    yield
    set_client_prompt_tokens(None)
    set_payload_estimate(None)


class TestClientPromptTokens:
    """The client-facing size must survive the gateway shrinking the prompt."""

    def test_records_the_measured_size(self):
        """
        What it does: Stores a conversation size and reads it back.
        Purpose: The streaming layer reports this number to the client.
        """
        print("Action: record 490_000 tokens...")
        set_client_prompt_tokens(490_000)

        assert get_client_prompt_tokens() == 490_000
        print("✓ Value recorded")

    def test_later_smaller_measurement_does_not_lower_the_value(self):
        """
        What it does: Records the pre-shrink size, then a post-shrink size.
        Purpose: Regression guard. Pruning and compaction run after the real
                 conversation was measured, so a second measurement is always
                 smaller. Reporting it made Claude Code believe a 1.4M-token
                 session was at 4% and it never auto-compacted, which left the
                 model stuck re-running the same commands.
        """
        print("Action: record 1_447_004, then the post-compaction 34_466...")
        set_client_prompt_tokens(1_447_004)
        set_client_prompt_tokens(34_466)

        assert get_client_prompt_tokens() == 1_447_004
        print("✓ Shrinking the prompt never hides the real conversation size")

    def test_larger_measurement_wins(self):
        """
        What it does: Records a smaller value first, then a larger one.
        Purpose: Order independence — whichever layer measures most accurately
                 should win, no matter which runs first.
        """
        set_client_prompt_tokens(120_000)
        set_client_prompt_tokens(900_000)

        assert get_client_prompt_tokens() == 900_000

    @pytest.mark.parametrize("value", [0, -1, -100_000])
    def test_ignores_non_positive_measurements(self, value):
        """
        What it does: Records zero and negative token counts.
        Purpose: An empty or broken measurement must not erase a real one.
        """
        set_client_prompt_tokens(500_000)
        set_client_prompt_tokens(value)

        assert get_client_prompt_tokens() == 500_000

    def test_none_clears_the_value(self):
        """
        What it does: Clears the context variable.
        Purpose: Tests and requests without measurements need a clean slate.
        """
        set_client_prompt_tokens(500_000)
        set_client_prompt_tokens(None)

        assert get_client_prompt_tokens() is None

    def test_is_independent_of_the_payload_estimate(self):
        """
        What it does: Sets both measurements to different values.
        Purpose: The admin log needs the real outgoing payload size for cost
                 analysis, while the client needs the untouched conversation
                 size. Sharing one variable is what caused the under-report.
        """
        set_client_prompt_tokens(1_447_004)
        set_payload_estimate(34_466)

        assert get_client_prompt_tokens() == 1_447_004
        assert get_payload_estimate() == 34_466
