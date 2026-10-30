# -*- coding: utf-8 -*-

"""
Unit tests for kiro/auth_failures.py.

Verifies the throttled, actionable logging used when an API key is rejected:
first attempts must be logged with enough context to act on, repeats must be
folded into the throttle window, and state must stay bounded.
"""

import threading
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from kiro.auth_failures import (
    DEFAULT_WINDOW_SECONDS,
    MAX_TRACKED_BUCKETS,
    REASON_MISSING_CREDENTIALS,
    REASON_UNKNOWN_KEY,
    UnauthorizedAttemptLogger,
    fingerprint_key,
    log_unauthorized_attempt,
    reset_unauthorized_attempts,
)


class FakeClock:
    """Controllable monotonic clock for throttle-window tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_request(
    path: str = "/v1/chat/completions",
    method: str = "POST",
    client=("127.0.0.1", 51000),
) -> Request:
    """Build a real Starlette Request with a controllable client address."""
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": client,
            "server": ("testserver", 80),
        }
    )


def warnings(records, marker: str = "Rejected"):
    """Filter captured loguru records down to our rejection warnings."""
    return [
        record
        for record in records
        if record["level"].name == "WARNING" and marker in record["message"]
    ]


class TestFingerprintKey:
    """Tests for the non-reversible key fingerprint."""

    def test_empty_key_is_reported_as_absent(self):
        """
        What it does: Verifies no-key requests are labelled explicitly.
        Purpose: Distinguish "no credentials" from "wrong credentials".
        """
        assert fingerprint_key("") == "no key"

    def test_fingerprint_contains_length_and_digest(self):
        """
        What it does: Verifies the fingerprint shape.
        Purpose: Users must be able to tell which key was presented.
        """
        result = fingerprint_key("sk-gw-abcdef")

        assert result.startswith("len=12")
        assert "sha256=" in result
        digest = result.split("sha256=")[1]
        assert len(digest) == 8
        assert all(char in "0123456789abcdef" for char in digest)

    def test_fingerprint_never_contains_the_key(self):
        """
        What it does: Verifies the raw secret is absent from the fingerprint.
        Purpose: Logs must never leak credentials.
        """
        key = "sk-gw-super-secret-value"

        assert key not in fingerprint_key(key)
        assert "super-secret-value" not in fingerprint_key(key)

    def test_fingerprint_is_stable_and_key_specific(self):
        """
        What it does: Verifies determinism and separation between keys.
        Purpose: The same key must match across log lines; other keys must not.
        """
        assert fingerprint_key("key-one") == fingerprint_key("key-one")
        assert fingerprint_key("key-one") != fingerprint_key("key-two")

    def test_fingerprint_handles_unicode_and_long_keys(self):
        """
        What it does: Verifies fingerprints work for odd key material.
        Purpose: Keys are user-controlled strings.
        """
        for key in ("ключ", "a" * 4096, "sk-gw-🔑"):
            result = fingerprint_key(key)
            assert f"len={len(key)}" in result


class TestFirstAttemptLogging:
    """Tests for the first rejection in a bucket."""

    def test_first_attempt_logs_one_actionable_warning(self, loguru_records):
        """
        What it does: Verifies the first rejection is reported.
        Purpose: Security signal must not be lost by throttling.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())

        emitted = logger.report(
            scheme="OpenAI",
            request=make_request(),
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert emitted is True
        logged = warnings(loguru_records)
        assert len(logged) == 1
        message = logged[0]["message"]
        assert "Rejected OpenAI request" in message
        assert "127.0.0.1:51000" in message
        assert "POST /v1/chat/completions" in message
        assert REASON_MISSING_CREDENTIALS in message
        assert "no key" in message
        # Actionable part: tell the user how to authenticate.
        assert "Authorization: Bearer" in message
        assert "x-api-key" in message

    def test_unknown_key_logs_fingerprint_and_not_secret(self, loguru_records):
        """
        What it does: Verifies invalid-key rejections include a fingerprint.
        Purpose: Users can identify the offending key without leaking it.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())
        key = "sk-gw-wrong-key"

        logger.report(
            scheme="Anthropic",
            request=make_request(path="/v1/messages", method="POST"),
            reason=REASON_UNKNOWN_KEY,
            key=key,
        )

        message = warnings(loguru_records)[0]["message"]
        assert REASON_UNKNOWN_KEY in message
        assert fingerprint_key(key) in message
        assert key not in message
        assert "POST /v1/messages" in message

    def test_client_without_transport_is_labelled(self, loguru_records):
        """
        What it does: Verifies requests without a client address still log.
        Purpose: ASGI transports may omit ``client``.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())

        logger.report(
            scheme="OpenAI",
            request=make_request(client=None),
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert "unknown-client" in warnings(loguru_records)[0]["message"]

    def test_mocked_request_does_not_crash_logging(self, loguru_records):
        """
        What it does: Verifies partially-mocked requests are tolerated.
        Purpose: Existing unit tests pass MagicMock requests to the routers.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())
        request = MagicMock()
        request.client = None

        emitted = logger.report(
            scheme="OpenAI",
            request=request,
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert emitted is True
        message = warnings(loguru_records)[0]["message"]
        assert "? /" in message


class TestThrottling:
    """Tests for folding repeated rejections into a window."""

    def test_repeats_inside_window_are_suppressed(self, loguru_records):
        """
        What it does: Verifies identical repeats do not log again.
        Purpose: A 30s poller must not add a warning every 30s.
        """
        clock = FakeClock()
        logger = UnauthorizedAttemptLogger(window_seconds=60, clock=clock)
        request = make_request()

        results = []
        for _ in range(5):
            results.append(
                logger.report(
                    scheme="OpenAI",
                    request=request,
                    reason=REASON_MISSING_CREDENTIALS,
                )
            )
            clock.advance(5)

        assert results[0] is True
        assert results[1:] == [False, False, False, False]
        assert len(warnings(loguru_records)) == 1

    def test_next_window_reports_suppressed_count(self, loguru_records):
        """
        What it does: Verifies the catch-up warning carries the folded count.
        Purpose: Operators still learn how noisy the caller was.
        """
        clock = FakeClock()
        logger = UnauthorizedAttemptLogger(window_seconds=60, clock=clock)
        request = make_request()

        logger.report(
            scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
        )
        for _ in range(3):
            logger.report(
                scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
            )

        clock.advance(61)
        emitted = logger.report(
            scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
        )

        assert emitted is True
        logged = warnings(loguru_records)
        assert len(logged) == 2
        assert "3 identical attempt(s) suppressed" in logged[1]["message"]

    def test_suppressed_counter_resets_after_each_warning(self, loguru_records):
        """
        What it does: Verifies the counter does not accumulate forever.
        Purpose: Each warning must describe only its own window.
        """
        clock = FakeClock()
        logger = UnauthorizedAttemptLogger(window_seconds=10, clock=clock)
        request = make_request()

        logger.report(scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY)
        clock.advance(11)
        logger.report(scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY)
        clock.advance(11)
        logger.report(scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY)

        logged = warnings(loguru_records)
        assert len(logged) == 3
        assert "suppressed" not in logged[1]["message"]
        assert "suppressed" not in logged[2]["message"]

    def test_window_boundary_exactly_at_limit_logs(self, loguru_records):
        """
        What it does: Verifies the inclusive edge of the window.
        Purpose: Boundary conditions must not silence real warnings.
        """
        clock = FakeClock()
        logger = UnauthorizedAttemptLogger(window_seconds=30, clock=clock)
        request = make_request()

        logger.report(scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY)
        clock.advance(30)
        emitted = logger.report(
            scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY
        )

        assert emitted is True
        assert len(warnings(loguru_records)) == 2

    def test_different_clients_are_tracked_separately(self, loguru_records):
        """
        What it does: Verifies throttling is per caller.
        Purpose: One noisy client must not hide another client's failures.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())

        logger.report(
            scheme="OpenAI",
            request=make_request(client=("10.0.0.1", 1000)),
            reason=REASON_MISSING_CREDENTIALS,
        )
        logger.report(
            scheme="OpenAI",
            request=make_request(client=("10.0.0.2", 1001)),
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert len(warnings(loguru_records)) == 2
        assert logger.tracked_buckets() == 2

    def test_different_paths_are_tracked_separately(self, loguru_records):
        """
        What it does: Verifies throttling is per endpoint.
        Purpose: A failure on one endpoint must not mute another.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())

        logger.report(
            scheme="OpenAI", request=make_request(path="/v1/chat/completions"),
            reason=REASON_MISSING_CREDENTIALS,
        )
        logger.report(
            scheme="OpenAI", request=make_request(path="/v1/messages"),
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert len(warnings(loguru_records)) == 2

    def test_different_reasons_are_tracked_separately(self, loguru_records):
        """
        What it does: Verifies missing vs unknown credentials are distinct.
        Purpose: A missing header is a different problem than a wrong key.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())
        request = make_request()

        logger.report(
            scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
        )
        logger.report(scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY)

        logged = warnings(loguru_records)
        assert len(logged) == 2
        assert REASON_MISSING_CREDENTIALS in logged[0]["message"]
        assert REASON_UNKNOWN_KEY in logged[1]["message"]

    def test_short_window_logs_again_quickly(self, loguru_records):
        """
        What it does: Verifies a 1s window re-logs after the window elapses.
        Purpose: Throttling must be tunable without breaking behaviour.
        """
        clock = FakeClock()
        logger = UnauthorizedAttemptLogger(window_seconds=1, clock=clock)
        request = make_request()

        for _ in range(2):
            logger.report(
                scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
            )
            clock.advance(1.1)

        assert len(warnings(loguru_records)) == 2

    def test_reset_allows_immediate_logging_again(self, loguru_records):
        """
        What it does: Verifies reset clears throttle state.
        Purpose: Tests and admin tooling need a clean slate.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())
        request = make_request()

        logger.report(scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY)
        assert logger.tracked_buckets() == 1

        logger.reset()

        assert logger.tracked_buckets() == 0
        assert logger.report(
            scheme="OpenAI", request=request, reason=REASON_UNKNOWN_KEY
        ) is True
        assert len(warnings(loguru_records)) == 2


class TestBoundedState:
    """Tests that throttle memory stays bounded."""

    def test_bucket_cap_is_enforced(self, loguru_records):
        """
        What it does: Verifies the tracked-bucket cap.
        Purpose: Port/path rotation must not grow memory without limit.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock(), max_buckets=3)

        for index in range(20):
            logger.report(
                scheme="OpenAI",
                request=make_request(client=(f"10.0.0.{index}", 1000 + index)),
                reason=REASON_MISSING_CREDENTIALS,
            )

        assert logger.tracked_buckets() <= 3

    def test_stale_buckets_are_pruned(self, loguru_records):
        """
        What it does: Verifies idle buckets expire.
        Purpose: Memory must not grow across days of polling.
        """
        clock = FakeClock()
        logger = UnauthorizedAttemptLogger(window_seconds=10, clock=clock)

        logger.report(
            scheme="OpenAI",
            request=make_request(client=("10.0.0.1", 1)),
            reason=REASON_MISSING_CREDENTIALS,
        )
        clock.advance(10 * 11)
        logger.report(
            scheme="OpenAI",
            request=make_request(client=("10.0.0.2", 2)),
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert logger.tracked_buckets() == 1

    def test_default_limits_are_sane(self):
        """
        What it does: Verifies the module defaults.
        Purpose: Defaults define production behaviour.
        """
        logger = UnauthorizedAttemptLogger()

        assert logger.window_seconds == DEFAULT_WINDOW_SECONDS
        assert logger.max_buckets == MAX_TRACKED_BUCKETS
        assert DEFAULT_WINDOW_SECONDS > 0
        assert MAX_TRACKED_BUCKETS > 0


class TestConcurrency:
    """Tests thread-safety of the shared throttle."""

    def test_concurrent_reports_emit_single_warning(self, loguru_records):
        """
        What it does: Verifies only one warning per bucket under contention.
        Purpose: Sync endpoints run in a thread pool and can race.
        """
        logger = UnauthorizedAttemptLogger(window_seconds=3600, clock=FakeClock())
        request = make_request()
        barrier = threading.Barrier(8)
        results = []

        def worker() -> None:
            barrier.wait()
            results.append(
                logger.report(
                    scheme="OpenAI",
                    request=request,
                    reason=REASON_MISSING_CREDENTIALS,
                )
            )

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results.count(True) == 1
        assert len(warnings(loguru_records)) == 1


class TestConstructorValidation:
    """Tests argument validation."""

    @pytest.mark.parametrize("window", [0, -1, -0.5])
    def test_invalid_window_is_rejected(self, window):
        """
        What it does: Verifies non-positive windows fail fast.
        Purpose: A zero window would divide the throttle semantics.
        """
        with pytest.raises(ValueError):
            UnauthorizedAttemptLogger(window_seconds=window)

    @pytest.mark.parametrize("max_buckets", [0, -1])
    def test_invalid_bucket_cap_is_rejected(self, max_buckets):
        """
        What it does: Verifies non-positive caps fail fast.
        Purpose: The cap must be a usable bound.
        """
        with pytest.raises(ValueError):
            UnauthorizedAttemptLogger(max_buckets=max_buckets)


class TestSharedLogger:
    """Tests for the module-level shared throttle."""

    def test_shared_logger_throttles_across_calls(self, loguru_records, reset_auth_throttle):
        """
        What it does: Verifies the process-wide helper throttles.
        Purpose: Both API surfaces must share one throttle.
        """
        request = make_request()

        first = log_unauthorized_attempt(
            scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
        )
        second = log_unauthorized_attempt(
            scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
        )

        assert first is True
        assert second is False
        assert len(warnings(loguru_records)) == 1

    def test_reset_clears_shared_state(self, loguru_records, reset_auth_throttle):
        """
        What it does: Verifies reset_unauthorized_attempts clears state.
        Purpose: Test isolation depends on it.
        """
        request = make_request()

        log_unauthorized_attempt(
            scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
        )
        reset_unauthorized_attempts()
        emitted = log_unauthorized_attempt(
            scheme="OpenAI", request=request, reason=REASON_MISSING_CREDENTIALS
        )

        assert emitted is True
        assert len(warnings(loguru_records)) == 2


class TestCapabilityProbe:
    """Anonymous model-list polling is discovery, not an auth attack."""

    def test_keyless_model_list_stays_at_debug(self, loguru_records):
        """
        What it does: Reports a missing key on GET /v1/models.
        Purpose: Desktop and editor clients poll this to find a local gateway.
                 Warning per poll flooded the log (measured: 231 in four hours)
                 and polluted authentication statistics.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())

        logger.report(
            scheme="OpenAI",
            request=make_request(path="/v1/models", method="GET"),
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert warnings(loguru_records) == []
        debugged = [
            r for r in loguru_records
            if r["level"].name == "DEBUG" and "GET /v1/models" in r["message"]
        ]
        assert debugged

    def test_wrong_key_on_model_list_still_warns(self, loguru_records):
        """
        What it does: Reports an invalid key on the same endpoint.
        Purpose: A presented-but-wrong key is a real authentication failure and
                 must stay visible.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())

        logger.report(
            scheme="OpenAI",
            request=make_request(path="/v1/models", method="GET"),
            reason=REASON_UNKNOWN_KEY,
            key="sk-wrong-key-1234567890",
        )

        assert len(warnings(loguru_records)) == 1

    def test_post_to_models_still_warns(self, loguru_records):
        """
        What it does: Reports a missing key on a non-GET request to the same path.
        Purpose: Only the read-only discovery probe is quiet; anything else keeps
                 the security signal.
        """
        logger = UnauthorizedAttemptLogger(clock=FakeClock())

        logger.report(
            scheme="OpenAI",
            request=make_request(path="/v1/models", method="POST"),
            reason=REASON_MISSING_CREDENTIALS,
        )

        assert len(warnings(loguru_records)) == 1
