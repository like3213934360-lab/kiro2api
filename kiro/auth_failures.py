# -*- coding: utf-8 -*-

"""
Throttled, actionable logging for rejected API-key attempts.

Local tooling (IDE agents, desktop chat clients, health monitors, ``curl``
one-liners) frequently probes ``GET /health`` and ``GET /v1/models`` on a fixed
interval. When such a probe carries no credentials, the gateway correctly
answers ``401`` but used to log a bare ``Access attempt with invalid API key.``
line for every single request. A 30-second poller therefore buried real
warnings under identical noise, and the message contained nothing that would
help identify the caller.

This module keeps the security signal and removes the noise:

* the first rejection in a ``(client, path, reason)`` bucket is logged at
  ``WARNING`` with actionable diagnostics (caller address, method, path,
  failure reason, key fingerprint, and how to authenticate);
* identical repeats inside the throttle window are counted, not logged;
* once the window expires, the next rejection is logged again and reports how
  many attempts were suppressed in the meantime.

Keys are never logged. ``fingerprint_key`` exposes only the key length and the
first 8 hex characters of its SHA-256 digest, which is enough for a user to
tell which of their keys was sent without leaking the secret.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from fastapi import Request
from loguru import logger


# Failure reasons. Stable strings so they can be grepped in logs.
REASON_MISSING_CREDENTIALS = "missing_credentials"
REASON_UNKNOWN_KEY = "unknown_key"

# Seconds during which repeated identical rejections are folded together.
DEFAULT_WINDOW_SECONDS = 60.0

# Upper bound on tracked buckets, so a hostile client cannot grow the dict
# without limit by rotating source ports or paths.
MAX_TRACKED_BUCKETS = 256

# Entries untouched for this many windows are dropped during pruning.
_STALE_WINDOW_MULTIPLIER = 10

_UNKNOWN_CLIENT = "unknown-client"

_AUTH_HINT = (
    "Send the admin key as 'Authorization: Bearer <PROXY_API_KEY>' (OpenAI) or "
    "'x-api-key: <key>' (Anthropic), or create an external key in the admin "
    "panel. Automated health checks must send the same credentials as real "
    "requests."
)


def fingerprint_key(key: str) -> str:
    """
    Build a non-reversible fingerprint of a presented API key.

    Args:
        key: Raw key extracted from the request, or an empty string

    Returns:
        ``"no key"`` when nothing was presented, otherwise
        ``"len=<n>, sha256=<first 8 hex chars>"``

    Examples:
        >>> fingerprint_key("")
        'no key'
        >>> fingerprint_key("sk-gw-secret")[:4]
        'len='
    """
    if not key:
        return "no key"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return f"len={len(key)}, sha256={digest}"


@dataclass
class _Bucket:
    """Throttling state for one ``(client, path, reason)`` bucket."""

    last_logged: float
    suppressed: int = 0


BucketKey = Tuple[str, str, str]


class UnauthorizedAttemptLogger:
    """
    Log rejected API-key attempts once per throttle window per caller.

    Args:
        window_seconds: Throttle window length in seconds
        max_buckets: Maximum number of tracked buckets before pruning
        clock: Monotonic clock, injectable for tests

    Attributes:
        window_seconds: Configured throttle window
        max_buckets: Configured bucket cap
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_buckets: int = MAX_TRACKED_BUCKETS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_buckets <= 0:
            raise ValueError("max_buckets must be positive")
        self.window_seconds = window_seconds
        self.max_buckets = max_buckets
        self._clock = clock
        self._buckets: Dict[BucketKey, _Bucket] = {}
        self._lock = threading.Lock()

    def report(
        self,
        *,
        scheme: str,
        request: Request,
        reason: str,
        key: str = "",
    ) -> bool:
        """
        Record a rejected request, logging it unless it is throttled.

        Args:
            scheme: API surface that rejected the request (``"OpenAI"``,
                ``"Anthropic"``)
            request: Incoming FastAPI request
            reason: One of ``REASON_MISSING_CREDENTIALS`` or
                ``REASON_UNKNOWN_KEY``
            key: Key presented by the caller, empty when none was sent

        Returns:
            True if a warning was emitted, False if the attempt was folded
            into the current throttle window
        """
        client = self._client_host(request)
        path = self._request_path(request)
        bucket_key: BucketKey = (client, path, reason)
        now = self._clock()

        with self._lock:
            bucket = self._buckets.get(bucket_key)
            if bucket is None:
                self._prune(now)
                bucket = _Bucket(last_logged=now)
                self._buckets[bucket_key] = bucket
                suppressed = 0
                should_log = True
            elif now - bucket.last_logged >= self.window_seconds:
                suppressed = bucket.suppressed
                bucket.suppressed = 0
                bucket.last_logged = now
                should_log = True
            else:
                bucket.suppressed += 1
                suppressed = bucket.suppressed
                should_log = False

        if should_log:
            self._emit_warning(
                scheme=scheme,
                request=request,
                reason=reason,
                key=key,
                client=client,
                suppressed=suppressed,
            )
        else:
            logger.debug(
                f"Throttled unauthorized {scheme} request from {client} on "
                f"{self._request_method(request)} {path} ({reason}); "
                f"{suppressed} attempt(s) within the {self.window_seconds:g}s window"
            )
        return should_log

    def reset(self) -> None:
        """Forget all throttle state (used by tests and admin tooling)."""
        with self._lock:
            self._buckets.clear()

    def tracked_buckets(self) -> int:
        """
        Return how many buckets are currently tracked.

        Returns:
            Number of ``(client, path, reason)`` entries in memory
        """
        with self._lock:
            return len(self._buckets)

    @staticmethod
    def _client_host(request: Request) -> str:
        """Return the caller address, or a placeholder when unavailable."""
        client = getattr(request, "client", None)
        host = getattr(client, "host", None)
        if isinstance(host, str) and host:
            return host
        return _UNKNOWN_CLIENT

    @staticmethod
    def _client_port(request: Request) -> Optional[int]:
        """Return the caller source port when the transport exposes one."""
        port = getattr(getattr(request, "client", None), "port", None)
        return port if isinstance(port, int) else None

    @staticmethod
    def _request_path(request: Request) -> str:
        """Return the request path, tolerating partly-mocked requests."""
        path = getattr(getattr(request, "url", None), "path", None)
        return path if isinstance(path, str) and path else "/"

    @staticmethod
    def _request_method(request: Request) -> str:
        """Return the HTTP method, tolerating partly-mocked requests."""
        method = getattr(request, "method", None)
        return method if isinstance(method, str) and method else "?"

    def _prune(self, now: float) -> None:
        """
        Drop stale buckets and enforce the bucket cap.

        Args:
            now: Current monotonic timestamp
        """
        stale_after = self.window_seconds * _STALE_WINDOW_MULTIPLIER
        stale = [
            bucket_key
            for bucket_key, bucket in self._buckets.items()
            if now - bucket.last_logged > stale_after
        ]
        for bucket_key in stale:
            del self._buckets[bucket_key]

        overflow = len(self._buckets) - self.max_buckets + 1
        if overflow <= 0:
            return
        oldest = sorted(
            self._buckets.items(), key=lambda item: item[1].last_logged
        )[:overflow]
        for bucket_key, _ in oldest:
            del self._buckets[bucket_key]

    def _emit_warning(
        self,
        *,
        scheme: str,
        request: Request,
        reason: str,
        key: str,
        client: str,
        suppressed: int,
    ) -> None:
        """
        Write the actionable warning for one rejected attempt.

        Args:
            scheme: API surface name
            request: Incoming request
            reason: Failure reason string
            key: Presented key (only its fingerprint is logged)
            client: Caller host
            suppressed: Repeats folded since the last warning
        """
        port = self._client_port(request)
        origin = f"{client}:{port}" if port else client
        message = (
            f"Rejected {scheme} request from {origin}: "
            f"{self._request_method(request)} {self._request_path(request)} "
            f"({reason}, {fingerprint_key(key)}). {_AUTH_HINT}"
        )
        if suppressed:
            message += (
                f" [{suppressed} identical attempt(s) suppressed in the last "
                f"{self.window_seconds:g}s]"
            )
        logger.warning(message)


# Process-wide throttle shared by every API surface, so a client that probes
# several endpoints still produces at most one warning per endpoint per window.
_unauthorized_attempts = UnauthorizedAttemptLogger()


def log_unauthorized_attempt(
    *,
    scheme: str,
    request: Request,
    reason: str,
    key: str = "",
) -> bool:
    """
    Report a rejected API-key attempt to the shared throttle.

    Args:
        scheme: API surface that rejected the request
        request: Incoming FastAPI request
        reason: ``REASON_MISSING_CREDENTIALS`` or ``REASON_UNKNOWN_KEY``
        key: Key presented by the caller, empty when none was sent

    Returns:
        True if a warning was emitted, False if it was throttled
    """
    return _unauthorized_attempts.report(
        scheme=scheme, request=request, reason=reason, key=key
    )


def reset_unauthorized_attempts() -> None:
    """Clear the shared throttle state (used by tests)."""
    _unauthorized_attempts.reset()
