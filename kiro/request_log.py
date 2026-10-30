# -*- coding: utf-8 -*-

"""
In-memory request log for the admin console.

This is intentionally not persisted: request metadata can contain model names,
token counts and error text, and the admin console only needs recent history.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List


def note_payload_estimate(request: Any, payload: Any) -> int:
    """
    Record the outgoing payload size on the request for the admin log.

    The gateway reports two token numbers per request: what Kiro says it
    processed (derived from ``contextUsagePercentage``) and what the payload
    actually weighs (serialized bytes / 3.5). They can diverge by an order of
    magnitude on long sessions, so both are kept for cost analysis.

    Args:
        request: FastAPI request whose ``state`` receives the values
        payload: Outgoing Kiro payload

    Returns:
        Estimated prompt tokens
    """
    from kiro.payload_guards import check_payload_size
    from kiro.request_context import set_payload_estimate

    size = int(check_payload_size(payload))
    estimate = int(size / 3.5)
    try:
        request.state.payload_bytes = size
        request.state.payload_estimate_tokens = estimate
    except AttributeError:
        pass
    set_payload_estimate(estimate)
    return estimate



class RequestLogStore:
    def __init__(self, max_entries: int = 500):
        self._entries: deque = deque(maxlen=max_entries)
        self._total = 0
        self._lock = threading.Lock()

    def add(self, **entry: Any) -> None:
        entry.setdefault("time", time.time())
        with self._lock:
            self._entries.append(entry)
            self._total += 1

    def list(self, limit: int = 200) -> Dict[str, Any]:
        limit = max(1, min(int(limit or 200), 500))
        with self._lock:
            entries = list(self._entries)[-limit:]
        return {"logs": entries, "total": self._total, "limit": limit}

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
        return count

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"current": len(self._entries), "total": self._total}


request_log_store = RequestLogStore()
