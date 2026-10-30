# -*- coding: utf-8 -*-

"""Tests for the in-memory admin request log."""

from kiro.request_log import RequestLogStore


def test_add_list_and_clear():
    store = RequestLogStore(max_entries=3)
    store.add(endpoint="/v1/messages", model="m1", status=200)
    store.add(endpoint="/v1/messages", model="m2", status=500)
    store.add(endpoint="/v1/messages", model="m3", status=200)
    store.add(endpoint="/v1/messages", model="m4", status=200)

    data = store.list(limit=10)
    assert len(data["logs"]) == 3
    assert [item["model"] for item in data["logs"]] == ["m2", "m3", "m4"]
    assert data["total"] == 4

    assert store.clear() == 3
    assert store.list()["logs"] == []


def test_limit_is_bounded():
    store = RequestLogStore(max_entries=5)
    for i in range(5):
        store.add(endpoint="/v1/messages", model=str(i), status=200)

    assert len(store.list(limit=2)["logs"]) == 2
    assert len(store.list(limit=999)["logs"]) == 5


def test_note_payload_estimate_records_size():
    """
    What it does: Records a payload's serialized size on the request.
    Purpose: The admin log keeps both Kiro-reported tokens and the real payload
             size so cost analysis is not based on a single estimate.
    """
    from types import SimpleNamespace

    from kiro.request_log import note_payload_estimate

    request = SimpleNamespace(state=SimpleNamespace())
    payload = {"conversationState": {"history": [{"userInputMessage": {"content": "x" * 350}}]}}

    estimate = note_payload_estimate(request, payload)

    assert request.state.payload_bytes > 350
    assert estimate == int(request.state.payload_bytes / 3.5)


def test_note_payload_estimate_tolerates_plain_object():
    """
    What it does: Passes an object without a writable state.
    Purpose: Logging helpers must never break a request.
    """
    from kiro.request_log import note_payload_estimate

    class _NoState:
        pass

    estimate = note_payload_estimate(_NoState(), {"a": 1})

    assert estimate >= 0
