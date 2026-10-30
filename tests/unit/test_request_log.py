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


def test_note_payload_estimate_sets_the_request_context():
    """
    What it does: Records a payload size and reads the context variable.
    Purpose: Streaming code reports this number to the client so its
             auto-compaction fires at the right time.
    """
    from types import SimpleNamespace

    from kiro.request_context import get_payload_estimate
    from kiro.request_log import note_payload_estimate

    request = SimpleNamespace(state=SimpleNamespace())
    estimate = note_payload_estimate(request, {"a": "x" * 700})

    assert get_payload_estimate() == estimate


class TestFormatPerfLine:
    """One greppable line per request, so latency questions do not need guessing."""

    def test_includes_latency_and_token_fields(self):
        """
        What it does: Formats a populated request entry.
        Purpose: ttfb and duration used to live only in the in-memory admin log,
                 so after a restart the only way to explain a slow turn was to
                 reconstruct it from the client's own transcripts.
        """
        from kiro.request_log import format_perf_line

        line = format_perf_line({
            "status": 200,
            "input_tokens": 201_595,
            "output_tokens": 190,
            "payload_estimate_tokens": 206_993,
            "payload_bytes": 724_478,
            "ttfb_ms": 1234.5,
            "duration_ms": 6543.2,
            "speed_tps": 28.4,
            "cache_hit": "hit",
            "stream": True,
        })

        print(line)
        assert line.startswith("[Perf]")
        assert "in=201595" in line
        assert "prompt_est=206993" in line
        assert "ttfb_ms=1234" in line
        assert "duration_ms=6543" in line
        assert "speed_tps=28" in line
        assert "cache=hit" in line

    def test_missing_values_are_explicit(self):
        """
        What it does: Formats an entry without timing data (non-streaming).
        Purpose: A missing measurement must read as "-", never as 0, which would
                 look like a real and suspiciously fast request.
        """
        from kiro.request_log import format_perf_line

        line = format_perf_line({"status": 200, "input_tokens": 10, "stream": False})

        assert "ttfb_ms=-" in line
        assert "duration_ms=-" in line
        assert "speed_tps=-" in line
        assert "cache=unknown" in line
