# -*- coding: utf-8 -*-

"""Tests for runtime effort-default overrides."""

import kiro.effort_store as store


def _reset(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_DEFAULTS_FILE", tmp_path / "effort_defaults.json")
    monkeypatch.setattr(store, "_overrides", {})
    monkeypatch.setattr(store, "_loaded", True)


def test_override_persists_and_reloads(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)

    store.set_override("claude-opus-5.5", "high")
    assert store.get_effective_default("claude-opus-5.5") == "high"
    assert store.get_override("claude-opus-5.5") == "high"

    # Simulate a process restart: clear memory, reload from disk.
    store._overrides.clear()
    store._loaded = False
    assert store.get_effective_default("claude-opus-5.5") == "high"


def test_empty_override_means_kiro_default(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)

    store.set_override("claude-haiku-5.5", "")
    assert store.get_override("claude-haiku-5.5") == ""
    assert store.get_effective_default("claude-haiku-5.5") is None


def test_removing_override_inherits_builtin(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)

    store.set_override("claude-sonnet-5.5", "low")
    assert store.get_effective_default("claude-sonnet-5.5") == "low"

    store.set_override("claude-sonnet-5.5", None)
    assert store.get_override("claude-sonnet-5.5") is None
    assert store.get_effective_default("claude-sonnet-5.5") == "high"
