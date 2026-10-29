# -*- coding: utf-8 -*-

"""
Runtime overrides for per-model reasoning effort.

The built-in defaults live in ``config.MODEL_EFFORT_DEFAULTS``.  The admin
panel can override them without restarting the gateway; overrides are stored
in a small JSON file next to the project (gitignored).

Tri-state per model:
- key absent              -> use the built-in/env default;
- value is an effort level -> use that level;
- value is ""             -> send no effort (Kiro's own default).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from kiro.config import MODEL_EFFORT_DEFAULTS


_DEFAULTS_FILE = Path(
    os.getenv(
        "EFFORT_DEFAULTS_FILE",
        str(Path(__file__).resolve().parent.parent / "effort_defaults.json"),
    )
)

_lock = threading.Lock()
_overrides: Dict[str, str] = {}
_loaded = False


def _load_locked() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        if _DEFAULTS_FILE.exists():
            data = json.loads(_DEFAULTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _overrides.update(
                    {
                        str(k): str(v)
                        for k, v in data.items()
                        if isinstance(k, str) and isinstance(v, str)
                    }
                )
                logger.info(
                    f"Loaded {len(_overrides)} effort override(s) from {_DEFAULTS_FILE}"
                )
    except Exception as exc:
        logger.warning(f"Failed to load effort overrides: {exc}")


def _save_locked() -> None:
    try:
        _DEFAULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DEFAULTS_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(_overrides, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, _DEFAULTS_FILE)
    except Exception as exc:
        logger.warning(f"Failed to save effort overrides: {exc}")


def get_override(model_id: str) -> Optional[str]:
    """Return the raw override, or None when the model inherits the default."""
    with _lock:
        _load_locked()
        return _overrides.get(model_id)


def get_effective_default(model_id: str) -> Optional[str]:
    """Return the effort the gateway will send by default for this model."""
    with _lock:
        _load_locked()
        override = _overrides.get(model_id)
    if override is not None:
        return override or None
    return MODEL_EFFORT_DEFAULTS.get(model_id) or None


def set_override(model_id: str, value: Optional[str]) -> None:
    """
    Set an admin override.

    Args:
        model_id: Kiro model id
        value: effort level, "" for Kiro's own default, or None to remove the
               override and inherit the built-in default again.
    """
    with _lock:
        _load_locked()
        if value is None:
            _overrides.pop(model_id, None)
        else:
            _overrides[model_id] = value
        _save_locked()


def all_overrides() -> Dict[str, str]:
    with _lock:
        _load_locked()
        return dict(_overrides)
