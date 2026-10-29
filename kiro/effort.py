# -*- coding: utf-8 -*-

"""
Kiro native reasoning-effort support.

Kiro exposes a native reasoning depth for effort-capable models.  The wire
shape is the same one the Kiro SDK uses:

    additionalModelRequestFields = {
        "output_config": {"effort": "high"}     # Claude models
    }
    additionalModelRequestFields = {
        "reasoning": {"effort": "high"}         # GPT-5.6 models on Mantle
    }

The gateway accepts the level from three places, in priority order:

1. a model-name suffix (``claude-opus-5.5-high``), which also lets the
   upstream advertise/choose a level without any client change;
2. ``output_config.effort`` from the client (Claude Code's /effort);
3. ``thinking.budget_tokens`` from the client, mapped to an effort level;
4. an optional upstream default (``MODEL_EFFORT_DEFAULTS``).

Levels: low, medium, high, xhigh, max.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from kiro.model_resolver import normalize_model_name


EFFORT_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

# Models Kiro accepts effort for. Keep both dotted and dashed spellings:
# normalize_model_name() usually produces the dotted form, but callers may
# resolve before normalization.
_OPUS_BASE = {
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-opus-4.8",
    "claude-opus-5",
    "claude-opus-5.5",
    "claude-opus-5.5-1m",
}
_SONNET_BASE = {
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "claude-sonnet-5",
    "claude-sonnet-5.5",
    "claude-sonnet-5.5-1m",
}
_HAIKU_BASE = {"claude-haiku-5.5"}
_GPT_BASE = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}

# Claude models use output_config.effort; GPT 5.6 uses reasoning.effort.
GPT_EFFORT_MODELS = frozenset(_GPT_BASE)

# xhigh is only accepted by these; on older effort models it is clamped to max.
XHIGH_CAPABLE_MODELS = frozenset(
    {
        "claude-opus-4.7",
        "claude-opus-4.8",
        "claude-opus-5",
        "claude-opus-5.5",
        "claude-opus-5.5-1m",
        "claude-sonnet-5",
        "claude-sonnet-5.5",
        "claude-sonnet-5.5-1m",
        "claude-haiku-5.5",
    }
    | _GPT_BASE
)

EFFORT_CAPABLE_MODELS = frozenset(_OPUS_BASE | _SONNET_BASE | _HAIKU_BASE | _GPT_BASE)


def _canonical_model(model_id: str) -> str:
    """Normalize a Kiro model id to the dotted form used by the catalog."""
    if not model_id:
        return ""
    return normalize_model_name(model_id)


def supports_effort(model_id: str) -> bool:
    """Return True when Kiro accepts a native effort level for this model."""
    return _canonical_model(model_id) in EFFORT_CAPABLE_MODELS


def _clamp_effort(model_id: str, effort: str) -> str:
    if effort == "xhigh" and _canonical_model(model_id) not in XHIGH_CAPABLE_MODELS:
        return "max"
    return effort


def normalize_effort(value: Any) -> Optional[str]:
    """Validate/normalize a client-provided effort value."""
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return value if value in EFFORT_LEVELS else None


def split_effort_suffix(model_name: str) -> Tuple[str, Optional[str], str]:
    """
    Split a model name like ``claude-opus-5.5-high`` into its base name.

    Returns:
        (base_model_name, effort or None, normalized_base_model_id)
    """
    if not isinstance(model_name, str):
        return model_name, None, model_name

    match = re.match(
        r"^(?P<base>.+)-(?P<effort>low|medium|high|xhigh|max)$",
        model_name.strip(),
        re.IGNORECASE,
    )
    if match:
        base = match.group("base")
        normalized = normalize_model_name(base)
        if supports_effort(normalized):
            return base, match.group("effort").lower(), normalized

    return model_name, None, normalize_model_name(model_name)


def budget_to_effort(budget: Any) -> Optional[str]:
    """Map an Anthropic thinking budget to a Kiro effort level."""
    if not isinstance(budget, (int, float)) or budget <= 0:
        return None
    if budget <= 10_000:
        return "low"
    if budget <= 20_000:
        return "medium"
    if budget <= 28_000:
        return "high"
    return "max"


def resolve_effort(
    original_model: str,
    model_id: str,
    output_config: Optional[Dict[str, Any]] = None,
    thinking: Optional[Dict[str, Any]] = None,
    default_effort: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve the native effort level for a request.

    Priority: model suffix > output_config.effort > thinking budget >
    upstream default. Returns None when the model does not support effort or
    when the client explicitly disabled thinking.
    """
    if not supports_effort(model_id):
        return None

    # An explicit "thinking disabled" is the strongest signal: do not send an
    # effort level even if output_config or a model suffix asks for one.
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        return None

    _, suffix_effort, _ = split_effort_suffix(original_model)
    if suffix_effort:
        return _clamp_effort(model_id, suffix_effort)

    if isinstance(output_config, dict):
        effort = normalize_effort(output_config.get("effort"))
        if effort:
            return _clamp_effort(model_id, effort)

    if isinstance(thinking, dict):
        if thinking.get("type") in ("enabled", "adaptive"):
            effort = budget_to_effort(thinking.get("budget_tokens"))
            if effort:
                return _clamp_effort(model_id, effort)

    effort = normalize_effort(default_effort)
    if effort:
        return _clamp_effort(model_id, effort)

    return None


def build_effort_fields(model_id: str, effort: str) -> Dict[str, Any]:
    """Build the ``additionalModelRequestFields`` payload for Kiro."""
    if _canonical_model(model_id) in GPT_EFFORT_MODELS:
        return {"reasoning": {"effort": effort}}
    return {"output_config": {"effort": effort}}
