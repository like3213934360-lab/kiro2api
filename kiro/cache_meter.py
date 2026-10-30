# -*- coding: utf-8 -*-

"""
Prompt-cache hit estimation from Kiro's billing signal.

Kiro does not report token-level cache counters: its stream carries
``meteringEvent`` (credits) and ``contextUsageEvent`` (a percentage), but no
``cacheReadInputTokens`` / ``cacheCreationInputTokens``. The only observable
signal is therefore the credit amount charged for a request.

Output tokens are billed roughly seven times an input token, so

    unit_price = credits / (input_tokens + 7 * output_tokens)

is stable for a given model. Requests billed clearly below that model's
uncached price were served (at least partly) from the upstream prefix cache.

Kiro's cache is content-addressed: an identical prompt prefix is reused even
across different conversation ids. That makes the *bytes* of the prompt the only
thing that matters, which is why rewriting an old tool result (context pruning)
kills the cache for every token after the edit.

The meter learns each model's uncached price from the most expensive request
seen and persists it, so a long warm streak (where every request is a hit) still
classifies correctly. No configuration is required.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

# Output tokens cost about this many input tokens on Kiro (measured: 6.9e-5
# credits/input token vs 4.9e-4 credits/output token on claude-opus-5.5).
OUTPUT_TOKEN_WEIGHT = 7.0

# Below this fraction of the learned uncached price the request is a hit.
HIT_RATIO = 0.60

# Between HIT_RATIO and PARTIAL_RATIO: partially cached.
PARTIAL_RATIO = 0.85

# Samples kept per model. Long enough that a cold request usually stays in the
# window; the persisted peak covers the case where it doesn't.
WINDOW = 500

# Minimum samples before a verdict is given from traffic alone.
MIN_SAMPLES = 3

# A window whose prices barely differ cannot distinguish hits from misses.
MIN_SPREAD = 1.30

# Where learned per-model prices survive restarts.
DEFAULT_STATE_PATH = Path(__file__).resolve().parent.parent / "cache_meter.json"


@dataclass(frozen=True)
class CacheObservation:
    """One classified request."""

    label: str
    ratio: Optional[float]

    def format(self) -> str:
        """
        Render as a compact log token.

        Returns:
            ``hit (0.11x)``, ``partial (0.72x)``, ``miss (1.02x)`` or ``unknown``
        """
        if self.ratio is None:
            return self.label
        return f"{self.label} ({self.ratio:.2f}x)"


# Kiro bills a cache hit at roughly this fraction of the uncached input price
# (measured: turns that reuse the prefix are billed ~10% of a cold prefill).
CACHE_READ_PRICE_FRACTION = 0.10


def estimate_cached_prompt_tokens(
    prompt_tokens: int, observation: Optional[CacheObservation]
) -> int:
    """
    Estimate how many prompt tokens were served from the prefix cache.

    Kiro reports no token-level counters, so the split is derived from the
    charge ratio. If a fraction ``f`` of the prompt is cached and cached tokens
    cost ``CACHE_READ_PRICE_FRACTION`` of an uncached one, the ratio is

        ratio = f * 0.10 + (1 - f) * 1.0

    which inverts to ``f = (1 - ratio) / 0.9``. The estimate is clamped to the
    prompt size and only produced for verdicts that actually indicate reuse: a
    ``miss`` is billed at the uncached price, and fabricating a small hit from
    the noise around it would overstate the cache. Cross-account and
    cross-conversation reuse is real, because Kiro's cache is content-addressed.

    Args:
        prompt_tokens: Prompt size reported for the request
        observation: Cache verdict for the request

    Returns:
        Estimated cached prompt tokens (0 when nothing was reused)
    """
    if not prompt_tokens or prompt_tokens <= 0:
        return 0
    if observation is None or observation.ratio is None:
        return 0
    if observation.label not in ("hit", "partial"):
        return 0
    fraction = (1.0 - observation.ratio) / (1.0 - CACHE_READ_PRICE_FRACTION)
    fraction = min(1.0, max(0.0, fraction))
    return int(round(prompt_tokens * fraction))


class CacheMeter:
    """
    Rolling per-model cache-hit estimator with a persisted uncached price.

    Args:
        window: Samples kept per model
        hit_ratio: Upper bound for a full hit
        partial_ratio: Upper bound for a partial hit
        min_samples: Samples required before classifying from traffic alone
        state_path: JSON file holding learned prices (None disables persistence)
    """

    def __init__(
        self,
        window: int = WINDOW,
        hit_ratio: float = HIT_RATIO,
        partial_ratio: float = PARTIAL_RATIO,
        min_samples: int = MIN_SAMPLES,
        state_path: Optional[Path] = DEFAULT_STATE_PATH,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0 < hit_ratio <= partial_ratio:
            raise ValueError("expected 0 < hit_ratio <= partial_ratio")
        self.window = window
        self.hit_ratio = hit_ratio
        self.partial_ratio = partial_ratio
        self.min_samples = max(1, min_samples)
        self.state_path = state_path
        self._samples: Dict[str, Deque[float]] = {}
        self._peaks: Dict[str, float] = self._load_peaks()
        self._lock = threading.Lock()

    @staticmethod
    def unit_price(
        prompt_tokens: int, completion_tokens: int, credits: Optional[float]
    ) -> Optional[float]:
        """
        Credits charged per weighted input token.

        Args:
            prompt_tokens: Input token count reported for the request
            completion_tokens: Output token count
            credits: Credits Kiro charged, if reported

        Returns:
            Unit price, or None when it cannot be computed
        """
        if credits is None or credits <= 0:
            return None
        weighted = float(prompt_tokens or 0) + OUTPUT_TOKEN_WEIGHT * float(completion_tokens or 0)
        if weighted <= 0:
            return None
        return credits / weighted

    def classify(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        credits: Optional[float],
    ) -> CacheObservation:
        """
        Classify one request against the model's known uncached price.

        Args:
            model: Model id
            prompt_tokens: Input token count
            completion_tokens: Output token count
            credits: Credits Kiro charged

        Returns:
            Observation labelled ``hit``, ``partial``, ``miss`` or ``unknown``
        """
        price = self.unit_price(prompt_tokens, completion_tokens, credits)
        if price is None:
            return CacheObservation("unknown", None)

        with self._lock:
            samples = self._samples.get(model)
            if samples is None:
                samples = deque(maxlen=self.window)
                self._samples[model] = samples
            reference = self._reference_locked(model, samples)
            samples.append(price)
            self._learn_peak_locked(model, samples)

        if reference is None or reference <= 0:
            return CacheObservation("unknown", None)

        ratio = price / reference
        if ratio < self.hit_ratio:
            label = "hit"
        elif ratio < self.partial_ratio:
            label = "partial"
        else:
            label = "miss"
        return CacheObservation(label, ratio)

    def _reference_locked(self, model: str, samples: Deque[float]) -> Optional[float]:
        """
        Uncached reference price for a model.

        Prefers the persisted peak (it survives long warm streaks); otherwise
        falls back to the window maximum, but only when the window actually
        shows a spread. An all-warm window with no known peak is reported as
        unknown rather than wrongly as a miss.

        Args:
            model: Model id
            samples: Recorded unit prices (before the current one)

        Returns:
            Reference price, or None when it cannot be established
        """
        persisted = self._peaks.get(model, 0.0)
        if persisted > 0:
            return max(persisted, max(samples) if samples else 0.0)

        if len(samples) < self.min_samples:
            return None
        highest = max(samples)
        lowest = min(samples)
        if lowest <= 0 or highest / lowest < MIN_SPREAD:
            return None
        return highest

    def _learn_peak_locked(self, model: str, samples: Deque[float]) -> None:
        """
        Record the uncached price once the window shows a real spread.

        A stream of identical prices cannot tell hits from misses; only when a
        cheaper request appears does the earlier, higher price become a usable
        reference. The peak is persisted so a later warm streak still classifies.

        Args:
            model: Model id
            samples: Recorded unit prices including the current request
        """
        if len(samples) < 2:
            return
        highest = max(samples)
        lowest = min(samples)
        if lowest <= 0 or highest / lowest < MIN_SPREAD:
            return
        if highest <= self._peaks.get(model, 0.0):
            return
        self._peaks[model] = highest
        self._save_peaks()

    def _load_peaks(self) -> Dict[str, float]:
        """
        Load persisted uncached prices.

        Returns:
            Mapping of model id to the most expensive unit price seen
        """
        if self.state_path is None:
            return {}
        try:
            data = json.loads(Path(self.state_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        peaks: Dict[str, float] = {}
        for model, value in (data.get("peaks") or {}).items():
            if isinstance(value, (int, float)) and value > 0:
                peaks[str(model)] = float(value)
        return peaks

    def _save_peaks(self) -> None:
        """
        Persist learned prices, merging with what is already on disk.

        A long-lived process holds only the prices it has seen itself; writing
        its dictionary verbatim would erase a better baseline learned by another
        run (or seeded by hand). Peaks only ever move up.
        """
        if self.state_path is None:
            return
        try:
            path = Path(self.state_path)
            merged = self._load_peaks()
            for model, value in self._peaks.items():
                merged[model] = max(value, merged.get(model, 0.0))
            self._peaks.update(merged)
            path.write_text(
                json.dumps({"peaks": merged}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            return

    def reset(self) -> None:
        """Forget every sample and learned price (tests, account switch)."""
        with self._lock:
            self._samples.clear()
            self._peaks.clear()
            self._save_peaks()


# Process-wide meter shared by all API surfaces.
cache_meter = CacheMeter()
