# -*- coding: utf-8 -*-

"""
Unit tests for kiro/cache_meter.py.

Cache hits are inferred from Kiro's billing signal (it reports no cache token
counters), so the estimator must be conservative: unknown until it has seen real
traffic for that model, and stable against output-token noise.
"""

import json

import pytest

from kiro.cache_meter import CacheMeter, OUTPUT_TOKEN_WEIGHT


def feed(meter: CacheMeter, model: str, count: int, price: float, prompt: int = 40_000) -> None:
    """Record `count` requests priced at `price` credits each."""
    for _ in range(count):
        meter.classify(model, prompt, 10, price)


class TestUnitPrice:
    """Tests for the weighted unit price."""

    def test_weights_output_tokens(self):
        """
        What it does: Computes the price for prompt-heavy and output-heavy requests.
        Purpose: Output tokens cost ~7x input, so ignoring them fakes cache hits.
        """
        prompt_price = CacheMeter.unit_price(10_000, 0, 1.0)
        mixed_price = CacheMeter.unit_price(10_000, 1_000, 1.0)

        assert prompt_price == pytest.approx(1.0 / 10_000)
        assert mixed_price == pytest.approx(1.0 / (10_000 + OUTPUT_TOKEN_WEIGHT * 1_000))
        assert mixed_price < prompt_price

    @pytest.mark.parametrize("credits", [None, 0, -1])
    def test_invalid_credits_yield_no_price(self, credits):
        """
        What it does: Passes missing or impossible credit values.
        Purpose: A zero charge must not look like a perfect cache hit.
        """
        assert CacheMeter.unit_price(1_000, 10, credits) is None

    def test_zero_tokens_yield_no_price(self):
        """
        What it does: Passes an empty request.
        Purpose: Division by zero must not leak into traffic.
        """
        assert CacheMeter.unit_price(0, 0, 1.0) is None


class TestClassification:
    """Tests for hit / partial / miss verdicts."""

    @staticmethod
    def _primed(**kwargs) -> CacheMeter:
        """Meter that has seen cold traffic plus one cheap request (spread)."""
        meter = CacheMeter(min_samples=3, state_path=None, **kwargs)
        for _ in range(3):
            meter.classify("m", 40_000, 10, 2.5)
        meter.classify("m", 40_000, 10, 0.25)  # reveals the spread, learns the peak
        return meter

    def test_uniform_traffic_is_unknown(self):
        """
        What it does: Feeds only identical prices.
        Purpose: Without a spread the meter cannot tell hits from misses and
                 must not guess.
        """
        meter = CacheMeter(min_samples=3, state_path=None)

        labels = [meter.classify("m", 40_000, 10, 2.5).label for _ in range(6)]

        assert set(labels) == {"unknown"}

    def test_full_price_is_miss(self):
        """
        What it does: Feeds full-price traffic after the peak is known.
        Purpose: Ordinary uncached requests must be labelled 'miss'.
        """
        meter = self._primed()

        observation = meter.classify("m", 40_000, 10, 2.5)

        assert observation.label == "miss"
        assert observation.ratio == pytest.approx(1.0, abs=0.02)

    def test_ten_percent_price_is_hit(self):
        """
        What it does: Charges ~10% of the uncached price.
        Purpose: Measured cache hits are billed ~10% of the uncached price.
        """
        meter = self._primed()

        observation = meter.classify("m", 40_000, 10, 0.25)

        assert observation.label == "hit"
        assert observation.ratio == pytest.approx(0.1, abs=0.02)

    def test_mid_price_is_partial(self):
        """
        What it does: Charges ~70% of the uncached price.
        Purpose: Partial prefix matches must not be reported as full hits.
        """
        meter = self._primed()

        assert meter.classify("m", 40_000, 10, 1.75).label == "partial"

    def test_models_are_tracked_separately(self):
        """
        What it does: Primes two models with very different prices.
        Purpose: A cheap model must not make an expensive one look cached.
        """
        meter = CacheMeter(min_samples=3, state_path=None)
        for _ in range(3):
            meter.classify("expensive", 40_000, 10, 2.5)
            meter.classify("cheap", 40_000, 10, 0.05)
        meter.classify("expensive", 40_000, 10, 0.25)
        meter.classify("cheap", 40_000, 10, 0.005)

        assert meter.classify("expensive", 40_000, 10, 2.5).label == "miss"
        assert meter.classify("cheap", 40_000, 10, 0.05).label == "miss"

    def test_long_output_does_not_look_like_a_hit(self):
        """
        What it does: Charges a similar amount for a much larger output.
        Purpose: Output weighting keeps long answers from being misread.
        """
        meter = CacheMeter(min_samples=3, state_path=None)
        for _ in range(3):
            meter.classify("m", 10_000, 100, 1.0)
        meter.classify("m", 10_000, 100, 0.1)

        observation = meter.classify("m", 10_000, 5_000, 3.0)

        assert observation.label in ("partial", "miss")

    def test_no_credits_reports_unknown(self):
        """
        What it does: Omits the credit amount.
        Purpose: Missing metering data must not be guessed.
        """
        meter = self._primed()

        assert meter.classify("m", 40_000, 10, None).label == "unknown"

    def test_reset_forgets_samples(self):
        """
        What it does: Resets after learning a price.
        Purpose: Account or credential switches must start from scratch.
        """
        meter = self._primed()

        meter.reset()

        assert meter.classify("m", 40_000, 10, 2.5).label == "unknown"

    def test_hits_stay_classified_when_they_dominate(self):
        """
        What it does: Sends one cold request followed by many hits.
        Purpose: With a percentile reference, a mostly-hit workload would drift
                 until hits looked like misses; the reference must stay a cold price.
        """
        meter = self._primed()

        labels = [meter.classify("m", 40_000, 10, 0.25).label for _ in range(50)]

        assert set(labels) == {"hit"}

    def test_format_is_log_friendly(self):
        """
        What it does: Renders an observation for the log line.
        Purpose: Operators read this on every request.
        """
        meter = self._primed()

        formatted = meter.classify("m", 40_000, 10, 0.25).format()

        assert formatted.startswith("hit")
        assert "0.10x" in formatted
        assert CacheMeter(min_samples=3, state_path=None).classify(
            "m", 1, 1, 1.0
        ).format() == "unknown"


class TestPersistence:
    """Tests for learned prices surviving restarts."""

    def test_peak_survives_a_new_meter(self, tmp_path):
        """
        What it does: Learns a cold price, then builds a second meter.
        Purpose: A long warm streak must still classify as hits after a restart.
        """
        state = tmp_path / "cache_meter.json"
        first = CacheMeter(min_samples=3, state_path=state)
        for _ in range(3):
            first.classify("m", 40_000, 10, 2.5)
        first.classify("m", 40_000, 10, 0.25)  # learns and persists the peak

        second = CacheMeter(min_samples=3, state_path=state)
        observation = second.classify("m", 40_000, 10, 0.25)

        assert observation.label == "hit"
        assert state.exists()

    def test_peak_only_moves_up(self, tmp_path):
        """
        What it does: Feeds a higher, then a lower price.
        Purpose: A cheap request must never lower the uncached reference.
        """
        state = tmp_path / "cache_meter.json"
        meter = CacheMeter(min_samples=1, state_path=state)
        meter.classify("m", 40_000, 10, 0.25)
        meter.classify("m", 40_000, 10, 5.0)  # 1.25e-4, above the seed
        meter.classify("m", 40_000, 10, 1.0)
        meter.classify("m", 40_000, 10, 2.0)

        stored = json.loads(state.read_text())["peaks"]["m"]

        assert stored == pytest.approx(CacheMeter.unit_price(40_000, 10, 5.0))

    def test_corrupt_state_file_is_ignored(self, tmp_path):
        """
        What it does: Points the meter at a non-JSON file.
        Purpose: Corrupt local state must never break request handling.
        """
        state = tmp_path / "cache_meter.json"
        state.write_text("{not json")

        meter = CacheMeter(min_samples=3, state_path=state)

        assert meter.classify("m", 40_000, 10, 2.5).label == "unknown"

    def test_reset_clears_persisted_prices(self, tmp_path):
        """
        What it does: Learns a price, resets, then re-reads the file.
        Purpose: Account switches must not keep foreign pricing.
        """
        state = tmp_path / "cache_meter.json"
        meter = CacheMeter(min_samples=1, state_path=state)
        meter.classify("m", 40_000, 10, 2.5)

        meter.reset()

        assert json.loads(state.read_text())["peaks"] == {}


class TestNoSpread:
    """Tests for windows that cannot distinguish hits from misses."""

    def test_uniform_prices_are_unknown(self):
        """
        What it does: Feeds several identical prices with no learned peak.
        Purpose: An all-warm streak must not be mislabelled as 'miss'.
        """
        meter = CacheMeter(min_samples=3, state_path=None)

        labels = [meter.classify("m", 40_000, 10, 0.25).label for _ in range(6)]

        assert set(labels) == {"unknown"}

    def test_spread_enables_classification(self):
        """
        What it does: Feeds one expensive and several cheap requests.
        Purpose: A visible spread is enough to identify the uncached price.
        """
        meter = CacheMeter(min_samples=3, state_path=None)
        meter.classify("m", 40_000, 10, 2.5)
        meter.classify("m", 40_000, 10, 0.25)
        meter.classify("m", 40_000, 10, 0.25)

        observation = meter.classify("m", 40_000, 10, 0.25)

        assert observation.label == "hit"


class TestValidation:
    """Tests for constructor validation."""

    @pytest.mark.parametrize("window", [0, -1])
    def test_window_must_be_positive(self, window):
        """A non-positive window cannot hold a reference price."""
        with pytest.raises(ValueError):
            CacheMeter(window=window, state_path=None)

    def test_thresholds_must_be_ordered(self):
        """Hit threshold above partial threshold is meaningless."""
        with pytest.raises(ValueError):
            CacheMeter(hit_ratio=0.9, partial_ratio=0.5, state_path=None)


class TestPeakMerging:
    """Persisted prices must never be erased by a process that saw less."""

    def test_saving_keeps_a_higher_price_from_disk(self, tmp_path):
        """
        What it does: Seeds a high price, then lets a meter learn a lower one.
        Purpose: A warm-only session would otherwise overwrite the real
                 uncached baseline with a hit price and mislabel every request.
        """
        state = tmp_path / "cache_meter.json"
        state.write_text(json.dumps({"peaks": {"m": 6.3e-05}}))

        meter = CacheMeter(min_samples=1, state_path=state)
        meter.classify("m", 40_000, 10, 0.25)  # cheap sample
        meter.classify("m", 40_000, 10, 2.0)   # 5e-5, still below the seed

        assert json.loads(state.read_text())["peaks"]["m"] == pytest.approx(6.3e-05)

    def test_saving_raises_a_lower_price_on_disk(self, tmp_path):
        """
        What it does: Seeds a low price, then observes a higher one.
        Purpose: Pricing changes (or a bad seed) must be correctable.
        """
        state = tmp_path / "cache_meter.json"
        state.write_text(json.dumps({"peaks": {"m": 1e-06}}))

        meter = CacheMeter(min_samples=1, state_path=state)
        meter.classify("m", 40_000, 10, 0.25)
        meter.classify("m", 40_000, 10, 5.0)  # 1.25e-4, above the seed

        assert json.loads(state.read_text())["peaks"]["m"] == pytest.approx(5.0 / 40_070)
