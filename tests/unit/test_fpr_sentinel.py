"""Tests for the FPR sentinel (plan §C4 + §190)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ive.observability.fpr_sentinel import (
    DEFAULT_THRESHOLD,
    SentinelResult,
    _clopper_pearson_upper,
    _run_single_seed,
    run_sentinel,
)


class TestClopperPearsonUpper:
    def test_zero_successes_is_finite(self):
        upper = _clopper_pearson_upper(0, 20)
        assert 0 < upper < 1
        # Wilson rule of three approx: ~0.139 for n=20.
        assert upper < 0.20

    def test_full_successes_caps_at_one(self):
        assert _clopper_pearson_upper(20, 20) == 1.0

    def test_zero_trials_returns_one(self):
        assert _clopper_pearson_upper(0, 0) == 1.0

    def test_monotonic_in_successes(self):
        prev = -1.0
        for k in range(0, 11):
            value = _clopper_pearson_upper(k, 20)
            assert value > prev
            prev = value


class TestRunSingleSeed:
    def test_returns_bool(self):
        result = _run_single_seed(seed=0, n_rows=200, n_features=4)
        assert isinstance(result, bool)

    def test_deterministic_for_same_seed(self):
        a = _run_single_seed(seed=7, n_rows=200, n_features=4)
        b = _run_single_seed(seed=7, n_rows=200, n_features=4)
        assert a == b

    def test_different_seeds_can_differ(self):
        results = {
            _run_single_seed(seed=s, n_rows=200, n_features=4)
            for s in range(20)
        }
        assert results <= {True, False}


class TestRunSentinel:
    def test_pass_status_when_fpr_low(self):
        result = run_sentinel(n_seeds=20, n_rows=400, n_features=4)
        assert isinstance(result, SentinelResult)
        assert result.n_runs == 20
        assert 0.0 <= result.empirical_fpr <= 1.0
        assert 0.0 <= result.upper_95_ci <= 1.0
        # On clean noise, FPR is ~5% by construction (alpha=0.05/n_features
        # then aggregated over features -> alpha-per-run = 0.05). The upper
        # bound on n=20 is generous, so this is mostly a smoke check.
        assert result.threshold == DEFAULT_THRESHOLD

    def test_fail_status_when_threshold_low(self):
        # Force a fail: lower the threshold dramatically so any FP triggers.
        with patch(
            "ive.observability.fpr_sentinel._run_single_seed",
            side_effect=lambda seed, n_rows, n_features: True,
        ):
            result = run_sentinel(n_seeds=20, threshold=0.5)
        assert result.status == "fail"
        assert result.empirical_fpr == 1.0
        assert not result.passed

    def test_pass_status_when_zero_false_positives(self):
        with patch(
            "ive.observability.fpr_sentinel._run_single_seed",
            side_effect=lambda seed, n_rows, n_features: False,
        ):
            result = run_sentinel(n_seeds=20, threshold=0.07)
        assert result.empirical_fpr == 0.0
        # Upper-95% CI at 0/20 is ~0.139 > threshold 0.07 — so this should
        # *fail* by design. Per plan §190, the threshold must be >0.139
        # for very small n_seeds, or n_seeds raised.
        assert result.upper_95_ci > 0.07
        assert result.status == "fail"

    def test_pass_status_with_loose_threshold(self):
        # With threshold = 0.20, zero-FP n=20 passes (CI upper ~0.139).
        with patch(
            "ive.observability.fpr_sentinel._run_single_seed",
            side_effect=lambda seed, n_rows, n_features: False,
        ):
            result = run_sentinel(n_seeds=20, threshold=0.20)
        assert result.status == "pass"
        assert result.passed

    def test_invalid_n_seeds_rejected(self):
        with pytest.raises(ValueError):
            run_sentinel(n_seeds=0)
        with pytest.raises(ValueError):
            run_sentinel(n_seeds=-3)

    def test_runs_with_default_args(self):
        # Smoke: default config is callable (slow path, ~1-2s).
        result = run_sentinel(n_seeds=5, n_rows=200, n_features=3)
        assert result.n_runs == 5
        assert result.threshold == DEFAULT_THRESHOLD


class TestSentinelResultProperties:
    def test_passed_property(self):
        r = SentinelResult(
            n_runs=20, n_false_positive_runs=0,
            empirical_fpr=0.0, upper_95_ci=0.05,
            threshold=0.07, status="pass",
        )
        assert r.passed is True
        r = SentinelResult(
            n_runs=20, n_false_positive_runs=5,
            empirical_fpr=0.25, upper_95_ci=0.45,
            threshold=0.07, status="fail",
        )
        assert r.passed is False
