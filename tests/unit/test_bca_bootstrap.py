"""Unit tests for ive.construction.bca_bootstrap.bca_confidence_interval."""

from __future__ import annotations

import numpy as np
import pytest

from ive.construction.bca_bootstrap import (
    BCA_MIN_N,
    bca_confidence_interval,
)

pytestmark = pytest.mark.unit


class TestBasicShape:
    def test_returns_finite_lower_upper(self):
        boot = np.random.default_rng(0).normal(loc=2.0, scale=0.5, size=500)
        ci = bca_confidence_interval(boot)
        assert ci.lower < ci.upper
        assert np.isfinite(ci.lower)
        assert np.isfinite(ci.upper)

    def test_method_is_percentile_when_no_sample_data(self):
        # Without (sample_data, sample_statistic) we can't compute a, so
        # fall back to percentile regardless of N.
        boot = np.random.default_rng(0).normal(0, 1, size=300)
        ci = bca_confidence_interval(boot)
        assert ci.method == "percentile"

    def test_method_is_bca_with_jackknife_data(self):
        rng = np.random.default_rng(42)
        sample = rng.standard_normal(150)

        def stat(arr):
            return float(np.mean(arr))

        # Synthetic bootstrap distribution from the sample.
        boot = rng.standard_normal(500)
        ci = bca_confidence_interval(
            boot,
            sample_data=sample,
            sample_statistic=stat,
            point_estimate=float(np.mean(sample)),
        )
        assert ci.method == "bca"

    def test_n_below_threshold_falls_back_to_percentile(self):
        # N below BCA_MIN_N → percentile regardless of arguments.
        rng = np.random.default_rng(1)
        sample = rng.standard_normal(BCA_MIN_N - 1)

        def stat(arr):
            return float(np.mean(arr))

        ci = bca_confidence_interval(
            rng.standard_normal(200),
            sample_data=sample,
            sample_statistic=stat,
        )
        assert ci.method == "percentile"


class TestEdgeCases:
    def test_empty_distribution_returns_degenerate(self):
        ci = bca_confidence_interval(np.array([]))
        assert ci.method == "degenerate"
        assert np.isnan(ci.lower)
        assert np.isnan(ci.upper)

    def test_all_nan_returns_degenerate(self):
        ci = bca_confidence_interval(np.array([np.nan, np.nan]))
        assert ci.method == "degenerate"
        assert ci.n_used == 0

    def test_constant_distribution(self):
        # All-equal bootstrap stats → CI collapses to the constant.
        boot = np.full(500, 1.5)
        ci = bca_confidence_interval(boot)
        assert ci.method == "percentile"
        assert ci.lower == 1.5
        assert ci.upper == 1.5


class TestCoverage:
    def test_normal_ci_covers_true_mean(self):
        rng = np.random.default_rng(123)
        true_mean = 0.4
        # Empirical bootstrap distribution: 1000 resampled means of
        # n=200 samples from N(true_mean, 1).
        boot = []
        sample = rng.normal(true_mean, 1.0, size=200)
        for _ in range(1000):
            idx = rng.integers(0, 200, size=200)
            boot.append(float(np.mean(sample[idx])))
        ci = bca_confidence_interval(np.array(boot))
        assert ci.lower < true_mean < ci.upper
