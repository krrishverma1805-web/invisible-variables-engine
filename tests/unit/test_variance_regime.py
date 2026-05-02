"""Unit tests for ive.detection.variance_regime.VarianceRegimeDetector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.detection.variance_regime import (
    VarianceRegimeDetector,
    VarianceRegimePattern,
)

pytestmark = pytest.mark.unit


class TestBasicShape:
    def test_homoscedastic_data_yields_no_regime(self):
        # constant-variance noise → LR ≈ 0 → no detection
        rng = np.random.default_rng(0)
        n = 500
        X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        residuals = rng.standard_normal(n)
        detector = VarianceRegimeDetector(min_spread_ratio=2.0, lr_alpha=0.05)
        out = detector.detect(X, residuals)
        assert out == []

    def test_heteroscedastic_data_detected(self):
        # σ(x) = x → variance grows with feature → strong LR + spread.
        rng = np.random.default_rng(42)
        n = 800
        x = rng.uniform(0.1, 10, n)
        residuals = rng.standard_normal(n) * x
        X = pd.DataFrame({"x": x})
        detector = VarianceRegimeDetector(min_spread_ratio=2.0, lr_alpha=0.05)
        out = detector.detect(X, residuals)
        assert len(out) >= 1
        assert out[0].feature == "x"
        assert out[0].spread_ratio >= 2.0
        assert out[0].p_value < 0.05

    def test_returns_pattern_dataclass(self):
        rng = np.random.default_rng(1)
        n = 300
        x = rng.uniform(0, 1, n)
        residuals = rng.standard_normal(n) * (1 + 5 * x)
        out = VarianceRegimeDetector().detect(pd.DataFrame({"x": x}), residuals)
        assert all(isinstance(p, VarianceRegimePattern) for p in out)
        if out:
            assert out[0].pattern_type == "variance_regime"


class TestEdgeCases:
    def test_too_few_samples_returns_empty(self):
        detector = VarianceRegimeDetector()
        X = pd.DataFrame({"x": [1.0, 2.0]})
        residuals = np.array([0.1, 0.2])
        assert detector.detect(X, residuals) == []

    def test_shape_mismatch_raises(self):
        detector = VarianceRegimeDetector()
        X = pd.DataFrame({"x": np.arange(100)})
        residuals = np.zeros(50)
        with pytest.raises(ValueError, match="shape mismatch"):
            detector.detect(X, residuals)

    def test_skips_non_numeric_features(self):
        rng = np.random.default_rng(7)
        n = 300
        X = pd.DataFrame(
            {
                "category": rng.choice(["a", "b"], n),
                "x": rng.uniform(0, 1, n),
            }
        )
        residuals = rng.standard_normal(n) * (1 + 5 * X["x"].to_numpy())
        # Pass feature_names=None so the detector picks numeric ones.
        out = VarianceRegimeDetector().detect(X, residuals)
        # Only "x" should be tested — "category" filtered out.
        assert all(p.feature != "category" for p in out)

    def test_explicit_feature_names_subset(self):
        rng = np.random.default_rng(3)
        n = 400
        X = pd.DataFrame(
            {
                "noise": rng.standard_normal(n),
                "het": rng.uniform(0.5, 5, n),
            }
        )
        residuals = rng.standard_normal(n) * X["het"].to_numpy()
        # Only test "noise" → no regime expected even though "het" would
        # qualify.
        out = VarianceRegimeDetector().detect(X, residuals, feature_names=["noise"])
        assert all(p.feature == "noise" for p in out)
        # Likely empty since "noise" isn't actually heteroscedastic.

    def test_all_nan_residuals_returns_empty(self):
        n = 300
        X = pd.DataFrame({"x": np.arange(n, dtype=float)})
        residuals = np.full(n, np.nan)
        assert VarianceRegimeDetector().detect(X, residuals) == []

    def test_some_nan_residuals_filtered(self):
        rng = np.random.default_rng(8)
        n = 500
        x = rng.uniform(0.1, 10, n)
        residuals = rng.standard_normal(n) * x
        residuals[:50] = np.nan  # head NaN like TimeSeriesSplit
        out = VarianceRegimeDetector().detect(
            pd.DataFrame({"x": x}), residuals
        )
        # Should still find the regime on the surviving 450 rows.
        assert any(p.feature == "x" for p in out)


class TestLrSignificance:
    def test_high_lr_alpha_passes_more_features(self):
        rng = np.random.default_rng(11)
        n = 400
        x = rng.uniform(0.5, 5, n)
        residuals = rng.standard_normal(n) * (1 + 0.3 * x)  # weak heteroscedasticity
        strict = VarianceRegimeDetector(min_spread_ratio=1.0, lr_alpha=0.001)
        loose = VarianceRegimeDetector(min_spread_ratio=1.0, lr_alpha=0.5)
        n_strict = len(strict.detect(pd.DataFrame({"x": x}), residuals))
        n_loose = len(loose.detect(pd.DataFrame({"x": x}), residuals))
        assert n_loose >= n_strict


class TestSorting:
    def test_results_sorted_by_lr_descending(self):
        rng = np.random.default_rng(13)
        n = 500
        X = pd.DataFrame(
            {
                "strong": rng.uniform(0.1, 10, n),
                "moderate": rng.uniform(0.5, 5, n),
            }
        )
        # Strong heteroscedasticity along "strong", weaker along "moderate".
        residuals = rng.standard_normal(n) * X["strong"].to_numpy()
        out = VarianceRegimeDetector(min_spread_ratio=1.5).detect(X, residuals)
        if len(out) >= 2:
            assert out[0].lr_statistic >= out[1].lr_statistic
