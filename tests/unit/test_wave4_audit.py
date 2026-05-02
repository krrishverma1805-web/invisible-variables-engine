"""Wave 4 audit regressions — flaws caught during rigorous testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.detection.interaction_features import (
    InteractionPair,
    synthesize_interaction_features,
)
from ive.detection.variance_regime import (
    VarianceRegimeDetector,
    _gaussian_loglik,
)

pytestmark = pytest.mark.unit


# ── B8 _gaussian_loglik degenerate-input semantics ─────────────────────────


class TestGaussianLoglikDegenerate:
    """A degenerate input must return NaN, not 0.0. The LR computation
    relies on this — returning 0.0 would falsely produce positive LR
    against any negative loglik_null (which is always the case)."""

    def test_constant_input_returns_nan(self):
        out = _gaussian_loglik(np.full(100, 1.0))
        assert np.isnan(out)

    def test_single_sample_returns_nan(self):
        out = _gaussian_loglik(np.array([1.0]))
        assert np.isnan(out)

    def test_empty_returns_nan(self):
        out = _gaussian_loglik(np.array([]))
        assert np.isnan(out)

    def test_normal_input_returns_finite_negative(self):
        out = _gaussian_loglik(np.random.default_rng(0).standard_normal(100))
        assert np.isfinite(out)
        # Gaussian loglik on 100 samples with sigma~1 is around -142.
        assert out < 0


class TestVarianceRegimeNoFalseRegimeOnDegenerateNull:
    """When |residuals| is constant (e.g. all-zero), the null loglik is
    NaN; the detector must skip rather than emit spurious patterns."""

    def test_all_zero_residuals_returns_empty(self):
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        # All residuals are exactly zero → |residuals| has zero variance
        # → loglik_null is NaN.
        residuals = np.zeros(n)
        out = VarianceRegimeDetector().detect(X, residuals)
        assert out == []

    def test_constant_nonzero_residuals_returns_empty(self):
        rng = np.random.default_rng(1)
        n = 300
        X = pd.DataFrame({"x": rng.uniform(0, 10, n)})
        residuals = np.full(n, 0.7)
        out = VarianceRegimeDetector().detect(X, residuals)
        assert out == []


class TestVarianceRegimeFiniteLoglikAlt:
    """If a fitted feature happens to perfectly fit |residuals| (degenerate
    alt loglik), skip cleanly rather than emit a positive LR."""

    def test_feature_exactly_predicts_abs_residuals(self):
        rng = np.random.default_rng(2)
        n = 300
        x = rng.uniform(0.1, 10, n)
        # Constructed so |residuals| ≈ x exactly (degenerate alt fit).
        residuals = x.copy()  # |residuals| = x → perfect linear fit
        X = pd.DataFrame({"x": x})
        out = VarianceRegimeDetector(min_spread_ratio=1.0).detect(X, residuals)
        # Either empty (caught by NaN guard) or a finite-LR detection —
        # never a falsely-infinite or negative LR.
        for p in out:
            assert np.isfinite(p.lr_statistic)
            assert p.lr_statistic > 0


# ── B7 dtype handling ──────────────────────────────────────────────────────


class TestB7DtypeRobustness:
    def test_bool_dtype_synthesizes_features(self):
        df = pd.DataFrame(
            {
                "flag": [True, False, True, False] * 25,
                "x": np.random.default_rng(0).standard_normal(100),
            }
        )
        out = synthesize_interaction_features(
            df, [InteractionPair("flag", "x", 0.5)]
        )
        new_cols = [c for c in out.columns if c.startswith("__ix__")]
        # bool×float is numeric → 3 columns synthesized.
        assert len(new_cols) == 3

    def test_int_dtype_synthesizes_features(self):
        df = pd.DataFrame(
            {
                "n": np.arange(100),
                "x": np.random.default_rng(0).standard_normal(100),
            }
        )
        out = synthesize_interaction_features(
            df, [InteractionPair("n", "x", 0.5)]
        )
        new_cols = [c for c in out.columns if c.startswith("__ix__")]
        assert len(new_cols) == 3

    def test_string_numeric_coerces(self):
        df = pd.DataFrame(
            {
                "a": ["1.0", "2.0", "3.0"] * 30,
                "b": np.random.default_rng(0).standard_normal(90),
            }
        )
        out = synthesize_interaction_features(
            df, [InteractionPair("a", "b", 0.5)]
        )
        # to_numeric(errors='coerce') turns these into floats successfully.
        new_cols = [c for c in out.columns if c.startswith("__ix__")]
        assert len(new_cols) == 3

    def test_partial_nan_feature_filled_with_zero(self):
        n = 100
        a_col = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0] * 20)
        df = pd.DataFrame(
            {"a": a_col, "b": np.random.default_rng(0).standard_normal(n)}
        )
        out = synthesize_interaction_features(
            df, [InteractionPair("a", "b", 0.5)]
        )
        # Product NaN-fills to 0, indicators NaN-fill to 0.
        for col in [
            "__ix__a__x__b",
            "__ix__a__hh__b",
            "__ix__a__xor__b",
        ]:
            assert out[col].isna().sum() == 0
