"""Unit tests for the Phase B8 variance-regime synthesizer + applicator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.construction.variable_synthesizer import (
    VariableSynthesizer,
    apply_construction_rule,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "x": rng.uniform(0, 10, 200),
            "y": rng.standard_normal(200),
        }
    )


class TestSynthesize:
    def test_creates_candidate_for_variance_regime(self, df):
        pattern = {
            "pattern_type": "variance_regime",
            "feature": "x",
            "effect_size": 3.0,
            "p_value": 0.001,
            "spread_ratio": 3.0,
            "lr_statistic": 25.0,
            "sample_count": len(df),
        }
        out = VariableSynthesizer().synthesize([pattern], df)
        assert len(out) == 1
        cand = out[0]
        assert cand["pattern_type"] == "variance_regime"
        assert "construction_rule" in cand
        assert cand["construction_rule"]["feature"] == "x"
        assert "high_variance_threshold" in cand["construction_rule"]

    def test_skip_missing_feature(self, df):
        pattern = {
            "pattern_type": "variance_regime",
            "feature": "doesnotexist",
        }
        out = VariableSynthesizer().synthesize([pattern], df)
        assert out == []

    def test_default_threshold_is_median(self, df):
        pattern = {
            "pattern_type": "variance_regime",
            "feature": "x",
        }
        out = VariableSynthesizer().synthesize([pattern], df)
        assert pytest.approx(out[0]["construction_rule"]["high_variance_threshold"]) == float(
            df["x"].median()
        )

    def test_explicit_threshold_preserved(self, df):
        pattern = {
            "pattern_type": "variance_regime",
            "feature": "x",
            "high_variance_threshold": 7.5,
        }
        out = VariableSynthesizer().synthesize([pattern], df)
        assert out[0]["construction_rule"]["high_variance_threshold"] == 7.5

    def test_supports_mixed_pattern_list(self, df):
        # variance_regime alongside subgroup pattern.
        patterns = [
            {
                "pattern_type": "variance_regime",
                "feature": "x",
                "effect_size": 2.0,
                "p_value": 0.01,
            },
            {
                "pattern_type": "subgroup",
                "column_name": "x",
                "bin_value": "(5.0, 10.0]",
                "effect_size": 0.4,
                "p_value": 0.001,
            },
        ]
        out = VariableSynthesizer().synthesize(patterns, df)
        types = [c["pattern_type"] for c in out]
        assert "variance_regime" in types


class TestApplyConstructionRule:
    def test_applies_high_variance_threshold(self, df):
        scores = apply_construction_rule(
            {"feature": "x", "high_variance_threshold": 5.0},
            "variance_regime",
            df,
        )
        # Indicator: x >= 5
        expected = (df["x"] >= 5.0).astype(np.float64).to_numpy()
        assert np.array_equal(scores, expected)

    def test_handles_missing_feature(self, df):
        scores = apply_construction_rule(
            {"feature": "missing", "high_variance_threshold": 0.0},
            "variance_regime",
            df,
        )
        # All-zero fallback when feature is absent.
        assert np.all(scores == 0.0)

    def test_handles_missing_threshold(self, df):
        scores = apply_construction_rule(
            {"feature": "x"},
            "variance_regime",
            df,
        )
        # No threshold → all-zero fallback.
        assert np.all(scores == 0.0)

    def test_nan_values_treated_as_below_threshold(self, df):
        df_nan = df.copy()
        df_nan.loc[:5, "x"] = np.nan
        scores = apply_construction_rule(
            {"feature": "x", "high_variance_threshold": 5.0},
            "variance_regime",
            df_nan,
        )
        # NaN rows must be 0 (not 1).
        assert (scores[:6] == 0.0).all()

    def test_compatible_with_bootstrap_resample(self, df):
        # The applicator must work on a bootstrap resample (pandas may
        # change index but the rule still applies).
        sampled = df.sample(frac=1.0, replace=True, random_state=0)
        scores = apply_construction_rule(
            {"feature": "x", "high_variance_threshold": 5.0},
            "variance_regime",
            sampled,
        )
        assert scores.shape == (len(sampled),)
        # Scores are 0/1.
        assert set(np.unique(scores).tolist()) <= {0.0, 1.0}
