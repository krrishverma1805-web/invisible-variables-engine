"""Wave 3 audit regressions — flaws caught during rigorous testing.

Locks in the Wave 3 audit fixes so they don't silently regress.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.construction.bca_bootstrap import (
    BCA_MIN_N,
    bca_confidence_interval,
)
from ive.construction.bootstrap_validator import BootstrapValidator
from ive.construction.stability_calibration import min_presence_rate

pytestmark = pytest.mark.unit


# ── Non-finite point_estimate fallback ─────────────────────────────────────


class TestBcaNonFinitePointEstimate:
    """A NaN/inf point_estimate would otherwise silently produce a
    one-point degenerate CI. The fix substitutes the bootstrap median
    so BCa stays viable."""

    def test_nan_point_estimate_substitutes_median(self):
        rng = np.random.default_rng(0)
        boot = rng.normal(0, 1, 200)
        sample = rng.standard_normal(200)
        ci = bca_confidence_interval(
            boot,
            sample_data=sample,
            sample_statistic=lambda a: float(np.mean(a)),
            point_estimate=float("nan"),
        )
        # CI must span a real interval, not collapse to a point.
        assert ci.lower < ci.upper
        assert np.isfinite(ci.lower) and np.isfinite(ci.upper)

    def test_inf_point_estimate_substitutes_median(self):
        rng = np.random.default_rng(1)
        boot = rng.normal(0, 1, 200)
        sample = rng.standard_normal(200)
        ci = bca_confidence_interval(
            boot,
            sample_data=sample,
            sample_statistic=lambda a: float(np.mean(a)),
            point_estimate=float("inf"),
        )
        assert ci.lower < ci.upper

    def test_neg_inf_point_estimate(self):
        rng = np.random.default_rng(2)
        boot = rng.normal(0, 1, 200)
        sample = rng.standard_normal(200)
        ci = bca_confidence_interval(
            boot,
            sample_data=sample,
            sample_statistic=lambda a: float(np.mean(a)),
            point_estimate=float("-inf"),
        )
        assert ci.lower < ci.upper


class TestBcaBoundary:
    def test_n_equal_threshold_uses_bca(self):
        rng = np.random.default_rng(3)
        boot = rng.normal(0, 1, 200)
        sample = rng.standard_normal(BCA_MIN_N)
        ci = bca_confidence_interval(
            boot,
            sample_data=sample,
            sample_statistic=lambda a: float(np.mean(a)),
            point_estimate=0.0,
        )
        assert ci.method == "bca"

    def test_n_below_threshold_uses_percentile(self):
        rng = np.random.default_rng(4)
        boot = rng.normal(0, 1, 200)
        sample = rng.standard_normal(BCA_MIN_N - 1)
        ci = bca_confidence_interval(
            boot,
            sample_data=sample,
            sample_statistic=lambda a: float(np.mean(a)),
            point_estimate=0.0,
        )
        assert ci.method == "percentile"


class TestValidatorEdgeCases:
    """The per-resample effect-size proxy must produce a defined CI even
    when the construction rule is degenerate (all-active or all-inactive)."""

    def _df(self, n: int = 150) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "feature_a": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )

    def _candidate(self, lower: float, upper: float):
        # Use the real numeric_bin rule shape that apply_construction_rule
        # supports — interval membership.
        return {
            "name": "edge",
            "pattern_type": "subgroup",
            "construction_rule": {
                "column": "feature_a",
                "subgroup_type": "numeric_bin",
                "lower": lower,
                "upper": upper,
                "left_closed": True,
                "right_closed": True,
                "value": f"({lower:.2f}, {upper:.2f}]",
            },
            "stability_score": 0.0,
            "bootstrap_presence_rate": 0.0,
        }

    def test_all_inactive_scores_returns_degenerate_ci(self):
        df = self._df()
        # bin entirely outside the data range → all-inactive
        cand = self._candidate(lower=100.0, upper=200.0)
        BootstrapValidator(mode="demo").validate(
            candidates=[cand], original_X=df, n_iterations=10
        )
        # No active subgroup → can't compute Cohen's d → degenerate.
        assert cand["effect_size_ci_method"] in ("degenerate", "percentile")
        assert cand["status"] == "rejected"

    def test_all_active_scores_returns_degenerate_ci(self):
        df = self._df()
        # bin spanning the entire data range → all-active
        cand = self._candidate(lower=-100.0, upper=100.0)
        BootstrapValidator(mode="demo").validate(
            candidates=[cand], original_X=df, n_iterations=10
        )
        assert cand["effect_size_ci_method"] in ("degenerate", "percentile")

    def test_normal_subgroup_returns_real_ci(self):
        df = self._df()
        # bin covering ~half the data → real Cohen's d each resample
        cand = self._candidate(lower=0.0, upper=10.0)
        BootstrapValidator(mode="demo").validate(
            candidates=[cand], original_X=df, n_iterations=20
        )
        # We should get a non-degenerate CI (or, on noise, a percentile).
        assert cand["effect_size_ci_method"] in ("bca", "percentile")


# ── Calibration cross-cutting ──────────────────────────────────────────────


class TestCalibrationProblemTypeIsolation:
    """The calibration table is keyed by problem_type; a regression-only
    table must not return values for binary lookups."""

    def test_regression_only_table_misses_binary(self):
        table = {
            "schema_version": "v2",
            "config_grid": {
                "n_rows": [200, 1000],
                "modes": ["demo", "production"],
                "problem_types": ["regression"],
            },
            "results": {
                "1000|demo|regression": 0.55,
                "1000|production|regression": 0.65,
            },
        }
        # Binary lookup falls back to legacy fixed.
        rate = min_presence_rate(
            1000, "production", problem_type="binary", strategy="table", table=table
        )
        # Legacy fixed for production is 0.7
        assert rate == 0.7

    def test_binary_lookup_uses_binary_value(self):
        table = {
            "schema_version": "v2",
            "config_grid": {"n_rows": [1000]},
            "results": {
                "1000|production|regression": 0.65,
                "1000|production|binary": 0.55,
            },
        }
        rate = min_presence_rate(
            1000, "production", problem_type="binary", strategy="table", table=table
        )
        assert rate == 0.55


class TestImportSafety:
    """No cycle between bca_bootstrap and stability_calibration."""

    def test_independent_imports(self):
        # Both should import successfully in either order.
        from importlib import reload

        from ive.construction import bca_bootstrap, stability_calibration

        reload(bca_bootstrap)
        reload(stability_calibration)
        # Constants visible from both surfaces.
        assert stability_calibration.BCA_MIN_N == bca_bootstrap.BCA_MIN_N
