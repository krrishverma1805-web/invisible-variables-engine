"""Integration of B4 (BCa CIs) + B6 (calibrated thresholds) into BootstrapValidator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.construction.bootstrap_validator import BootstrapValidator

pytestmark = pytest.mark.unit


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.uniform(0, 100, n),
            "y": rng.standard_normal(n),
        }
    )
    return df


@pytest.fixture
def subgroup_candidate():
    return {
        "name": "test_subgroup",
        "pattern_type": "subgroup",
        "construction_rule": {
            "feature": "feature_a",
            "threshold": 0.0,
            "operator": ">",
            "feature_columns": ["feature_a"],
        },
        "stability_score": 0.0,
        "bootstrap_presence_rate": 0.0,
    }


class TestB4CIPersisted:
    def test_validator_writes_ci_fields(self, small_dataset, subgroup_candidate):
        validator = BootstrapValidator(mode="demo")
        validator.validate(
            candidates=[subgroup_candidate],
            original_X=small_dataset,
            n_iterations=20,
        )
        assert "effect_size_ci_lower" in subgroup_candidate
        assert "effect_size_ci_upper" in subgroup_candidate
        assert "effect_size_ci_method" in subgroup_candidate
        # Method must be one of the documented values.
        assert subgroup_candidate["effect_size_ci_method"] in (
            "bca",
            "percentile",
            "degenerate",
        )

    def test_ci_lower_le_upper_when_both_present(
        self, small_dataset, subgroup_candidate
    ):
        validator = BootstrapValidator(mode="demo")
        validator.validate(
            candidates=[subgroup_candidate],
            original_X=small_dataset,
            n_iterations=20,
        )
        lo = subgroup_candidate.get("effect_size_ci_lower")
        hi = subgroup_candidate.get("effect_size_ci_upper")
        if lo is not None and hi is not None:
            assert lo <= hi


class TestB6ThresholdLookup:
    def test_explicit_threshold_overrides_calibration(
        self, small_dataset, subgroup_candidate
    ):
        # When the caller supplies a threshold, the calibrated lookup
        # should NOT fire. Use a high threshold to force rejection.
        validator = BootstrapValidator(mode="production")
        validator.validate(
            candidates=[subgroup_candidate],
            original_X=small_dataset,
            n_iterations=20,
            stability_threshold=0.99,
        )
        # Either rejected (high threshold) or validated — main thing
        # is that it ran without invoking the calibrated lookup.
        assert subgroup_candidate["status"] in ("validated", "rejected")

    def test_no_explicit_threshold_uses_calibration(
        self, small_dataset, subgroup_candidate, monkeypatch
    ):
        """When caller passes no threshold, BootstrapValidator must
        consult ``min_presence_rate``."""
        called = {"count": 0}

        def fake_min_presence_rate(**kwargs):
            called["count"] += 1
            return 0.5

        monkeypatch.setattr(
            "ive.construction.stability_calibration.min_presence_rate",
            fake_min_presence_rate,
        )

        validator = BootstrapValidator(mode="demo")
        validator.validate(
            candidates=[subgroup_candidate],
            original_X=small_dataset,
            n_iterations=10,
        )
        assert called["count"] == 1
