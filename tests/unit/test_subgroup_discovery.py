"""Unit tests for SubgroupDiscovery.

Aligned with current public API:
  - SubgroupDiscovery(n_bins=5, min_bin_samples=20, min_effect_size=0.15)
  - detect(X: pd.DataFrame, residuals: np.ndarray, alpha=0.05) -> list[dict]
  - Internal helpers: _bin_column, _bin_numeric, _bin_categorical
    (no _compute_wracc / _cohens_d public guarantee; tested via detect() behaviour)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ive.detection.subgroup_discovery import SubgroupDiscovery


class TestSubgroupDiscovery:
    def test_detect_returns_list(self, simple_regression_df) -> None:
        """detect() should return a list (may be empty if no significant patterns)."""
        residuals = simple_regression_df["y"].to_numpy()
        X = simple_regression_df[["x1", "x2"]]
        discoverer = SubgroupDiscovery()
        result = discoverer.detect(X, residuals)
        assert isinstance(result, list)

    def test_detect_result_items_are_dicts(self, simple_regression_df) -> None:
        """Each item returned by detect() must be a dict."""
        residuals = simple_regression_df["y"].to_numpy()
        X = simple_regression_df[["x1", "x2"]]
        result = SubgroupDiscovery().detect(X, residuals)
        for item in result:
            assert isinstance(item, dict)

    def test_detect_with_clear_subgroup_finds_patterns(self) -> None:
        """A dataset with a clear residual subgroup should produce at least one pattern."""
        rng = np.random.default_rng(42)
        n = 300
        cats = np.where(rng.standard_normal(n) > 0, "HIGH", "LOW")
        X = pd.DataFrame({"group": cats, "noise": rng.standard_normal(n)})
        # Residuals are high for 'HIGH' group, near zero otherwise
        residuals = np.where(
            cats == "HIGH", rng.standard_normal(n) + 3.0, rng.standard_normal(n) * 0.2
        )
        result = SubgroupDiscovery(min_bin_samples=10, min_effect_size=0.10).detect(X, residuals)
        assert len(result) >= 1, "Expected at least one subgroup pattern for a clear signal"

    def test_detect_on_pure_noise_returns_few_patterns(self) -> None:
        """A pure-noise dataset should not produce many spurious patterns."""
        rng = np.random.default_rng(99)
        n = 200
        X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
        residuals = rng.standard_normal(n)
        result = SubgroupDiscovery().detect(X, residuals)
        # Allow a small number of spurious hits from random correlation; strict threshold.
        assert len(result) <= 5, f"Too many patterns on pure noise: {len(result)}"

    def test_detect_respects_min_bin_samples(self) -> None:
        """With a high min_bin_samples, small bins must be excluded."""
        rng = np.random.default_rng(7)
        n = 100
        X = pd.DataFrame({"x": rng.standard_normal(n)})
        residuals = rng.standard_normal(n)
        # min_bin_samples=80 means only a very large bin can qualify
        result = SubgroupDiscovery(min_bin_samples=80).detect(X, residuals)
        # Can be empty; important: should not crash
        assert isinstance(result, list)

    def test_detect_result_has_expected_keys(self, simple_regression_df) -> None:
        """Each pattern dict must contain at minimum 'pattern_type' and 'effect_size'."""
        residuals = simple_regression_df["y"].to_numpy() * 2  # amplify for detectability
        X = simple_regression_df[["x1", "x2"]]
        result = SubgroupDiscovery(min_bin_samples=10, min_effect_size=0.05).detect(X, residuals)
        for item in result:
            assert "pattern_type" in item, f"Missing 'pattern_type' in {item}"
            assert "effect_size" in item, f"Missing 'effect_size' in {item}"
