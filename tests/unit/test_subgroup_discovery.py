"""Unit tests for subgroup discovery module."""

from __future__ import annotations

import numpy as np

from ive.detection.subgroup_discovery import SubgroupDiscoverer


class TestSubgroupDiscoverer:
    def test_discover_returns_list(self, simple_regression_df, residuals_array) -> None:
        """discover() should return a list (possibly empty until implemented)."""
        discoverer = SubgroupDiscoverer()
        result = discoverer.discover(simple_regression_df, residuals_array, ["x1", "x2"])
        assert isinstance(result, list)

    def test_compute_wracc_respects_min_coverage(self, residuals_array) -> None:
        """WRAcc should return 0 for masks below min_coverage threshold."""
        discoverer = SubgroupDiscoverer(min_coverage=0.5)
        # Mask with only 10% coverage
        mask = np.zeros(len(residuals_array), dtype=bool)
        mask[: int(len(residuals_array) * 0.1)] = True
        wracc = discoverer._compute_wracc(mask, residuals_array)
        assert wracc == 0.0

    def test_cohens_d_opposite_groups(self) -> None:
        """Cohen's d should be non-zero for groups with different means."""
        group1 = np.array([1.0, 1.5, 2.0, 1.8])
        group2 = np.array([5.0, 5.5, 6.0, 5.8])
        d = SubgroupDiscoverer()._cohens_d(group1, group2)
        assert d < 0  # group1 mean < group2 mean

    def test_cohens_d_identical_groups(self) -> None:
        """Cohen's d should be 0 for identical groups."""
        group = np.array([2.0, 2.0, 2.0, 2.0])
        d = SubgroupDiscoverer()._cohens_d(group, group)
        assert d == 0.0

    def test_discover_top_k_limit(self, simple_regression_df, residuals_array) -> None:
        """discover() should return at most top_k patterns."""
        discoverer = SubgroupDiscoverer()
        result = discoverer.discover(simple_regression_df, residuals_array, ["x1", "x2"], top_k=3)
        assert len(result) <= 3
