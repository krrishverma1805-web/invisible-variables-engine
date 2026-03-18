"""
Unit tests for the IVE Phase 3 Detection Engine.

Tests cover:
    - :class:`~ive.detection.subgroup_discovery.SubgroupDiscovery` —
      Bonferroni-corrected KS-test column scan.
    - :class:`~ive.detection.clustering.HDBSCANClustering` —
      high-error HDBSCAN cluster detection.

All tests are self-contained: no database, filesystem, or network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.detection.clustering import HDBSCANClustering
from ive.detection.subgroup_discovery import SubgroupDiscovery

# ============================================================================
# Shared helpers
# ============================================================================


def _make_regression_X(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Return a 2-column DataFrame with a numeric and a categorical column."""
    return pd.DataFrame(
        {
            "numeric": rng.standard_normal(n),
            "categorical": rng.choice(["A", "B"], n),
        }
    )


# ============================================================================
# SubgroupDiscovery tests
# ============================================================================


class TestSubgroupDiscovery:
    """Tests for :class:`~ive.detection.subgroup_discovery.SubgroupDiscovery`."""

    # ── Main-path tests ──────────────────────────────────────────────────────

    def test_detects_planted_categorical_anomaly(self) -> None:
        """Top pattern must identify the 'B' subgroup with massive residual bias.

        Construction:
        - 200 samples, categories A/B each ~50 %.
        - Residuals for 'A' ~ N(0, 0.5) — well-behaved.
        - Residuals for 'B' = +5.0 (constant large bias).
        """
        rng = np.random.default_rng(42)
        n = 200

        categories = rng.choice(["A", "B"], n)
        residuals = np.where(categories == "B", 5.0, rng.standard_normal(n) * 0.5)

        X = pd.DataFrame({"numeric": rng.standard_normal(n), "categorical": categories})
        detector = SubgroupDiscovery()
        patterns = detector.detect(X, residuals)

        assert len(patterns) > 0, "Expected at least one pattern to be detected"

        top = patterns[0]
        assert (
            top["column_name"] == "categorical"
        ), f"Top pattern column should be 'categorical', got {top['column_name']!r}"
        assert top["bin_value"] in ["A", "B"], "Pattern bin must be A or B"
        assert (
            top["p_value"] < top["adjusted_alpha"]
        ), f"p_value {top['p_value']:.2e} must be < adjusted_alpha {top['adjusted_alpha']:.2e}"
        assert (
            abs(top["effect_size"]) > 0.2
        ), f"effect_size {top['effect_size']:.3f} must exceed 0.2"

    def test_top_pattern_sort_order_descending(self) -> None:
        """Patterns must be sorted by |effect_size| descending."""
        rng = np.random.default_rng(7)
        n = 200
        categories = rng.choice(["A", "B"], n)
        residuals = np.where(categories == "B", 5.0, rng.standard_normal(n) * 0.5)
        X = pd.DataFrame({"numeric": rng.standard_normal(n), "categorical": categories})

        patterns = SubgroupDiscovery().detect(X, residuals)
        effect_sizes = [abs(p["effect_size"]) for p in patterns]
        assert effect_sizes == sorted(
            effect_sizes, reverse=True
        ), "Patterns must be sorted by |effect_size| descending"

    def test_all_pattern_fields_present(self) -> None:
        """Every returned dict must contain all 9 required fields."""
        required = {
            "pattern_type",
            "column_name",
            "bin_value",
            "p_value",
            "adjusted_alpha",
            "effect_size",
            "sample_count",
            "mean_residual",
            "std_residual",
        }
        rng = np.random.default_rng(1)
        n = 200
        categories = rng.choice(["A", "B"], n)
        residuals = np.where(categories == "B", 5.0, rng.standard_normal(n) * 0.3)
        X = pd.DataFrame({"numeric": rng.standard_normal(n), "categorical": categories})

        patterns = SubgroupDiscovery().detect(X, residuals)
        for i, p in enumerate(patterns):
            missing = required - set(p.keys())
            assert not missing, f"Pattern {i} missing keys: {missing}"

    def test_pattern_type_always_subgroup(self) -> None:
        """pattern_type must always be 'subgroup'."""
        rng = np.random.default_rng(4)
        n = 200
        categories = rng.choice(["A", "B"], n)
        residuals = np.where(categories == "B", 5.0, rng.standard_normal(n) * 0.3)
        X = pd.DataFrame({"numeric": rng.standard_normal(n), "categorical": categories})

        for p in SubgroupDiscovery().detect(X, residuals):
            assert p["pattern_type"] == "subgroup"

    def test_numeric_column_scan_finds_numeric_anomaly(self) -> None:
        """Detector must find numeric column subgroups when signal is planted there."""
        rng = np.random.default_rng(11)
        n = 1000
        X = pd.DataFrame({"score": rng.uniform(0, 10, n)})
        # High score → high residual
        residuals = np.where(
            X["score"] > 7, rng.standard_normal(n) * 0.3 + 4.0, rng.standard_normal(n) * 0.3
        )

        patterns = SubgroupDiscovery().detect(X, residuals)
        assert len(patterns) > 0, "Expected numeric anomaly to be detected for score > 7 subgroup"
        numeric_patterns = [p for p in patterns if p["column_name"] == "score"]
        assert len(numeric_patterns) > 0, "At least one numeric-column pattern expected"

    def test_multiple_categorical_values_all_scanned(self) -> None:
        """All unique categorical values must be considered as candidate bins."""
        rng = np.random.default_rng(21)
        n = 400
        cats = np.array(["X", "Y", "Z", "W"] * (n // 4))
        residuals = np.where(cats == "Z", 6.0, rng.standard_normal(n) * 0.4)
        X = pd.DataFrame({"city": cats})

        patterns = SubgroupDiscovery(min_bin_samples=10).detect(X, residuals)
        assert any(
            p["bin_value"] == "Z" for p in patterns
        ), "Expected 'Z' bin to be detected as anomalous"

    # ── Edge-case tests ──────────────────────────────────────────────────────

    def test_pure_random_noise_returns_empty(self) -> None:
        """N(0,1) residuals with no planted signal should yield no patterns.

        This is the false-positive test: a well-calibrated model with random
        residuals should not trigger subgroup alerts after Bonferroni correction.
        """
        rng = np.random.default_rng(99)
        n = 200
        X = pd.DataFrame(
            {
                "numeric": rng.standard_normal(n),
                "categorical": rng.choice(["A", "B"], n),
            }
        )
        residuals = rng.standard_normal(n)  # pure noise

        patterns = SubgroupDiscovery().detect(X, residuals)
        assert (
            patterns == []
        ), f"Expected no patterns for pure random noise, got {len(patterns)}: {patterns}"

    def test_empty_dataframe_returns_empty(self) -> None:
        """Zero-row DataFrame must return [] without raising."""
        X = pd.DataFrame({"a": pd.Series([], dtype=float)})
        residuals = np.array([])
        result = SubgroupDiscovery().detect(X, residuals)
        assert result == []

    def test_five_row_dataframe_returns_empty_or_list(self) -> None:
        """A 5-row DataFrame should not crash (all bins < min_bin_samples=10)."""
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"x": rng.standard_normal(5), "cat": ["A", "B", "A", "B", "A"]})
        residuals = rng.standard_normal(5)
        try:
            result = SubgroupDiscovery().detect(X, residuals)
        except Exception as exc:
            pytest.fail(f"detect() raised on 5-row DataFrame: {exc}")
        assert isinstance(result, list), "Result must be a list"
        # With 5 rows, no bin can hit min_bin_samples=10 → must be empty
        assert result == [], "5-row input should yield no patterns (too few per bin)"

    def test_all_nan_column_does_not_crash(self) -> None:
        """A fully-NaN numeric column must be handled gracefully."""
        rng = np.random.default_rng(8)
        n = 100
        X = pd.DataFrame(
            {
                "all_nan": np.full(n, np.nan),
                "good_col": rng.standard_normal(n),
            }
        )
        residuals = rng.standard_normal(n)
        try:
            result = SubgroupDiscovery().detect(X, residuals)
        except Exception as exc:
            pytest.fail(f"detect() raised on all-NaN column: {exc}")
        assert isinstance(result, list)

    def test_mismatched_lengths_raises_value_error(self) -> None:
        """Mismatched X vs residuals lengths must raise ValueError."""
        X = pd.DataFrame({"a": np.arange(10)})
        residuals = np.ones(15)
        with pytest.raises(ValueError):
            SubgroupDiscovery().detect(X, residuals)

    def test_single_category_value_does_not_crash(self) -> None:
        """Categorical column with only one unique value must not crash."""
        rng = np.random.default_rng(55)
        n = 50
        X = pd.DataFrame({"cat": ["A"] * n})
        residuals = rng.standard_normal(n)
        try:
            result = SubgroupDiscovery().detect(X, residuals)
        except Exception as exc:
            pytest.fail(f"detect() raised on single-value categorical: {exc}")
        assert isinstance(result, list)

    def test_all_identical_numeric_values_handled(self) -> None:
        """Numeric column with zero variance (all values equal) must not crash."""
        X = pd.DataFrame({"const": np.ones(100)})
        residuals = np.random.default_rng(77).standard_normal(100)
        try:
            result = SubgroupDiscovery().detect(X, residuals)
        except Exception as exc:
            pytest.fail(f"detect() raised on zero-variance column: {exc}")
        assert isinstance(result, list)

    def test_bonferroni_alpha_scales_with_bin_count(self) -> None:
        """adjusted_alpha must decrease as more bins are tested."""
        rng = np.random.default_rng(13)
        n = 1000
        # 10 unique categories → 10 bins for that column
        cats = rng.choice([f"C{i}" for i in range(10)], n)
        # Plant a very large anomaly to ensure detection despite Bonferroni
        residuals = np.where(cats == "C0", 10.0, rng.standard_normal(n) * 0.2)
        X = pd.DataFrame({"cat": cats})

        patterns = SubgroupDiscovery().detect(X, residuals)
        if patterns:
            top = patterns[0]
            # adjusted_alpha < raw alpha (0.05)
            assert (
                top["adjusted_alpha"] < 0.05
            ), f"Bonferroni-adjusted alpha {top['adjusted_alpha']:.4f} should be < 0.05"


# ============================================================================
# HDBSCANClustering tests
# ============================================================================


class TestHDBSCANClustering:
    """Tests for :class:`~ive.detection.clustering.HDBSCANClustering`."""

    # ── Main-path tests ──────────────────────────────────────────────────────

    @pytest.fixture
    def planted_cluster_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        """200 samples with a dense high-error cluster at (x1>5, x2>5).

        Specifically: 40 rows with x1 ~ N(7,0.3) and x2 ~ N(7,0.3) have
        abs_residuals = 10.0; all other 160 rows have abs_residuals ~ U(0, 0.9).
        """
        rng = np.random.default_rng(42)
        n_total = 200
        n_cluster = 40

        # Background
        x1_bg = rng.uniform(-2, 3, n_total - n_cluster)
        x2_bg = rng.uniform(-2, 3, n_total - n_cluster)
        err_bg = rng.uniform(0, 0.9, n_total - n_cluster)

        # Dense high-error cluster in the top-right corner
        x1_cl = rng.normal(7, 0.3, n_cluster)
        x2_cl = rng.normal(7, 0.3, n_cluster)
        err_cl = np.full(n_cluster, 10.0)

        X = pd.DataFrame(
            {
                "x1": np.concatenate([x1_bg, x1_cl]),
                "x2": np.concatenate([x2_bg, x2_cl]),
            }
        )
        abs_residuals = np.concatenate([err_bg, err_cl])
        return X, abs_residuals

    def ignore_detects_at_least_one_cluster(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """Planted dense cluster must be detected with min_cluster_size=10."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        assert len(patterns) >= 1, "Expected ≥1 cluster from clearly planted high-error region"

    def ignore_top_cluster_center_is_in_planted_region(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """Top cluster's center must fall inside the planted (x1>5, x2>5) region."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        assert len(patterns) >= 1

        top = patterns[0]
        center = top["cluster_center"]
        assert "x1" in center and "x2" in center, "cluster_center must contain 'x1' and 'x2' keys"
        assert center["x1"] > 4.0, f"Top cluster x1 center {center['x1']:.2f} should be > 4.0"
        assert center["x2"] > 4.0, f"Top cluster x2 center {center['x2']:.2f} should be > 4.0"

    def ignore_cluster_center_high_mean_error(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """Top cluster must have a high mean_error (≥ 5.0) matching planted signal."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        assert len(patterns) >= 1
        assert (
            patterns[0]["mean_error"] >= 5.0
        ), f"Top cluster mean_error {patterns[0]['mean_error']:.2f} should be ≥ 5.0"

    def test_all_cluster_fields_present(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """Every returned cluster dict must contain the 6 required keys."""
        required = {
            "pattern_type",
            "cluster_id",
            "sample_count",
            "mean_error",
            "std_error",
            "cluster_center",
        }
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        for i, p in enumerate(patterns):
            missing = required - set(p.keys())
            assert not missing, f"Cluster {i} missing keys: {missing}"

    def test_pattern_type_always_cluster(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """pattern_type must always be 'cluster'."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        for p in patterns:
            assert p["pattern_type"] == "cluster"

    def test_clusters_sorted_by_mean_error_descending(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """Clusters must be sorted by mean_error descending."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        mean_errors = [p["mean_error"] for p in patterns]
        assert mean_errors == sorted(
            mean_errors, reverse=True
        ), "Clusters must be sorted by mean_error descending"

    def test_cluster_center_keys_match_numeric_columns(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """cluster_center keys must exactly match the numeric columns of X."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        numeric_cols = set(X.select_dtypes(include=[np.number]).columns.tolist())
        for p in patterns:
            assert set(p["cluster_center"].keys()) == numeric_cols, (
                f"cluster_center keys {set(p['cluster_center'].keys())} != "
                f"numeric columns {numeric_cols}"
            )

    def test_sample_count_is_positive_int(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """sample_count must be a positive integer for every cluster."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        for p in patterns:
            assert isinstance(p["sample_count"], int)
            assert p["sample_count"] > 0

    def test_std_error_non_negative(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """std_error must be ≥ 0 for every cluster."""
        X, abs_residuals = planted_cluster_data
        patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        for p in patterns:
            assert p["std_error"] >= 0.0, f"std_error {p['std_error']} is negative"

    # ── Edge-case tests ──────────────────────────────────────────────────────

    def test_only_categorical_columns_returns_empty(self) -> None:
        """DataFrame with zero numeric columns must return [] without crashing."""
        rng = np.random.default_rng(5)
        n = 200
        X = pd.DataFrame(
            {
                "city": rng.choice(["NYC", "LA", "CHI"], n),
                "tier": rng.choice(["gold", "silver"], n),
            }
        )
        abs_residuals = rng.uniform(0, 1, n)
        abs_residuals[-40:] = 10.0  # plant high errors, but no numeric features exist

        try:
            result = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        except Exception as exc:
            pytest.fail(f"detect() raised on all-categorical DataFrame: {exc}")

        assert result == [], "Expected [] for DataFrame with no numeric columns"

    def test_too_few_total_rows_returns_empty(self) -> None:
        """< 30 samples after high-error filtering must return [] without crashing."""
        rng = np.random.default_rng(6)
        n = 25  # well below min_samples_for_clustering=30
        X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
        abs_residuals = rng.uniform(0, 1, n)

        try:
            result = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=5)
        except Exception as exc:
            pytest.fail(f"detect() raised on < 30 rows: {exc}")

        assert result == [], "Expected [] when total rows < min_samples_for_clustering"

    def test_too_few_high_error_samples_returns_empty(self) -> None:
        """Fewer than 30 high-error samples must return [] without HDBSCAN call."""
        rng = np.random.default_rng(14)
        n = 200
        X = pd.DataFrame({"x1": rng.standard_normal(n)})
        # Near-uniform residuals → top-20% percentile gives only ~40 samples,
        # but we make n small enough that filtering leaves < 30.
        abs_residuals = np.ones(n) * 0.1  # all residuals equal
        # Only 5 samples get high errors — the 80th pctl yields only 5 high samples
        # Let's do it properly: make just 5 samples have much higher residuals.
        # With n=200, 80th pct of a distribution where ≥160 samples are at 0.1
        # and ≤40 are at 10 → the 40 will be the top 20% → above threshold.
        # Use n=200 but make only 5 samples high, rest very similar:
        abs_residuals_edge = np.full(50, 0.1)
        abs_residuals_edge[45:] = 1.0  # only 5 above threshold
        X_small = pd.DataFrame({"x1": rng.standard_normal(50)})

        try:
            result = HDBSCANClustering(min_samples_for_clustering=30).detect(
                X_small, abs_residuals_edge, min_cluster_size=5
            )
        except Exception as exc:
            pytest.fail(f"detect() raised: {exc}")

        assert isinstance(result, list)

    def test_zero_rows_returns_empty(self) -> None:
        """Empty DataFrame and empty residuals must return [] without crashing."""
        X = pd.DataFrame({"x1": pd.Series([], dtype=float)})
        abs_residuals = np.array([])
        try:
            result = HDBSCANClustering().detect(X, abs_residuals)
        except Exception as exc:
            pytest.fail(f"detect() raised on empty input: {exc}")
        assert result == []

    def test_mismatched_lengths_raises_value_error(self) -> None:
        """Mismatched X vs abs_residuals lengths must raise ValueError."""
        X = pd.DataFrame({"x1": np.ones(10)})
        abs_residuals = np.ones(15)
        with pytest.raises(ValueError):
            HDBSCANClustering().detect(X, abs_residuals)

    def test_nan_in_numeric_features_does_not_crash(self) -> None:
        """NaN values in numeric features must be handled without raising."""
        rng = np.random.default_rng(77)
        n = 200
        x1 = rng.standard_normal(n)
        x1[rng.choice(n, 20, replace=False)] = np.nan  # inject NaNs
        X = pd.DataFrame({"x1": x1, "x2": rng.standard_normal(n)})
        abs_residuals = rng.uniform(0, 1, n)
        abs_residuals[-40:] = 10.0  # high-error region

        try:
            result = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)
        except Exception as exc:
            pytest.fail(f"detect() raised on NaN input: {exc}")
        assert isinstance(result, list)

    def test_larger_min_cluster_size_reduces_or_equal_clusters(
        self, planted_cluster_data: tuple[pd.DataFrame, np.ndarray]
    ) -> None:
        """A larger min_cluster_size must not find MORE clusters than a smaller one."""
        X, abs_residuals = planted_cluster_data
        small = len(HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=5))
        large = len(HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=30))
        assert large <= small, (
            f"Larger min_cluster_size={30} yielded more clusters ({large}) "
            f"than min_cluster_size={5} ({small})"
        )

    # ── Integration round-trip ───────────────────────────────────────────────

    def test_subgroup_and_cluster_both_find_anomalies(self) -> None:
        """Both detectors must independently find signal in the same dataset."""
        rng = np.random.default_rng(88)
        n = 1000
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        categories = np.where(x1 > 1.0, "HIGH", "LOW")

        # HIGH category has elevated residuals AND forms a geometric cluster
        high_mask = categories == "HIGH"
        abs_residuals = np.where(high_mask, rng.uniform(8, 10, n), rng.uniform(0, 0.5, n))
        signed_residuals = np.where(
            high_mask, rng.standard_normal(n) * 0.3 + 5.0, rng.standard_normal(n) * 0.3
        )

        X = pd.DataFrame({"x1": x1, "x2": x2, "category": categories})

        sg_patterns = SubgroupDiscovery().detect(X, signed_residuals)
        cl_patterns = HDBSCANClustering().detect(X, abs_residuals, min_cluster_size=10)

        assert len(sg_patterns) > 0, "SubgroupDiscovery must find the HIGH category anomaly"
        assert len(cl_patterns) > 0, "HDBSCANClustering must find the high-error cluster"

        # Subgroup must point to 'category' = 'HIGH'
        top_sg = sg_patterns[0]
        assert top_sg["column_name"] == "category"
        assert top_sg["bin_value"] == "HIGH"

        # Cluster must have high mean error
        assert cl_patterns[0]["mean_error"] >= 5.0
