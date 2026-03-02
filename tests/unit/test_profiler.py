"""
Unit tests for DataProfiler — Invisible Variables Engine.

Tests cover:
    * profile() on regression and classification datasets
    * Numeric column statistics (mean, std, median, IQR outliers, skewness)
    * Categorical column top_values
    * Quality issue detection (missing values, constant columns, high correlation,
      class imbalance, skewness, outliers)
    * Quality score arithmetic
    * Recommendation generation and deduplication
    * Edge cases: all-null column, single-column DataFrame, empty DataFrame

No database, Redis, or filesystem access required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.data.profiler import (
    DataProfile,
    DataProfiler,
    QualityIssue,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _build_regression_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.uniform(0, 100, n),
            "cat": rng.choice(["A", "B", "C"], n),
            "y": rng.standard_normal(n) * 10 + 50,
        }
    )


def _build_classification_df(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "income": rng.normal(50_000, 20_000, n),
            "cat": rng.choice(["X", "Y"], n),
            "label": rng.integers(0, 2, n),  # binary classification
        }
    )


# ===========================================================================
# Core profile output structure
# ===========================================================================


@pytest.mark.unit
class TestProfileOutput:
    """Tests that profile() returns a correctly structured DataProfile."""

    def test_profile_returns_dataprofile(self) -> None:
        """profile() returns a DataProfile Pydantic model instance."""
        profiler = DataProfiler()
        df = _build_regression_df()
        result = profiler.profile(df, target_column="y")
        assert isinstance(result, DataProfile)

    def test_profile_regression_row_col_counts(self) -> None:
        """row_count and col_count match the input DataFrame."""
        profiler = DataProfiler()
        df = _build_regression_df(n=300)
        result = profiler.profile(df, target_column="y")
        assert result.row_count == 300
        assert result.col_count == len(df.columns)

    def test_profile_regression_task_type(self) -> None:
        """task_type is 'regression' for a continuous numeric target."""
        profiler = DataProfiler()
        df = _build_regression_df()
        result = profiler.profile(df, target_column="y")
        assert result.target_stats.task_type == "regression"

    def test_profile_classification_task_type(self) -> None:
        """task_type is 'classification' for a low-cardinality integer target."""
        profiler = DataProfiler()
        df = _build_classification_df()
        result = profiler.profile(df, target_column="label")
        assert result.target_stats.task_type == "classification"

    def test_profile_memory_usage_positive(self) -> None:
        """memory_usage_mb is a positive float."""
        profiler = DataProfiler()
        df = _build_regression_df()
        result = profiler.profile(df, target_column="y")
        assert result.memory_usage_mb > 0.0

    def test_profile_quality_score_in_range(self) -> None:
        """quality_score is between 0 and 100 inclusive."""
        profiler = DataProfiler()
        df = _build_regression_df()
        result = profiler.profile(df, target_column="y")
        assert 0.0 <= result.quality_score <= 100.0

    def test_profile_empty_df_raises(self) -> None:
        """Profiling an empty DataFrame raises ValueError."""
        profiler = DataProfiler()
        with pytest.raises(ValueError, match="empty"):
            profiler.profile(pd.DataFrame(), target_column="y")

    def test_profile_column_profiles_count(self) -> None:
        """One ColumnProfile is returned per column."""
        profiler = DataProfiler()
        df = _build_regression_df()
        result = profiler.profile(df, target_column="y")
        assert len(result.column_profiles) == len(df.columns)

    def test_profile_dataset_id_embedded(self) -> None:
        """dataset_id is passed through into the DataProfile."""
        profiler = DataProfiler()
        df = _build_regression_df()
        result = profiler.profile(df, target_column="y", dataset_id="test-uuid-123")
        assert result.dataset_id == "test-uuid-123"


# ===========================================================================
# Numeric column statistics
# ===========================================================================


@pytest.mark.unit
class TestNumericColumnStats:
    """Tests for per-column numeric statistics in ColumnProfile."""

    def _get_col(self, df: pd.DataFrame, col: str, target: str = "y"):
        profiler = DataProfiler()
        result = profiler.profile(df, target_column=target)
        return next(cp for cp in result.column_profiles if cp.name == col)

    def test_numeric_mean_is_correct(self) -> None:
        """mean is close to the true mean of the column."""
        rng = np.random.default_rng(0)
        n = 1_000
        vals = rng.standard_normal(n) * 5 + 10  # mean ≈ 10
        df = pd.DataFrame({"x": vals, "z": rng.standard_normal(n), "y": rng.standard_normal(n)})
        cp = self._get_col(df, "x")
        assert cp.mean is not None
        assert abs(cp.mean - vals.mean()) < 0.5, f"mean={cp.mean}, expected≈{vals.mean()}"

    def test_numeric_std_is_correct(self) -> None:
        """std is close to the true standard deviation."""
        rng = np.random.default_rng(1)
        n = 1_000
        vals = rng.standard_normal(n) * 3.0
        df = pd.DataFrame({"x": vals, "z": rng.standard_normal(n), "y": rng.standard_normal(n)})
        cp = self._get_col(df, "x")
        assert cp.std is not None
        assert abs(cp.std - vals.std()) < 0.3

    def test_numeric_median_is_correct(self) -> None:
        """median matches pandas median."""
        rng = np.random.default_rng(2)
        n = 500
        vals = rng.standard_normal(n)
        df = pd.DataFrame({"x": vals, "z": rng.standard_normal(n), "y": rng.standard_normal(n)})
        cp = self._get_col(df, "x")
        assert cp.median is not None
        assert abs(cp.median - float(pd.Series(vals).median())) < 0.01

    def test_numeric_floats_rounded_to_4dp(self) -> None:
        """All float fields in ColumnProfile are rounded to 4 decimal places."""
        rng = np.random.default_rng(3)
        n = 200
        df = pd.DataFrame(
            {"x": rng.standard_normal(n), "z": rng.standard_normal(n), "y": rng.standard_normal(n)}
        )
        cp = self._get_col(df, "x")
        for field_name in ("mean", "std", "min", "max", "median", "q25", "q75"):
            val = getattr(cp, field_name)
            if val is not None:
                assert round(val, 4) == val, f"{field_name}={val} not rounded to 4dp"

    def test_outlier_detection_via_iqr(self) -> None:
        """Extreme outliers are counted via IQR method."""
        rng = np.random.default_rng(10)
        n = 200
        normal = rng.standard_normal(n)
        outliers_arr = np.array([100.0, -100.0, 200.0])  # definitely outside IQR
        vals = np.concatenate([normal, outliers_arr])
        df = pd.DataFrame(
            {"x": vals, "z": rng.standard_normal(len(vals)), "y": rng.standard_normal(len(vals))}
        )
        cp = self._get_col(df, "x")
        assert cp.outlier_count is not None
        assert cp.outlier_count >= 3, f"Expected ≥3 outliers, got {cp.outlier_count}"

    def test_zero_count(self) -> None:
        """zero_count field counts exactly the number of zeros."""
        n = 200
        vals = np.zeros(50).tolist() + list(np.random.default_rng(5).standard_normal(n - 50))
        df = pd.DataFrame({"x": vals, "z": range(n), "y": range(n)})
        cp = self._get_col(df, "x")
        assert cp.zero_count is not None
        assert cp.zero_count == 50


# ===========================================================================
# Categorical column statistics
# ===========================================================================


@pytest.mark.unit
class TestCategoricalColumnStats:
    """Tests for categorical column profiling."""

    def _get_col(self, df: pd.DataFrame, col: str, target: str = "y"):
        profiler = DataProfiler()
        result = profiler.profile(df, target_column=target)
        return next(cp for cp in result.column_profiles if cp.name == col)

    def test_categorical_top_values_present(self) -> None:
        """top_values list is populated for categorical columns."""
        rng = np.random.default_rng(0)
        n = 300
        df = pd.DataFrame(
            {
                "cat": rng.choice(["A", "B", "C"], n),
                "x": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        cp = self._get_col(df, "cat")
        assert cp.top_values is not None
        assert len(cp.top_values) > 0

    def test_categorical_top_values_max_10(self) -> None:
        """top_values contains at most 10 entries."""
        rng = np.random.default_rng(1)
        n = 500
        categories = [f"cat_{i}" for i in range(30)]  # 30 distinct values
        df = pd.DataFrame(
            {
                "cat": rng.choice(categories, n),
                "x": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        cp = self._get_col(df, "cat")
        assert cp.top_values is not None
        assert len(cp.top_values) <= 10

    def test_categorical_top_value_pct_sums(self) -> None:
        """Sum of top_values pct is ≤100% (may not cover all values)."""
        rng = np.random.default_rng(2)
        n = 200
        df = pd.DataFrame(
            {
                "cat": rng.choice(["X", "Y"], n),
                "x": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        cp = self._get_col(df, "cat")
        assert cp.top_values is not None
        total_pct = sum(v["pct"] for v in cp.top_values)
        assert total_pct <= 100.01  # floating-point tolerance

    def test_categorical_unique_count(self) -> None:
        """unique_count matches the actual number of distinct non-null values."""
        n = 300
        df = pd.DataFrame({"cat": ["A", "B", "C"] * 100, "x": range(n), "y": range(n)})
        cp = self._get_col(df, "cat")
        assert cp.unique_count == 3


# ===========================================================================
# Quality issue detection
# ===========================================================================


@pytest.mark.unit
class TestQualityIssues:
    """Tests that DataProfiler correctly surfaces data quality problems."""

    def _issues(self, df: pd.DataFrame, target: str = "y") -> list[QualityIssue]:
        return DataProfiler().profile(df, target_column=target).quality_issues

    def test_detects_missing_values_medium(self) -> None:
        """A column with 40% nulls generates a 'medium' missing_values issue."""
        rng = np.random.default_rng(0)
        n = 500
        col = np.where(rng.random(n) < 0.40, float("nan"), rng.standard_normal(n))
        df = pd.DataFrame({"x1": col, "x2": rng.standard_normal(n), "y": rng.standard_normal(n)})
        issues = self._issues(df)
        missing = [i for i in issues if i.category == "missing_values" and i.column == "x1"]
        assert missing, "Expected missing_values issue for x1"
        assert missing[0].severity in ("medium", "high")

    def test_detects_missing_values_high(self) -> None:
        """A column with 80% nulls generates a 'high' missing_values issue."""
        rng = np.random.default_rng(1)
        n = 500
        col = np.where(rng.random(n) < 0.80, float("nan"), rng.standard_normal(n))
        df = pd.DataFrame({"x1": col, "x2": rng.standard_normal(n), "y": rng.standard_normal(n)})
        issues = self._issues(df)
        missing = [i for i in issues if i.category == "missing_values" and i.column == "x1"]
        assert missing
        assert missing[0].severity == "high"

    def test_detects_constant_column(self) -> None:
        """A constant column (zero variance) generates a 'constant_column' issue."""
        n = 300
        df = pd.DataFrame({"const": [7.0] * n, "x2": range(n), "y": range(n)})
        issues = self._issues(df)
        const_issues = [
            i for i in issues if i.category == "constant_column" and i.column == "const"
        ]
        assert const_issues, f"Expected constant_column issue, got: {issues}"
        assert const_issues[0].severity == "high"

    def test_detects_high_correlation(self, sample_large_df: pd.DataFrame) -> None:
        """Columns with |r| ≥ 0.95 generate a 'high_correlation' issue."""
        # sample_large_df has x1 and x3 correlated at ~0.97
        issues = self._issues(sample_large_df, target="target")
        high_corr = [i for i in issues if i.category == "high_correlation"]
        assert high_corr, "Expected high_correlation issue for x1/x3"

    def test_detects_class_imbalance(self) -> None:
        """Highly imbalanced binary target generates a 'class_imbalance' issue."""
        rng = np.random.default_rng(5)
        n = 500
        # 95% class 0, 5% class 1  → imbalance ratio = 0.05/0.95 ≈ 0.053 < 0.20
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "label": rng.choice([0, 1], n, p=[0.95, 0.05]),
            }
        )
        issues = self._issues(df, target="label")
        imbalance = [i for i in issues if i.category == "class_imbalance"]
        assert imbalance, f"Expected class_imbalance issue, got: {[i.category for i in issues]}"

    def test_no_class_imbalance_for_balanced(self) -> None:
        """Balanced binary target does NOT generate a class_imbalance issue."""
        rng = np.random.default_rng(6)
        n = 500
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "label": rng.choice([0, 1], n, p=[0.5, 0.5]),
            }
        )
        issues = self._issues(df, target="label")
        imbalance = [i for i in issues if i.category == "class_imbalance"]
        assert not imbalance, "No imbalance expected for 50/50 split"

    def test_detects_outliers(self) -> None:
        """A column with >50% IQR outliers generates a 'low' outlier issue."""
        # Create a bimodal column where the majority are far from the IQR
        rng = np.random.default_rng(7)
        n = 400
        # half near 0, half at 1000 (clear outliers for the lower group)
        extremes = np.concatenate(
            [rng.standard_normal(n // 2), rng.standard_normal(n // 2) * 0.1 + 1000]
        )
        df = pd.DataFrame({"x": extremes, "z": rng.standard_normal(n), "y": rng.standard_normal(n)})
        issues = DataProfiler().profile(df, target_column="y").quality_issues
        outlier_issues = [i for i in issues if i.category == "outliers"]
        assert outlier_issues, "Expected outlier issue for bimodal column"


# ===========================================================================
# All-null column edge case
# ===========================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge cases for DataProfiler."""

    def test_handles_all_null_column(self) -> None:
        """A fully-null column is profiled without error (null_pct=100)."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "ghost": [float("nan")] * n,  # all-null
                "y": rng.standard_normal(n),
            }
        )
        result = DataProfiler().profile(df, target_column="y")
        ghost_cp = next(cp for cp in result.column_profiles if cp.name == "ghost")
        assert ghost_cp.null_pct == 100.0
        assert ghost_cp.unique_count == 0

    def test_handles_single_numeric_category(self) -> None:
        """Target with a single unique value causes a validation-level quality issue."""
        n = 200
        df = pd.DataFrame({"x1": range(n), "x2": range(n), "y": [42] * n})
        result = DataProfiler().profile(df, target_column="y")
        # 42 repeated n times → constant column issue
        const_issues = [i for i in result.quality_issues if i.category == "constant_column"]
        assert const_issues or result.quality_score < 100  # some quality signal

    def test_quality_score_zero_floor(self) -> None:
        """quality_score never goes below 0 even with many issues."""
        rng = np.random.default_rng(99)
        n = 300
        # Build a pathological DataFrame: many constant + missing columns
        df = pd.DataFrame(
            {
                "x1": [float("nan")] * n,  # all null
                "x2": [1.0] * n,  # constant
                "x3": [float("nan")] * (n - 5) + list(rng.standard_normal(5)),  # ~98% null
                "x4": [2.0] * n,  # constant
                "y": rng.standard_normal(n),  # ok target
            }
        )
        result = DataProfiler().profile(df, target_column="y")
        assert result.quality_score >= 0.0

    def test_correlation_matrix_symmetric(self, sample_large_df: pd.DataFrame) -> None:
        """Correlation matrix values are symmetric: corr[a][b] == corr[b][a]."""
        result = DataProfiler().profile(sample_large_df, target_column="target")
        matrix = result.correlation_matrix
        for a, row in matrix.items():
            for b, r_ab in row.items():
                r_ba = matrix.get(b, {}).get(a)
                if r_ba is not None:
                    assert (
                        abs(r_ab - r_ba) < 1e-6
                    ), f"Asymmetry: [{a}][{b}]={r_ab} vs [{b}][{a}]={r_ba}"

    def test_top_correlations_sorted_by_abs(self, sample_large_df: pd.DataFrame) -> None:
        """top_correlations list is sorted descending by abs_correlation."""
        result = DataProfiler().profile(sample_large_df, target_column="target")
        pairs = result.top_correlations
        for i in range(len(pairs) - 1):
            assert (
                pairs[i].abs_correlation >= pairs[i + 1].abs_correlation
            ), f"Not sorted at index {i}: {pairs[i].abs_correlation} < {pairs[i+1].abs_correlation}"


# ===========================================================================
# Quality score and recommendations
# ===========================================================================


@pytest.mark.unit
class TestQualityScoreAndRecommendations:
    """Tests for quality score arithmetic and recommendation generation."""

    def test_perfect_quality_score(self) -> None:
        """A clean, well-balanced dataset scores 100.0."""
        rng = np.random.default_rng(0)
        n = 500
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y": rng.standard_normal(n) * 5 + 10,
            }
        )
        result = DataProfiler().profile(df, target_column="y")
        assert result.quality_score == 100.0, f"Expected 100.0, got {result.quality_score}"

    def test_quality_score_decreases_with_issues(self) -> None:
        """Adding quality problems reduces quality_score below 100."""
        rng = np.random.default_rng(1)
        n = 500
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": [float("nan")] * n,  # all null → high issue
                "y": rng.standard_normal(n),
            }
        )
        result = DataProfiler().profile(df, target_column="y")
        assert result.quality_score < 100.0

    def test_recommendations_are_unique(self) -> None:
        """Recommendations list has no duplicate strings."""
        rng = np.random.default_rng(2)
        n = 500
        col = np.where(rng.random(n) < 0.80, float("nan"), rng.standard_normal(n))
        df = pd.DataFrame({"x1": col, "x2": rng.standard_normal(n), "y": rng.standard_normal(n)})
        result = DataProfiler().profile(df, target_column="y")
        assert len(result.recommendations) == len(
            set(result.recommendations)
        ), "Duplicate recommendations found"

    def test_recommendations_correspond_to_issues(self) -> None:
        """Each recommendation maps to at least one quality issue."""
        rng = np.random.default_rng(3)
        n = 500
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        result = DataProfiler().profile(df, target_column="y")
        all_suggestions = {i.suggestion for i in result.quality_issues if i.suggestion}
        for rec in result.recommendations:
            assert rec in all_suggestions, f"Recommendation not backed by an issue: {rec!r}"

    def test_quality_score_arithmetic(self) -> None:
        """Quality score equals 100 minus weighted issue deductions, floored at 0."""
        # Manufacture a specific issue list via the profiler with a known dataset
        rng = np.random.default_rng(9)
        n = 500
        # Create a constant column → 1 high issue (−10)
        # and a 75%-null column → 1 high issue (−10)
        col_null = np.where(rng.random(n) < 0.75, float("nan"), rng.standard_normal(n))
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "const": [5.0] * n,
                "heavy_null": col_null,
                "y": rng.standard_normal(n),
            }
        )
        result = DataProfiler().profile(df, target_column="y")
        # Count deductions manually
        expected = 100.0
        for issue in result.quality_issues:
            if issue.severity == "high":
                expected -= 10
            elif issue.severity == "medium":
                expected -= 5
            else:
                expected -= 2
        expected = max(0.0, round(expected, 2))
        assert result.quality_score == expected, f"Expected {expected}, got {result.quality_score}"
