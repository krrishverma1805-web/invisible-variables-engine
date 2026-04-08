"""
Data Profiler — Invisible Variables Engine.

Generates comprehensive statistical summaries and quality reports for
datasets.  The profile is used by the detection engine (phase 3) and
displayed in the Streamlit UI.

Usage::

    from ive.data.profiler import DataProfiler

    profiler = DataProfiler()
    profile = profiler.profile(df, target_column="price")
    print(profile.quality_score)
    print([i.message for i in profile.quality_issues])

All computations are vectorised (no row-by-row loops).
Floats are rounded to 4 decimal places throughout.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ive.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_MISSING_MEDIUM = 30.0  # % null → medium issue
_MISSING_HIGH = 70.0  # % null → high issue
_NEAR_CONSTANT_UNIQUE = 5  # fewer unique numeric values → near-constant
_HIGH_CORR = 0.95  # |r| above this → high-correlation issue
_CORR_REPORT_MIN = 0.30  # |r| below this → skip in matrix
_OUTLIER_HIGH = 50.0  # >50% outliers → low issue
_SKEW_THRESHOLD = 2.0  # |skewness| above this → low issue
_IMBALANCE_RATIO = 0.20  # minority/majority below this → imbalanced
_VIF_WARNING = 5.0  # VIF above this → medium multicollinearity warning
_VIF_HIGH = 10.0  # VIF above this → high multicollinearity issue
_VIF_CAP = 1000.0  # cap VIF to avoid inf values
_VIF_MIN_ROWS = 10  # minimum rows required for VIF computation
_TOP_VALUES = 10  # max top_values entries per categorical column
_TOP_CORRELATIONS = 20  # max CorrelationPairs returned
_SAMPLE_N = 5  # random sample values shown per column

# Quality score deductions
_DEDUCT_HIGH = 10
_DEDUCT_MEDIUM = 5
_DEDUCT_LOW = 2


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TargetProfile(BaseModel):  # type: ignore[misc]
    """Statistical summary of the target / label column."""

    name: str
    task_type: str  # "regression" | "classification"

    # Regression stats
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    min: float | None = None
    max: float | None = None

    # Classification stats
    class_distribution: dict[str, int] | None = None
    is_imbalanced: bool = False
    num_classes: int | None = None


class ColumnProfile(BaseModel):  # type: ignore[misc]
    """Per-column statistical summary."""

    name: str
    detected_type: str
    dtype: str

    # Completeness
    null_count: int
    null_pct: float

    # Uniqueness
    unique_count: int
    unique_pct: float

    # Numeric
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    outlier_count: int | None = None
    zero_count: int | None = None

    # Categorical
    top_values: list[dict[str, Any]] | None = None

    # All
    sample_values: list[Any] = Field(default_factory=list)


class CorrelationPair(BaseModel):  # type: ignore[misc]
    """A pair of features with their Pearson correlation coefficient."""

    feature_a: str
    feature_b: str
    correlation: float
    abs_correlation: float


class VifResult(BaseModel):  # type: ignore[misc]
    """Variance Inflation Factor result for a single feature."""

    feature: str
    vif: float
    severity: str  # "high" | "medium"


class QualityIssue(BaseModel):  # type: ignore[misc]
    """A data quality issue detected during profiling."""

    severity: str  # "high" | "medium" | "low"
    category: str  # see module-level constants below
    column: str | None = None  # None → dataset-level issue
    message: str
    suggestion: str


class DataProfile(BaseModel):  # type: ignore[misc]
    """Complete statistical profile for a dataset."""

    dataset_id: str
    row_count: int
    col_count: int
    memory_usage_mb: float

    target_stats: TargetProfile

    column_profiles: list[ColumnProfile]

    correlation_matrix: dict[str, dict[str, float]]
    top_correlations: list[CorrelationPair]
    vif_results: list[VifResult] = Field(default_factory=list)

    quality_score: float  # 0–100
    quality_issues: list[QualityIssue]

    recommendations: list[str]


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class DataProfiler:
    """Generate a :class:`DataProfile` from a Pandas ``DataFrame``.

    The profiler is stateless — every call to :meth:`profile` is independent.
    All heavy lifting is done via vectorised Pandas / NumPy operations.
    """

    def profile(
        self,
        df: pd.DataFrame,
        target_column: str,
        time_column: str | None = None,
        column_types: list[Any] | None = None,
        dataset_id: str = "",
    ) -> DataProfile:
        """Generate a complete profile for a dataset.

        Args:
            df:            The dataset to profile (Pandas DataFrame).
            target_column: Name of the target / label column.
            time_column:   Optional datetime column (excluded from numeric
                           correlation analysis).
            column_types:  Optional list of :class:`ColumnTypeInfo` from the
                           ingestion step.  When provided, detected_type is
                           taken directly; otherwise it is re-inferred.
            dataset_id:    UUID string to embed in the profile.

        Returns:
            A fully-populated :class:`DataProfile` instance.
        """
        if df.empty or len(df.columns) == 0:
            raise ValueError("Cannot profile an empty DataFrame.")

        logger.info(
            "profiler.start",
            rows=len(df),
            cols=len(df.columns),
            target=target_column,
        )

        # Build a type-lookup from ColumnTypeInfo if provided
        type_map: dict[str, str] = {}
        if column_types:
            type_map = {ct.name: ct.detected_type for ct in column_types}

        # ── Basic metadata ────────────────────────────────────────────
        memory_mb = round(df.memory_usage(deep=True).sum() / (1024**2), 4)

        # ── Target profiling ─────────────────────────────────────────
        target_stats = self._profile_target(df, target_column)

        # ── Column profiling ─────────────────────────────────────────
        col_profiles = self._profile_columns(df, target_column, time_column, type_map)

        # ── Correlation analysis ──────────────────────────────────────
        corr_matrix, top_corrs = self._compute_correlations(df, target_column, time_column)

        # ── VIF multicollinearity analysis ────────────────────────────
        vif_results = self._compute_vif(df, target_column, time_column)

        # ── Quality issues ────────────────────────────────────────────
        issues = self._detect_quality_issues(
            df, col_profiles, top_corrs, target_stats, target_column,
            vif_results=vif_results,
        )

        # ── Quality score ─────────────────────────────────────────────
        score = self._compute_quality_score(issues)

        # ── Recommendations ───────────────────────────────────────────
        recs = self._generate_recommendations(issues)

        logger.info(
            "profiler.complete",
            quality_score=score,
            issues=len(issues),
            recommendations=len(recs),
        )

        return DataProfile(
            dataset_id=dataset_id,
            row_count=len(df),
            col_count=len(df.columns),
            memory_usage_mb=memory_mb,
            target_stats=target_stats,
            column_profiles=col_profiles,
            correlation_matrix=corr_matrix,
            top_correlations=top_corrs,
            vif_results=vif_results,
            quality_score=score,
            quality_issues=issues,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Target profiling
    # ------------------------------------------------------------------

    def _profile_target(self, df: pd.DataFrame, target_column: str) -> TargetProfile:
        """Build the :class:`TargetProfile` for the label column.

        Args:
            df:            The dataset DataFrame.
            target_column: Name of the target column.

        Returns:
            :class:`TargetProfile` with regression or classification stats.
        """
        if target_column not in df.columns:
            return TargetProfile(name=target_column, task_type="regression")

        series = df[target_column].dropna()
        n_unique = series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(series)

        # --- Task type detection ---
        is_int_like = is_numeric and (
            pd.api.types.is_integer_dtype(series)
            or (series == series.astype(int, errors="ignore")).all()
        )
        task_type = (
            "classification"
            if (n_unique <= 20 and (is_int_like or not is_numeric))
            else "regression"
        )

        if task_type == "regression" and is_numeric:
            return TargetProfile(
                name=target_column,
                task_type=task_type,
                mean=round(float(series.mean()), 4),
                std=round(float(series.std()), 4),
                median=round(float(series.median()), 4),
                min=round(float(series.min()), 4),
                max=round(float(series.max()), 4),
                skewness=round(float(series.skew()), 4),
                kurtosis=round(float(series.kurtosis()), 4),
            )

        # Classification
        value_counts = series.value_counts()
        dist = {str(k): int(v) for k, v in value_counts.items()}
        majority = value_counts.iloc[0]
        minority = value_counts.iloc[-1]
        is_imbalanced = (minority / majority) < _IMBALANCE_RATIO if majority > 0 else False

        return TargetProfile(
            name=target_column,
            task_type=task_type,
            class_distribution=dist,
            is_imbalanced=is_imbalanced,
            num_classes=n_unique,
        )

    # ------------------------------------------------------------------
    # Column profiling
    # ------------------------------------------------------------------

    def _profile_columns(
        self,
        df: pd.DataFrame,
        target_column: str,
        time_column: str | None,
        type_map: dict[str, str],
    ) -> list[ColumnProfile]:
        """Build a :class:`ColumnProfile` for every column in the DataFrame.

        Args:
            df:            The dataset DataFrame.
            target_column: Target column name (profiled separately but included).
            time_column:   Optional datetime column name.
            type_map:      Mapping of column name → detected_type from ingestion.

        Returns:
            List of :class:`ColumnProfile`, one per column.
        """
        n_rows = len(df)
        profiles: list[ColumnProfile] = []

        for col in df.columns:
            series = df[col]
            dtype_str = str(series.dtype)
            detected_type = type_map.get(col, self._infer_type(series, col))

            null_count = int(series.isna().sum())
            null_pct = round(null_count / n_rows * 100, 4) if n_rows > 0 else 0.0
            non_null = series.dropna()
            unique_count = int(non_null.nunique())
            unique_pct = round(unique_count / len(non_null) * 100, 4) if len(non_null) > 0 else 0.0

            # Sample values (up to _SAMPLE_N random non-null)
            sample: list[Any] = []
            if len(non_null) > 0:
                n = min(_SAMPLE_N, len(non_null))
                sample = non_null.sample(n=n, random_state=42).tolist()

            # Numeric stats
            num_kwargs: dict[str, Any] = {}
            if detected_type in ("numeric", "boolean") and pd.api.types.is_numeric_dtype(series):
                num_kwargs = self._numeric_stats(non_null)

            # Categorical top_values
            top_vals: list[dict[str, Any]] | None = None
            if detected_type in ("categorical", "boolean"):
                vc = non_null.value_counts().head(_TOP_VALUES)
                total = len(non_null)
                top_vals = [
                    {
                        "value": str(val),
                        "count": int(cnt),
                        "pct": round(cnt / total * 100, 4) if total > 0 else 0.0,
                    }
                    for val, cnt in vc.items()
                ]

            profiles.append(
                ColumnProfile(
                    name=col,
                    detected_type=detected_type,
                    dtype=dtype_str,
                    null_count=null_count,
                    null_pct=null_pct,
                    unique_count=unique_count,
                    unique_pct=unique_pct,
                    top_values=top_vals,
                    sample_values=sample,
                    **num_kwargs,
                )
            )

        return profiles

    def _numeric_stats(self, non_null: pd.Series) -> dict[str, Any]:
        """Compute vectorised numeric statistics for a non-null Series.

        Args:
            non_null: Series with NaN already dropped.

        Returns:
            Dict of keyword arguments to unpack into :class:`ColumnProfile`.
        """
        if len(non_null) == 0:
            return {}

        def r4(x: Any) -> float:
            return round(float(x), 4)

        q25, median, q75 = (
            non_null.quantile(0.25),
            non_null.quantile(0.50),
            non_null.quantile(0.75),
        )
        iqr = q75 - q25
        if iqr > 0:
            lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            outlier_count = int(((non_null < lo) | (non_null > hi)).sum())
        else:
            outlier_count = 0

        zero_count = int((non_null == 0).sum())

        return dict(
            mean=r4(non_null.mean()),
            std=r4(non_null.std()),
            min=r4(non_null.min()),
            max=r4(non_null.max()),
            median=r4(median),
            q25=r4(q25),
            q75=r4(q75),
            skewness=r4(non_null.skew()),
            kurtosis=r4(non_null.kurtosis()),
            outlier_count=outlier_count,
            zero_count=zero_count,
        )

    def _infer_type(self, series: pd.Series, col_name: str) -> str:
        """Lightweight type inference used when no ``type_map`` is provided."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        non_null = series.dropna()
        if non_null.nunique() < 50 or (non_null.nunique() / max(len(non_null), 1) * 100) < 5:
            return "categorical"
        return "text"

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def _compute_correlations(
        self,
        df: pd.DataFrame,
        target_column: str,
        time_column: str | None,
    ) -> tuple[dict[str, dict[str, float]], list[CorrelationPair]]:
        """Compute the Pearson correlation matrix for numeric columns.

        Only correlations with ``|r| > _CORR_REPORT_MIN`` are retained
        in the returned sparse matrix.  Returns the top ``_TOP_CORRELATIONS``
        pairs by absolute correlation.

        Args:
            df:            The dataset DataFrame.
            target_column: Excluded from pair-listing (but included in matrix).
            time_column:   Excluded from correlation analysis.

        Returns:
            Tuple of (sparse_corr_matrix_dict, top_correlation_pairs).
        """
        # Select numeric columns, exclude time column
        exclude = {time_column} if time_column else set()
        num_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude
        ]

        if len(num_cols) < 2:
            return {}, []

        corr_df = df[num_cols].corr(method="pearson")

        # Sparse matrix: only keep |r| > threshold
        sparse: dict[str, dict[str, float]] = {}
        pairs: list[tuple[str, str, float]] = []

        for i, a in enumerate(corr_df.columns):
            for j, b in enumerate(corr_df.columns):
                if i >= j:
                    continue  # upper triangle + skip diagonal
                r = corr_df.loc[a, b]
                if pd.isna(r):
                    continue
                r_r = round(float(r), 4)
                if abs(r_r) >= _CORR_REPORT_MIN:
                    sparse.setdefault(a, {})[b] = r_r
                    sparse.setdefault(b, {})[a] = r_r
                    pairs.append((a, b, r_r))

        # Sort by absolute correlation descending, keep top N
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        top_corrs = [
            CorrelationPair(
                feature_a=a,
                feature_b=b,
                correlation=r,
                abs_correlation=round(abs(r), 4),
            )
            for a, b, r in pairs[:_TOP_CORRELATIONS]
        ]

        return sparse, top_corrs

    # ------------------------------------------------------------------
    # VIF multicollinearity analysis
    # ------------------------------------------------------------------

    def _compute_vif(
        self,
        df: pd.DataFrame,
        target_column: str,
        time_column: str | None,
    ) -> list[VifResult]:
        """Compute Variance Inflation Factor for numeric features.

        VIF > 10 indicates problematic multicollinearity.
        VIF > 5 is a warning.

        Args:
            df:            The dataset DataFrame.
            target_column: Excluded from VIF computation.
            time_column:   Excluded from VIF computation.

        Returns:
            List of :class:`VifResult` for features with VIF > 5.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        exclude: set[str] = {target_column}
        if time_column:
            exclude.add(time_column)

        num_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude
        ]

        if len(num_cols) < 2:
            return []

        # Drop rows with NaN and limit to numeric columns
        X = df[num_cols].dropna()
        if len(X) < _VIF_MIN_ROWS:
            return []

        results: list[VifResult] = []
        try:
            X_values = X.values.astype(float)
            for i, col in enumerate(num_cols):
                vif = variance_inflation_factor(X_values, i)
                # Cap VIF to avoid inf values from singular matrices
                vif = min(float(vif), _VIF_CAP)
                if vif > _VIF_WARNING:
                    severity = "high" if vif > _VIF_HIGH else "medium"
                    results.append(
                        VifResult(
                            feature=col,
                            vif=round(vif, 2),
                            severity=severity,
                        )
                    )
        except Exception:
            # VIF can fail with singular or near-singular matrices
            logger.warning("profiler.vif_failed", num_cols=len(num_cols))

        return results

    # ------------------------------------------------------------------
    # Quality issues
    # ------------------------------------------------------------------

    def _detect_quality_issues(
        self,
        df: pd.DataFrame,
        col_profiles: list[ColumnProfile],
        top_corrs: list[CorrelationPair],
        target_stats: TargetProfile,
        target_column: str,
        *,
        vif_results: list[VifResult] | None = None,
    ) -> list[QualityIssue]:
        """Identify all data quality problems.

        Categories checked:
            * missing_values     — columns >30% or >70% null
            * constant_column    — zero variance
            * near_constant      — <5 unique numeric values
            * high_correlation   — |r| > 0.95 between two features
            * multicollinearity  — VIF > 5 (medium) or VIF > 10 (high)
            * outliers           — >50% of values are IQR outliers
            * skewness           — |skewness| > 2
            * class_imbalance    — minority class < 20% of majority

        Args:
            df:            The dataset DataFrame.
            col_profiles:  Pre-computed column profiles.
            top_corrs:     Pre-computed top correlations.
            target_stats:  Pre-computed target profile.
            target_column: Name of target column.
            vif_results:   Pre-computed VIF results (optional).

        Returns:
            List of :class:`QualityIssue`.
        """
        issues: list[QualityIssue] = []

        for cp in col_profiles:
            # Missing values
            if cp.null_pct > _MISSING_HIGH:
                issues.append(
                    QualityIssue(
                        severity="high",
                        category="missing_values",
                        column=cp.name,
                        message=f"'{cp.name}' is {cp.null_pct:.1f}% null.",
                        suggestion=f"Consider dropping or imputing '{cp.name}'.",
                    )
                )
            elif cp.null_pct > _MISSING_MEDIUM:
                issues.append(
                    QualityIssue(
                        severity="medium",
                        category="missing_values",
                        column=cp.name,
                        message=f"'{cp.name}' is {cp.null_pct:.1f}% null.",
                        suggestion=f"Consider imputing missing values in '{cp.name}'.",
                    )
                )

            # Constant column (zero unique non-null values, or 1 unique value)
            if cp.unique_count <= 1 and cp.null_pct < 100.0:
                issues.append(
                    QualityIssue(
                        severity="high",
                        category="constant_column",
                        column=cp.name,
                        message=f"'{cp.name}' has {cp.unique_count} unique value(s) — it is effectively constant.",
                        suggestion=f"Remove '{cp.name}'; it provides no predictive information.",
                    )
                )

            # Near-constant numeric
            elif (
                cp.detected_type == "numeric"
                and cp.unique_count < _NEAR_CONSTANT_UNIQUE
                and cp.null_pct < 100.0
            ):
                issues.append(
                    QualityIssue(
                        severity="medium",
                        category="constant_column",
                        column=cp.name,
                        message=f"'{cp.name}' has only {cp.unique_count} unique numeric values.",
                        suggestion=f"Treat '{cp.name}' as categorical or review its meaning.",
                    )
                )

            # Outliers
            if cp.outlier_count is not None:
                non_null_count = max(1, cp.null_count - len(df) + len(df))  # re-derive
                non_null_n = len(df) - cp.null_count
                if non_null_n > 0 and (cp.outlier_count / non_null_n * 100) > _OUTLIER_HIGH:
                    issues.append(
                        QualityIssue(
                            severity="low",
                            category="outliers",
                            column=cp.name,
                            message=(
                                f"'{cp.name}' has {cp.outlier_count} IQR outliers "
                                f"({cp.outlier_count / non_null_n * 100:.1f}%)."
                            ),
                            suggestion=f"Consider capping or transforming '{cp.name}'.",
                        )
                    )

            # Heavy skewness
            if cp.skewness is not None and abs(cp.skewness) > _SKEW_THRESHOLD:
                issues.append(
                    QualityIssue(
                        severity="low",
                        category="skewness",
                        column=cp.name,
                        message=f"'{cp.name}' is heavily skewed (skewness={cp.skewness:.2f}).",
                        suggestion=f"Consider log or Box-Cox transform for '{cp.name}'.",
                    )
                )

        # High correlation pairs
        for pair in top_corrs:
            if (
                pair.abs_correlation >= _HIGH_CORR
                and pair.feature_a != target_column
                and pair.feature_b != target_column
            ):
                issues.append(
                    QualityIssue(
                        severity="medium",
                        category="high_correlation",
                        column=None,
                        message=(
                            f"'{pair.feature_a}' and '{pair.feature_b}' are highly correlated "
                            f"(r={pair.correlation:.2f})."
                        ),
                        suggestion=(
                            f"Consider removing one of '{pair.feature_a}' or '{pair.feature_b}' "
                            "to reduce multicollinearity."
                        ),
                    )
                )

        # Class imbalance
        if target_stats.is_imbalanced and target_stats.class_distribution:
            issues.append(
                QualityIssue(
                    severity="high",
                    category="class_imbalance",
                    column=target_column,
                    message=(
                        f"Target '{target_column}' is class-imbalanced. "
                        f"Distribution: {dict(list(target_stats.class_distribution.items())[:5])!r}"
                    ),
                    suggestion=(
                        "Apply SMOTE, class weighting, or oversample the minority class "
                        "before training."
                    ),
                )
            )

        # VIF multicollinearity
        for vif_item in vif_results or []:
            desc = "severe" if vif_item.severity == "high" else "moderate"
            issues.append(
                QualityIssue(
                    severity=vif_item.severity,
                    category="multicollinearity",
                    column=vif_item.feature,
                    message=(
                        f"'{vif_item.feature}' has VIF={vif_item.vif}, "
                        f"indicating {desc} multicollinearity."
                    ),
                    suggestion=(
                        f"Consider removing or combining '{vif_item.feature}' with "
                        "correlated features to reduce multicollinearity."
                    ),
                )
            )

        return issues

    # ------------------------------------------------------------------
    # Quality score
    # ------------------------------------------------------------------

    def _compute_quality_score(self, issues: list[QualityIssue]) -> float:
        """Compute a 0–100 quality score from detected issues.

        Deductions:
            High severity   → −10 per issue
            Medium severity → −5  per issue
            Low severity    → −2  per issue

        Args:
            issues: Detected :class:`QualityIssue` list.

        Returns:
            Float in [0, 100].
        """
        score = 100.0
        for issue in issues:
            if issue.severity == "high":
                score -= _DEDUCT_HIGH
            elif issue.severity == "medium":
                score -= _DEDUCT_MEDIUM
            else:
                score -= _DEDUCT_LOW
        return round(max(0.0, score), 2)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(self, issues: list[QualityIssue]) -> list[str]:
        """Convert quality issues into human-readable recommendation strings.

        Recommendations are de-duplicated and sorted by severity.

        Args:
            issues: Detected :class:`QualityIssue` list.

        Returns:
            Deduplicated list of recommendation strings.
        """
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_issues = sorted(issues, key=lambda i: severity_order.get(i.severity, 99))
        seen: set[str] = set()
        recs: list[str] = []
        for issue in sorted_issues:
            if issue.suggestion and issue.suggestion not in seen:
                seen.add(issue.suggestion)
                recs.append(issue.suggestion)
        return recs
