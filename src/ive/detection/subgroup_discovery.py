"""
Subgroup Discovery — Phase 3 Detection Engine.

Discovers feature-value subgroups where the model's residual distribution
differs significantly from the global residual distribution.  Uses a
column-by-column scan with binned quantile splits for numeric features and
per-value grouping for categoricals, applying a two-sample
Kolmogorov–Smirnov test with Bonferroni-corrected significance thresholds
and filtering by Cohen's *d* effect size.

Statistical pipeline
--------------------
For each column in *X*:

1. **Numeric** → up to 5 quantile bins via ``pd.qcut``; falls back to ``pd.cut``
   if ``qcut`` fails (e.g. too many ties), and finally treats the column as a
   single bin if cutting is impossible.
2. **Categorical / boolean** → one bin per unique value (NaN-aware).

For each bin (subgroup) with ≥ ``min_subgroup_size`` samples (default 20):

* Two-sample KS test:  ``scipy.stats.ks_2samp(bin_residuals, global_residuals)``
* Cohen's *d*:  ``(mean_bin − mean_global) / std_global``  (guarded against σ = 0)
* Bonferroni correction:  ``α_adj = α / max(1, total_bins_tested)``

A pattern is retained when ``p < α_adj`` **and** ``|d| > min_effect_size`` (default 0.15).

The output is a list of pattern dicts sorted by ``|effect_size|`` descending,
ready for the ``PatternScorer`` stage and eventual DB persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy.stats import ks_2samp

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Minimum samples required for a subgroup to be statistically tested.
# ---------------------------------------------------------------------------
_MIN_BIN_SAMPLES: int = 20

# ---------------------------------------------------------------------------
# Default number of quantile bins for numeric columns.
# ---------------------------------------------------------------------------
_DEFAULT_N_BINS: int = 5

# ---------------------------------------------------------------------------
# Minimum absolute Cohen's d to keep a pattern.
# ---------------------------------------------------------------------------
_MIN_EFFECT_SIZE: float = 0.15


# ---------------------------------------------------------------------------
# Legacy dataclass kept for backward-compatibility with __init__.py exports.
# ---------------------------------------------------------------------------


@dataclass
class SubgroupPattern:
    """A discovered subgroup pattern with its quality metrics."""

    rule: str
    conditions: list[dict[str, Any]]
    coverage: float = 0.0
    wracc: float = 0.0
    mean_residual: float = 0.0
    mean_residual_outside: float = 0.0
    effect_size: float = 0.0
    sample_mask: np.ndarray[Any, Any] | None = None


# ---------------------------------------------------------------------------
# New Phase-3 class requested by the user brief
# ---------------------------------------------------------------------------


class SubgroupDiscovery:
    """Column-scan subgroup discovery with Bonferroni-corrected KS tests.

    This is the primary Phase 3 class called by the detection orchestrator.
    It scans every column in *X*, bins values into subgroups, and tests
    whether the residual distribution within each subgroup significantly
    differs from the global distribution.

    Args:
        n_bins:          Maximum number of quantile bins for numeric columns.
        min_bin_samples: Minimum samples required in a bin before testing.
        min_effect_size: Minimum absolute Cohen's *d* to retain a pattern.
    """

    def __init__(
        self,
        n_bins: int = _DEFAULT_N_BINS,
        min_bin_samples: int = _MIN_BIN_SAMPLES,
        min_effect_size: float = _MIN_EFFECT_SIZE,
    ) -> None:
        self.n_bins = n_bins
        self.min_bin_samples = min_bin_samples
        self.min_effect_size = min_effect_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        X: pd.DataFrame,
        residuals: np.ndarray[Any, Any],
        alpha: float = 0.05,
    ) -> list[dict[str, Any]]:
        """Scan all columns of *X* for residual-distributional anomalies.

        Args:
            X:         Feature DataFrame (n_samples × n_features).
            residuals: OOF residual array of shape ``(n_samples,)``.
            alpha:     Family-wise significance level *before* Bonferroni
                       correction (default 0.05).

        Returns:
            List of pattern dicts sorted by ``|effect_size|`` descending.
            Each dict contains:

            * ``pattern_type``  — always ``"subgroup"``
            * ``column_name``   — feature column that defines the bin
            * ``bin_value``     — string representation of the bin
            * ``p_value``       — raw KS *p*-value
            * ``adjusted_alpha``— Bonferroni-corrected threshold
            * ``effect_size``   — Cohen's *d*
            * ``sample_count``  — number of samples in this bin
            * ``mean_residual`` — mean residual within the bin
            * ``std_residual``  — std of residuals within the bin

        Raises:
            ValueError: If lengths of *X* and *residuals* differ.
        """
        if len(X) != len(residuals):
            raise ValueError(f"X has {len(X)} rows but residuals has {len(residuals)} entries.")

        if len(residuals) == 0:
            log.info("ive.subgroup_discovery.skip_empty")
            return []

        residuals = np.asarray(residuals, dtype=np.float64)
        global_mean = float(np.nanmean(residuals))
        global_std = float(np.nanstd(residuals, ddof=0))

        log.info(
            "ive.subgroup_discovery.start",
            n_samples=len(residuals),
            n_features=len(X.columns),
            alpha=alpha,
        )

        # ── Pass 1: Build bins and count total tests ───────────────────
        column_bins: list[tuple[str, list[tuple[str, np.ndarray]]]] = []
        num_total_bins_tested = 0

        for col in X.columns:
            bins = self._bin_column(X[col], col_name=str(col))
            column_bins.append((str(col), bins))
            num_total_bins_tested += len(bins)

        adjusted_alpha = alpha / max(1, num_total_bins_tested)

        log.debug(
            "ive.subgroup_discovery.bins_counted",
            total_bins=num_total_bins_tested,
            adjusted_alpha=round(adjusted_alpha, 8),
        )

        # ── Pass 2: Test each bin ──────────────────────────────────────
        patterns: list[dict[str, Any]] = []

        for col_name, bins in column_bins:
            for bin_label, mask in bins:
                bin_residuals = residuals[mask]

                if len(bin_residuals) < self.min_bin_samples:
                    continue

                # KS test: compare bin distribution vs global
                try:
                    stat, p_value = ks_2samp(bin_residuals, residuals)
                except Exception:
                    continue

                # Cohen's d (protect against zero global std)
                if global_std > 0.0:
                    effect_size = (float(np.mean(bin_residuals)) - global_mean) / global_std
                else:
                    effect_size = 0.0

                if p_value < adjusted_alpha and abs(effect_size) > self.min_effect_size:
                    patterns.append(
                        {
                            "pattern_type": "subgroup",
                            "column_name": col_name,
                            "bin_value": str(bin_label),
                            "p_value": float(p_value),
                            "adjusted_alpha": float(adjusted_alpha),
                            "effect_size": float(effect_size),
                            "sample_count": int(len(bin_residuals)),
                            "mean_residual": float(np.mean(bin_residuals)),
                            "std_residual": float(np.std(bin_residuals, ddof=0)),
                        }
                    )

        # Sort by absolute effect size descending
        patterns.sort(key=lambda p: abs(p["effect_size"]), reverse=True)

        log.info(
            "ive.subgroup_discovery.complete",
            patterns_found=len(patterns),
            bins_tested=num_total_bins_tested,
        )
        return patterns

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bin_column(
        self,
        series: pd.Series,
        col_name: str,
    ) -> list[tuple[str, np.ndarray]]:
        """Partition a single column into labelled bins with boolean masks.

        Numeric columns are binned into up to ``self.n_bins`` quantiles via
        ``pd.qcut``.  If ``qcut`` fails (too many duplicate edges), falls
        back to ``pd.cut``.  If both fail, the column is treated as a single
        bin (all samples).

        Categorical / boolean columns yield one bin per unique value,
        skipping ``NaN``.

        Args:
            series:   Column data as a pandas Series.
            col_name: Name of the column (used for log messages).

        Returns:
            List of ``(bin_label, boolean_mask_array)`` tuples.
        """
        bins: list[tuple[str, np.ndarray]] = []

        if pd.api.types.is_numeric_dtype(series):
            bins = self._bin_numeric(series, col_name)
        else:
            bins = self._bin_categorical(series)

        return bins

    def _bin_numeric(
        self,
        series: pd.Series,
        col_name: str,
    ) -> list[tuple[str, np.ndarray]]:
        """Bin a numeric column into quantile-based groups.

        Strategy cascade:
        1. ``pd.qcut(series, n_bins, duplicates='drop')``
        2. ``pd.cut(series, n_bins)`` on failure
        3. Single bin containing all non-NaN samples on total failure

        Args:
            series:   Numeric pandas Series.
            col_name: Column name for logging.

        Returns:
            List of ``(bin_label, boolean_mask)`` tuples.
        """
        non_null = series.dropna()

        if len(non_null) == 0:
            return []

        # If all values are identical ⇒ single bin, no split possible
        if non_null.nunique() <= 1:
            mask = series.notna().values.astype(bool)
            return [(str(non_null.iloc[0]), mask)]

        # Attempt 1: qcut
        labels: pd.Series | None = None
        try:
            labels = pd.qcut(series, q=self.n_bins, duplicates="drop")
        except (ValueError, IndexError):
            pass

        # Attempt 2: pd.cut
        if labels is None:
            try:
                labels = pd.cut(series, bins=self.n_bins)
            except (ValueError, IndexError):
                pass

        # Attempt 3: single-bin fallback
        if labels is None:
            log.debug(
                "ive.subgroup_discovery.bin_fallback",
                column=col_name,
                strategy="single_bin",
            )
            mask = series.notna().values.astype(bool)
            return [("all", mask)]

        # Build mask for each bin label
        result: list[tuple[str, np.ndarray]] = []
        for cat in labels.cat.categories:
            mask = (labels == cat).values.astype(bool)
            if mask.sum() > 0:
                result.append((str(cat), mask))

        return result

    @staticmethod
    def _bin_categorical(series: pd.Series) -> list[tuple[str, np.ndarray]]:
        """Bin a categorical or boolean column by unique value.

        ``NaN`` values are skipped — they do not form their own bin.

        Args:
            series: Categorical / boolean / object pandas Series.

        Returns:
            List of ``(value_str, boolean_mask)`` tuples.
        """
        result: list[tuple[str, np.ndarray]] = []
        for val in series.dropna().unique():
            mask = (series == val).values.astype(bool)
            if mask.sum() > 0:
                result.append((str(val), mask))
        return result


# ---------------------------------------------------------------------------
# Legacy class kept for backward-compatibility with __init__.py exports.
# The new SubgroupDiscovery class above is the canonical Phase 3 API.
# ---------------------------------------------------------------------------


class SubgroupDiscoverer:
    """Beam-search subgroup discovery on the residual space.

    Iteratively builds conjunctive rules by selecting conditions that
    maximise WRAcc within a beam of width *W*.

    This class is retained for backward compatibility.  New callers should
    use :class:`SubgroupDiscovery` instead.
    """

    def __init__(
        self,
        beam_width: int = 10,
        search_depth: int = 3,
        min_coverage: float = 0.05,
        n_bins: int = 5,
    ) -> None:
        self.beam_width = beam_width
        self.search_depth = search_depth
        self.min_coverage = min_coverage
        self.n_bins = n_bins
        self._inner = SubgroupDiscovery(n_bins=n_bins)

    def discover(
        self,
        df: object,
        residuals: np.ndarray,
        feature_columns: list[str],
        top_k: int = 20,
    ) -> list[SubgroupPattern]:
        """Discover top-k subgroups with high systematic residual error.

        Delegates to :meth:`SubgroupDiscovery.detect` and wraps each
        result dict into a :class:`SubgroupPattern` dataclass.

        Args:
            df:              DataFrame containing *feature_columns*.
            residuals:       OOF residuals array ``(n_samples,)``.
            feature_columns: Columns to use as condition candidates.
            top_k:           Maximum number of patterns to return.

        Returns:
            List of :class:`SubgroupPattern` objects ordered by effect
            size descending.
        """
        if not isinstance(df, pd.DataFrame):
            log.warning("ive.subgroup_discovery.not_dataframe")
            return []

        X = df[feature_columns].copy()
        raw_patterns = self._inner.detect(X, residuals)

        patterns: list[SubgroupPattern] = []
        for p in raw_patterns[:top_k]:
            sp = SubgroupPattern(
                rule=f"{p['column_name']} in {p['bin_value']}",
                conditions=[{"column": p["column_name"], "bin": p["bin_value"]}],
                coverage=p["sample_count"] / max(1, len(residuals)),
                wracc=self._compute_wracc_from_stats(
                    p["sample_count"],
                    p["mean_residual"],
                    float(np.mean(residuals)),
                    len(residuals),
                ),
                mean_residual=p["mean_residual"],
                mean_residual_outside=float(np.mean(residuals)),
                effect_size=p["effect_size"],
            )
            patterns.append(sp)

        log.info("ive.subgroup_discovery.done", n_patterns=len(patterns))
        return patterns

    @staticmethod
    def _compute_wracc_from_stats(
        sample_count: int,
        mean_residual: float,
        global_mean: float,
        total_samples: int,
    ) -> float:
        """Compute WRAcc from pre-computed summary statistics.

        WRAcc = P(cond) × (mean_residual(cond) − mean_residual(all))

        Args:
            sample_count:  Number of samples in the subgroup.
            mean_residual: Mean residual within the subgroup.
            global_mean:   Mean residual over the entire dataset.
            total_samples: Total number of samples.

        Returns:
            WRAcc value as a float.
        """
        if total_samples == 0:
            return 0.0
        coverage = sample_count / total_samples
        return coverage * (mean_residual - global_mean)

    def _compute_wracc(self, mask: np.ndarray, residuals: np.ndarray) -> float:
        """Compute WRAcc for a given boolean mask.

        Args:
            mask:      Boolean array of shape ``(n_samples,)``.
            residuals: OOF residuals array.

        Returns:
            WRAcc value as a float.
        """
        coverage = float(mask.mean())
        if coverage < self.min_coverage:
            return 0.0
        return coverage * (float(np.mean(residuals[mask])) - float(np.mean(residuals)))

    @staticmethod
    def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cohen's *d* effect size between two groups.

        Uses the pooled standard deviation as the denominator.

        Args:
            group1: First sample array.
            group2: Second sample array.

        Returns:
            Cohen's *d* as a float.  Returns 0 if either group has < 2
            samples or pooled std is zero.
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        pooled_std = np.sqrt(
            ((n1 - 1) * group1.std() ** 2 + (n2 - 1) * group2.std() ** 2) / (n1 + n2 - 2)
        )
        if pooled_std == 0:
            return 0.0
        return float((group1.mean() - group2.mean()) / pooled_std)
