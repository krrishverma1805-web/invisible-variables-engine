"""
Temporal Analysis.

Detects time-dependent systematic error patterns in the residuals,
which could indicate a latent variable that changes over time (e.g.,
seasonality, regime shifts, trending confounders).

Analyses:
    - Residuals binned by time period → compare means across bins
    - Rolling-window residual mean → detect drift
    - ACF/PACF of residuals → detect autocorrelation structure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class TemporalPattern:
    """A detected temporal pattern in the residuals."""

    column: str  # The datetime column used for analysis
    pattern_type: str  # 'trend' | 'seasonality' | 'regime_shift'
    description: str = ""
    affected_period: str = ""
    effect_size: float = 0.0
    p_value: float = 1.0


class TemporalAnalyzer:
    """
    Detects time-based residual patterns that suggest temporal latent variables.

    Only active when the dataset contains datetime columns (detected by Phase 1).
    """

    def __init__(self, n_bins: int = 10, window_size: int = 50) -> None:
        self.n_bins = n_bins
        self.window_size = window_size

    def analyze(
        self,
        df: object,  # pd.DataFrame
        residuals: np.ndarray[Any, Any],
        datetime_columns: list[str],
    ) -> list[TemporalPattern]:
        """
        Analyze each datetime column for temporal error patterns.

        Args:
            df: DataFrame containing datetime_columns.
            residuals: OOF residuals.
            datetime_columns: Columns detected as datetime by DataProfiler.

        Returns:
            List of TemporalPattern objects (empty if no datetime columns).
        """
        if not datetime_columns:
            return []

        import pandas as pd

        patterns: list[TemporalPattern] = []
        log.info("ive.temporal.analyze", n_datetime_cols=len(datetime_columns))

        for col in datetime_columns:
            # --- 1. Sort data by datetime column ---
            try:
                time_series = pd.to_datetime(df[col], errors="coerce")  # type: ignore[index]
            except Exception:
                log.warning("ive.temporal.parse_fail", column=col)
                continue

            valid_mask = ~time_series.isna()
            if valid_mask.sum() < 20:
                log.debug("ive.temporal.skip_few_valid", column=col, n_valid=int(valid_mask.sum()))
                continue

            # Check if all timestamps are identical
            unique_times = time_series[valid_mask].nunique()
            if unique_times <= 1:
                log.debug("ive.temporal.skip_constant", column=col)
                continue

            sorted_idx = time_series[valid_mask].argsort()
            sorted_residuals = np.asarray(residuals[valid_mask][sorted_idx])
            sorted_times = time_series[valid_mask].iloc[sorted_idx]

            # --- 2. Trend detection (Kendall's tau) ---
            try:
                from scipy.stats import kendalltau

                bins = np.array_split(sorted_residuals, self.n_bins)
                bin_means = [float(np.mean(b)) for b in bins if len(b) > 0]
                bin_indices = list(range(len(bin_means)))

                tau, p_value = kendalltau(bin_indices, bin_means)
                if p_value < 0.05 and abs(tau) > 0.3:
                    direction = "increasing" if tau > 0 else "decreasing"
                    patterns.append(
                        TemporalPattern(
                            column=col,
                            pattern_type="trend",
                            description=f"Residuals show a {direction} trend over time (tau={tau:.3f})",
                            affected_period=f"{sorted_times.iloc[0]} to {sorted_times.iloc[-1]}",
                            effect_size=abs(float(tau)),
                            p_value=float(p_value),
                        )
                    )
            except Exception:
                log.debug("ive.temporal.kendall_fail", column=col)

            # --- 3. Variance regime shift (Levene's test) ---
            try:
                from scipy.stats import levene

                bins_for_levene = [
                    b for b in np.array_split(sorted_residuals, self.n_bins) if len(b) >= 5
                ]
                if len(bins_for_levene) >= 2:
                    stat, p_value = levene(*bins_for_levene)
                    if p_value < 0.05:
                        variances = [float(np.var(b)) for b in bins_for_levene]
                        max_var_idx = int(np.argmax(variances))
                        patterns.append(
                            TemporalPattern(
                                column=col,
                                pattern_type="regime_shift",
                                description=(
                                    f"Residual variance changes significantly across time periods "
                                    f"(Levene stat={stat:.3f})"
                                ),
                                affected_period=(
                                    f"Period {max_var_idx + 1} of {len(bins_for_levene)} "
                                    f"has highest variance"
                                ),
                                effect_size=float(max(variances) / max(min(variances), 1e-10)),
                                p_value=float(p_value),
                            )
                        )
            except Exception:
                log.debug("ive.temporal.levene_fail", column=col)

            # --- 4. Autocorrelation (Ljung-Box test) ---
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox

                if len(sorted_residuals) >= 20:
                    lb_result = acorr_ljungbox(
                        sorted_residuals,
                        lags=min(10, len(sorted_residuals) // 5),
                        return_df=True,
                    )
                    significant_lags = lb_result[lb_result["lb_pvalue"] < 0.05]
                    if len(significant_lags) > 0:
                        min_p = float(significant_lags["lb_pvalue"].min())
                        first_sig_lag = int(significant_lags.index[0])
                        patterns.append(
                            TemporalPattern(
                                column=col,
                                pattern_type="trend",
                                description=(
                                    f"Significant residual autocorrelation detected at lag "
                                    f"{first_sig_lag} (Ljung-Box p={min_p:.4f})"
                                ),
                                affected_period=f"Lag {first_sig_lag}",
                                effect_size=float(lb_result.loc[first_sig_lag, "lb_stat"]),
                                p_value=min_p,
                            )
                        )
            except Exception:
                log.debug("ive.temporal.ljungbox_fail", column=col)

            # --- 5. Rolling-window drift detection ---
            try:
                if len(sorted_residuals) >= self.window_size * 2:
                    from scipy.stats import spearmanr

                    rolling_mean = (
                        pd.Series(sorted_residuals)
                        .rolling(window=self.window_size)
                        .mean()
                        .dropna()
                    )
                    time_index = np.arange(len(rolling_mean))
                    corr, p_value = spearmanr(time_index, rolling_mean.values)
                    if p_value < 0.05 and abs(corr) > 0.3:
                        direction = "upward" if corr > 0 else "downward"
                        patterns.append(
                            TemporalPattern(
                                column=col,
                                pattern_type="trend",
                                description=(
                                    f"Rolling-window residual mean shows {direction} drift "
                                    f"(Spearman r={corr:.3f})"
                                ),
                                affected_period=f"Window size: {self.window_size}",
                                effect_size=abs(float(corr)),
                                p_value=float(p_value),
                            )
                        )
            except Exception:
                log.debug("ive.temporal.rolling_drift_fail", column=col)

        log.info("ive.temporal.done", n_patterns=len(patterns))
        return patterns
