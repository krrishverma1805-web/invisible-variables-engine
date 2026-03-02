"""
Residual Analyzer.

Analyzes the distribution and structure of model residuals to characterize
the nature of model errors. This informs Phase 3 detection strategies.

Analyses performed:
    - Basic statistics (mean, std, skew, kurtosis)
    - Heteroscedasticity test (Breusch-Pagan)
    - Normality test (Shapiro-Wilk on a sample)
    - Autocorrelation (Durbin-Watson, Ljung-Box)
    - Spatial structure detection (Moran's I if spatial columns present)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class ResidualAnalysis:
    """Container for the outputs of residual analysis."""

    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    max_abs: float = 0.0
    pct_large: float = 0.0           # % of |residual| > 2*std
    heteroscedastic: bool = False
    breusch_pagan_p: float | None = None
    normal: bool = False
    shapiro_p: float | None = None
    durbin_watson: float | None = None
    warnings: list[str] = field(default_factory=list)


class ResidualAnalyzer:
    """
    Characterizes model residual distributions.

    Used after cross-validation to understand *what kind* of errors
    remain, which guides which detection strategies to prioritise.
    """

    def __init__(self, large_residual_threshold: float = 2.0) -> None:
        """
        Args:
            large_residual_threshold: Number of standard deviations beyond
                which a residual is considered 'large'.
        """
        self.threshold = large_residual_threshold

    def analyze(self, residuals: np.ndarray, X: np.ndarray | None = None) -> ResidualAnalysis:
        """
        Perform full residual analysis.

        Args:
            residuals: 1-D array of residual values (y_true - y_pred).
            X: Optional feature matrix for heteroscedasticity tests.

        Returns:
            ResidualAnalysis dataclass with all computed statistics.

        TODO:
            - scipy.stats.skew / kurtosis for shape statistics
            - scipy.stats.shapiro (on a sample ≤ 5000) for normality
            - statsmodels.stats.diagnostic.het_breuschpagan for heteroscedasticity
            - statsmodels.stats.stattools.durbin_watson for autocorrelation
        """
        from scipy import stats

        analysis = ResidualAnalysis()

        if len(residuals) == 0:
            return analysis

        analysis.mean = float(np.mean(residuals))
        analysis.std = float(np.std(residuals))
        analysis.skewness = float(stats.skew(residuals))
        analysis.kurtosis = float(stats.kurtosis(residuals))
        analysis.max_abs = float(np.max(np.abs(residuals)))
        analysis.pct_large = float(
            np.mean(np.abs(residuals) > self.threshold * analysis.std) * 100
        )

        # TODO: Normality test (Shapiro-Wilk)
        # sample = residuals[:5000] if len(residuals) > 5000 else residuals
        # stat, p = stats.shapiro(sample)
        # analysis.shapiro_p = p
        # analysis.normal = p > 0.05

        # TODO: Heteroscedasticity test (Breusch-Pagan)
        # if X is not None:
        #     from statsmodels.stats.diagnostic import het_breuschpagan
        #     _, p, _, _ = het_breuschpagan(residuals, X)
        #     analysis.breusch_pagan_p = p
        #     analysis.heteroscedastic = p < 0.05

        # TODO: Autocorrelation (Durbin-Watson)
        # from statsmodels.stats.stattools import durbin_watson
        # analysis.durbin_watson = durbin_watson(residuals)

        if analysis.pct_large > 10:
            analysis.warnings.append(
                f"{analysis.pct_large:.1f}% of residuals exceed {self.threshold}σ — "
                "strong evidence of systematic error."
            )

        log.info(
            "ive.residual_analyzer.done",
            mean=round(analysis.mean, 4),
            std=round(analysis.std, 4),
            pct_large=round(analysis.pct_large, 2),
        )
        return analysis
