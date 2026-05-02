"""Variance-regime (heteroscedasticity) detection (Phase B8).

Plan reference: §B8 + §100.

Detects features along which the **conditional variance** of the
residuals shifts — i.e., the model is more or less reliable depending
on the feature's value. These regimes are operationally useful even
when the feature has no main effect: "the model gets worse on these
records" is itself an actionable signal.

The test is a **likelihood-ratio test of homoscedastic vs
heteroscedastic regression** on ``|residuals|``:

    H0:  |e| ~ const
    H1:  |e| ~ feature

Compute LR = -2 (loglik_null - loglik_alt) against χ² with 1 dof.
A feature passes when:

  1. LR is significant at ``lr_alpha`` (default 0.05), AND
  2. the spread ratio (max(|e_q|) / min(|e_q|) over feature quantile
     bins) exceeds ``min_spread_ratio`` (default 2.0).

This catches genuine variance regimes without the bimodality and
outlier false positives that a raw quantile-spread test would produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VarianceRegimePattern:
    """A detected variance-regime pattern."""

    pattern_type: str  # always "variance_regime"
    feature: str
    lr_statistic: float
    p_value: float
    spread_ratio: float
    n_samples: int
    bin_summary: dict[str, float]  # {bin_label: mean_abs_residual}


def _gaussian_loglik(residuals: np.ndarray[Any, Any]) -> float:
    """Loglikelihood of N(0, sigma^2) given absolute-residual data.

    Used as a coarse proxy for the loglik of the homoscedastic model.
    The MLE for sigma is the sample SD; substituting back yields a
    closed-form loglik.

    Returns ``nan`` for degenerate inputs (zero / non-finite variance,
    fewer than 2 samples). The caller must propagate this NaN through
    the LR computation — returning ``0.0`` would falsely make
    ``LR = -2 * (loglik_null - 0)`` positive whenever loglik_null is
    negative, which is always (Wave 4 audit fix).
    """
    n = residuals.size
    if n < 2:
        return float("nan")
    sigma2 = float(np.var(residuals, ddof=1))
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return float("nan")
    return float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))


def _per_bin_stats(
    abs_resid: np.ndarray[Any, Any],
    feature_values: np.ndarray[Any, Any],
    *,
    n_bins: int,
) -> dict[str, float]:
    """Compute mean |residual| within each quantile bin of ``feature_values``."""
    n = abs_resid.size
    if n == 0:
        return {}
    try:
        q_edges = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
        # Force monotonic edges (np.quantile with ties can repeat).
        q_edges = np.unique(q_edges)
        if q_edges.size < 2:
            return {}
        bin_idx = np.digitize(feature_values, q_edges[1:-1])
    except Exception:  # pragma: no cover - defensive
        return {}

    summary: dict[str, float] = {}
    for b in range(int(bin_idx.max()) + 1):
        mask = bin_idx == b
        if mask.sum() < 2:
            continue
        summary[f"bin_{b}"] = float(np.mean(abs_resid[mask]))
    return summary


class VarianceRegimeDetector:
    """Detect heteroscedastic-regime features via likelihood-ratio test.

    Attributes:
        min_spread_ratio: Filters out features whose per-bin |residual|
            ratio is below this. Default 2.0 means "at least 2× higher
            variance in the worst bin vs the best."
        lr_alpha: Significance level for the LR test (default 0.05).
        n_bins: Quantile bin count used for the spread-ratio diagnostic
            (default 5).
    """

    def __init__(
        self,
        min_spread_ratio: float = 2.0,
        lr_alpha: float = 0.05,
        n_bins: int = 5,
    ) -> None:
        self.min_spread_ratio = float(min_spread_ratio)
        self.lr_alpha = float(lr_alpha)
        self.n_bins = int(n_bins)

    def detect(
        self,
        X: pd.DataFrame,
        residuals: np.ndarray[Any, Any],
        feature_names: list[str] | None = None,
    ) -> list[VarianceRegimePattern]:
        """Run the detector on a residual array.

        Args:
            X: Feature DataFrame; rows align with ``residuals``.
            residuals: 1-D residual array (regression residuals or
                signed deviance for classification — both work because
                the test is on |e|).
            feature_names: Optional subset to test. Default: all numeric
                columns in X. Caller can pass top-N-by-SHAP names per
                plan §B8 to bound runtime.

        Returns:
            List of :class:`VarianceRegimePattern`, one per surviving
            feature, sorted by descending LR statistic.
        """
        if len(X) != len(residuals):
            raise ValueError(
                f"VarianceRegimeDetector.detect: shape mismatch X={len(X)} "
                f"vs residuals={len(residuals)}"
            )
        abs_resid = np.abs(np.asarray(residuals, dtype=float))
        finite_mask = np.isfinite(abs_resid)
        if finite_mask.sum() < 4:
            log.warning("ive.variance_regime.too_few_finite", n=int(finite_mask.sum()))
            return []
        abs_resid_f = abs_resid[finite_mask]

        if feature_names is None:
            feature_names = [
                c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
            ]

        results: list[VarianceRegimePattern] = []
        loglik_null = _gaussian_loglik(abs_resid_f)
        # If the null model itself is degenerate (no variance in
        # |residuals|, e.g. all residuals are exactly zero) there's
        # nothing to compare against. Return empty rather than risking
        # spurious LR values further down.
        if not np.isfinite(loglik_null):
            log.warning("ive.variance_regime.degenerate_null", abs_resid_var=0.0)
            return []

        for feature in feature_names:
            if feature not in X.columns:
                continue
            col = pd.to_numeric(X[feature], errors="coerce").to_numpy(dtype=float)
            col_f = col[finite_mask]
            row_mask = np.isfinite(col_f)
            if row_mask.sum() < 4:
                continue

            y = abs_resid_f[row_mask]
            x = col_f[row_mask]

            # Alternative model: |e| ~ a + b·x. Fit by OLS and compute
            # gaussian loglik of residuals around the fit line.
            try:
                slope, intercept, _r, _p, _se = stats.linregress(x, y)
            except Exception:  # pragma: no cover - defensive
                continue
            preds = slope * x + intercept
            res_alt = y - preds
            loglik_alt = _gaussian_loglik(res_alt)

            # Skip when either loglik is degenerate (NaN) — the LR
            # statistic is meaningless without two finite likelihoods.
            if not (np.isfinite(loglik_null) and np.isfinite(loglik_alt)):
                continue

            lr = -2.0 * (loglik_null - loglik_alt)
            if not np.isfinite(lr) or lr <= 0:
                continue

            # Wilks' theorem: 1 dof for the slope.
            p_value = float(stats.chi2.sf(lr, df=1))
            if p_value >= self.lr_alpha:
                continue

            # Diagnostic: per-bin |residual| spread.
            bin_summary = _per_bin_stats(y, x, n_bins=self.n_bins)
            if not bin_summary:
                continue
            min_bin = min(bin_summary.values())
            max_bin = max(bin_summary.values())
            if min_bin <= 0:
                spread_ratio = float("inf")
            else:
                spread_ratio = max_bin / min_bin
            if spread_ratio < self.min_spread_ratio:
                continue

            results.append(
                VarianceRegimePattern(
                    pattern_type="variance_regime",
                    feature=feature,
                    lr_statistic=float(lr),
                    p_value=p_value,
                    spread_ratio=float(spread_ratio),
                    n_samples=int(row_mask.sum()),
                    bin_summary=bin_summary,
                )
            )

        results.sort(key=lambda p: p.lr_statistic, reverse=True)
        log.info(
            "ive.variance_regime.complete",
            n_features_tested=len(feature_names),
            n_patterns=len(results),
        )
        return results


__all__ = [
    "VarianceRegimeDetector",
    "VarianceRegimePattern",
]
