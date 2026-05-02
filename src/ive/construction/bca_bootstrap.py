"""Bias-corrected and accelerated (BCa) bootstrap confidence intervals.

Per plan §B4 + §29 + §101:
    - BCa is the default for subgroup N ≥ 100 — better small-sample
      coverage than the percentile method.
    - Below N = 100, BCa is unstable; we fall back to the simpler
      percentile bootstrap.
    - Selection-bias correction lives at the cross-fitting layer
      (planned for Phase B.5); these CIs are conditional on the
      subgroup definition.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

# Per plan §29/§101: the BCa method is unstable below N=100, so fall back
# to plain percentile in that regime.
BCA_MIN_N = 100
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class CIResult:
    """Confidence interval outcome with method provenance."""

    lower: float
    upper: float
    method: str  # "bca" | "percentile" | "degenerate"
    n_used: int


def _percentile_ci(
    bootstrap_stats: np.ndarray[Any, Any],
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Symmetric percentile CI."""
    finite = bootstrap_stats[np.isfinite(bootstrap_stats)]
    if finite.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(finite, 100 * alpha))
    hi = float(np.percentile(finite, 100 * (1.0 - alpha)))
    return lo, hi


def bca_confidence_interval(
    bootstrap_stats: np.ndarray[Any, Any],
    *,
    sample_data: np.ndarray[Any, Any] | None = None,
    sample_statistic: Callable[[np.ndarray[Any, Any]], float] | None = None,
    point_estimate: float | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
) -> CIResult:
    """Compute a BCa confidence interval for a bootstrap distribution.

    Args:
        bootstrap_stats: Array of bootstrap statistic values, one per
            resample. NaN entries are dropped before estimation.
        sample_data: The original sample (rows × features or 1-D
            array). Required for BCa's acceleration term via jackknife;
            when omitted, the result falls back to percentile method.
        sample_statistic: Callable that maps an array slice to the same
            statistic used to populate ``bootstrap_stats``. Used for
            jackknife acceleration. Required alongside ``sample_data``.
        point_estimate: The statistic value computed on the original
            (non-resampled) sample. Required for the bias-correction
            term; falls back to ``np.median(bootstrap_stats)`` when None.
        confidence: Desired confidence level (default 0.95).

    Returns:
        :class:`CIResult` carrying ``(lower, upper, method, n_used)``.
        ``method`` is one of:

        - ``"bca"`` — full BCa, only when N ≥ ``BCA_MIN_N`` and a usable
          sample / statistic is available.
        - ``"percentile"`` — fallback for small N or when jackknife data
          is unavailable.
        - ``"degenerate"`` — empty / all-NaN bootstrap distribution; the
          returned ``(lower, upper)`` are both NaN.
    """
    finite = np.asarray(bootstrap_stats, dtype=float)
    finite = finite[np.isfinite(finite)]
    n_used = int(finite.size)

    if n_used == 0:
        return CIResult(float("nan"), float("nan"), "degenerate", 0)

    n_total = (
        int(np.asarray(sample_data).shape[0]) if sample_data is not None else n_used
    )

    # Below the BCa stability threshold, percentile is more reliable.
    if n_total < BCA_MIN_N or sample_data is None or sample_statistic is None:
        lo, hi = _percentile_ci(finite, confidence)
        return CIResult(lo, hi, "percentile", n_used)

    # ── Bias correction ────────────────────────────────────────────────
    # Wave 3 audit fix: a non-finite point_estimate (NaN / ±inf) makes
    # the comparison `finite < pe` degenerate, which produces a fake
    # single-point CI. Fall back to the bootstrap median when that
    # happens — keeps BCa available without trusting the upstream
    # point estimate.
    if point_estimate is None or not np.isfinite(point_estimate):
        pe = float(np.median(finite))
    else:
        pe = float(point_estimate)
    # If after substitution we still have a degenerate value (every
    # bootstrap stat identical), fall back to percentile instead of
    # producing a one-point BCa interval.
    if not np.isfinite(pe):
        lo, hi = _percentile_ci(finite, confidence)
        return CIResult(lo, hi, "percentile", n_used)
    fraction_below = float(np.mean(finite < pe))
    # Clip to avoid ±inf when the distribution is degenerate.
    fraction_below = float(np.clip(fraction_below, 1e-6, 1.0 - 1e-6))
    z0 = float(stats.norm.ppf(fraction_below))

    # ── Acceleration via jackknife on the original sample ─────────────
    sample = np.asarray(sample_data)
    n = sample.shape[0]
    jack = np.empty(n, dtype=float)
    for i in range(n):
        loo = np.delete(sample, i, axis=0)
        try:
            jack[i] = float(sample_statistic(loo))
        except Exception:
            jack[i] = np.nan
    jack = jack[np.isfinite(jack)]
    if jack.size == 0:
        # Jackknife failed; degrade to percentile.
        lo, hi = _percentile_ci(finite, confidence)
        return CIResult(lo, hi, "percentile", n_used)

    jack_mean = float(np.mean(jack))
    diffs = jack_mean - jack
    num = float(np.sum(diffs**3))
    den = 6.0 * (float(np.sum(diffs**2)) ** 1.5)
    a = num / den if den > 0 else 0.0

    # ── BCa-adjusted percentiles ──────────────────────────────────────
    alpha = (1.0 - confidence) / 2.0
    z_alpha_lo = float(stats.norm.ppf(alpha))
    z_alpha_hi = float(stats.norm.ppf(1.0 - alpha))

    def _adjusted(z_alpha: float) -> float:
        denom = 1.0 - a * (z0 + z_alpha)
        if denom == 0:
            return alpha
        return float(stats.norm.cdf(z0 + (z0 + z_alpha) / denom))

    pct_lo = max(0.0, min(1.0, _adjusted(z_alpha_lo)))
    pct_hi = max(0.0, min(1.0, _adjusted(z_alpha_hi)))

    if pct_lo >= pct_hi:
        # Numerical instability — fall back to percentile.
        lo, hi = _percentile_ci(finite, confidence)
        return CIResult(lo, hi, "percentile", n_used)

    lo = float(np.percentile(finite, 100 * pct_lo))
    hi = float(np.percentile(finite, 100 * pct_hi))
    return CIResult(lo, hi, "bca", n_used)


__all__ = ["BCA_MIN_N", "DEFAULT_CONFIDENCE", "CIResult", "bca_confidence_interval"]
