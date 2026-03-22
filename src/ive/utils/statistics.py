"""
Statistics Utilities.

Common statistical helper functions used across the IVE pipeline.
Wraps scipy/statsmodels calls with consistent error handling and
sensible defaults.
"""

from __future__ import annotations
from typing import Any

import numpy as np


def cohens_d(group1: np.ndarray[Any, Any], group2: np.ndarray[Any, Any]) -> float:
    """
    Compute Cohen's d effect size between two independent groups.

    Args:
        group1: Residuals for the first group (e.g., inside a cluster).
        group2: Residuals for the second group (e.g., outside a cluster).

    Returns:
        Cohen's d value. Sign indicates direction.
        |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std() ** 2 + (n2 - 1) * group2.std() ** 2) / (n1 + n2 - 2)
    )
    return float((group1.mean() - group2.mean()) / pooled_std) if pooled_std != 0 else 0.0


def cramers_v(confusion_matrix: np.ndarray[Any, Any]) -> float:
    """
    Compute Cramér's V association measure for categorical variables.

    Args:
        confusion_matrix: 2D contingency table as numpy array.

    Returns:
        Cramér's V in [0, 1]. 0 = no association, 1 = perfect.

    TODO:
        - chi2, n, min_dim = scipy.stats.chi2_contingency(confusion_matrix)[:3], ...
        - return sqrt(chi2 / (n * (min_dim - 1)))
    """
    from scipy.stats import chi2_contingency

    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum()
    min_dim = min(confusion_matrix.shape) - 1
    if n == 0 or min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def permutation_test(
    observed_stat: float,
    group1: np.ndarray[Any, Any],
    group2: np.ndarray[Any, Any],
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """
    Compute two-sided p-value via permutation test.

    Args:
        observed_stat: The test statistic computed on the real data.
        group1: First group values.
        group2: Second group values.
        n_permutations: Number of random permutations.
        seed: Random seed for reproducibility.

    Returns:
        p-value (fraction of permutations with |stat| >= |observed_stat|).

    TODO:
        - Combine groups, shuffle, split, recompute statistic n_permutations times
        - Return proportion ≥ |observed_stat|
    """
    rng = np.random.default_rng(seed)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = float(cohens_d(combined[:n1], combined[n1:]))
        if abs(perm_stat) >= abs(observed_stat):
            count += 1
    return count / n_permutations


def confidence_interval_bootstrap(
    data: np.ndarray[Any, Any],
    statistic_fn: object,  # Callable[[np.ndarray], float]
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a scalar statistic.

    Args:
        data: Input data array.
        statistic_fn: A callable that takes a 1D array and returns a float.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 = 95% CI).
        seed: Random seed.

    Returns:
        (ci_lower, ci_upper) tuple.

    TODO:
        - Resample data with replacement n_bootstrap times
        - Apply statistic_fn to each resample
        - Return (percentile(alpha/2), percentile(1-alpha/2))
    """
    rng = np.random.default_rng(seed)
    stats = [
        statistic_fn(rng.choice(data, size=len(data), replace=True))  # type: ignore[operator]
        for _ in range(n_bootstrap)
    ]
    arr = np.array(stats)
    return float(np.percentile(arr, 100 * alpha / 2)), float(
        np.percentile(arr, 100 * (1 - alpha / 2))
    )


def normalise_scores(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Normalise an array to [0, 1] range (min-max)."""
    from typing import cast
    rng = scores.max() - scores.min()
    if rng == 0:
        return np.zeros_like(scores, dtype=float)
    return cast(np.ndarray[Any, Any], (scores - scores.min()) / rng)
