"""SHAP-interaction-derived features for subgroup discovery (Phase B7).

Plan reference: §B7 + §99.

After Phase 2 produces SHAP interaction pairs, we synthesize three
numerically-stable derived columns per pair so subgroup discovery can
treat the interaction as a first-class feature:

1. **product** ``a * b``
2. **high-high quadrant indicator** ``(a > median(a)) AND (b > median(b))``
3. **mixed-quadrant XOR** ``(a > median(a)) XOR (b > median(b))``

The ratio ``a / (b + ε)`` was rejected (per plan §99): unstable when
``b`` straddles zero; downstream KS tests get dominated by the resulting
outliers.

The BH correction over the family-wise union of (original-feature
subgroup tests + interaction-derived subgroup tests) is applied at the
existing pattern_scorer pass (already family-wise; nothing to do here).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class InteractionPair:
    """A pair of feature names with their SHAP-interaction strength."""

    feature_a: str
    feature_b: str
    interaction_strength: float


def select_top_interactions(
    pairs: list[tuple[str, str, float]],
    *,
    top_k: int = 5,
    min_strength: float = 0.0,
) -> list[InteractionPair]:
    """Filter and sort SHAP-interaction pairs.

    Args:
        pairs: ``(feature_a, feature_b, strength)`` tuples as produced
            by :class:`SHAPInteractionAnalyzer._rank_interaction_pairs`.
        top_k: Maximum number to keep.
        min_strength: Drop pairs whose absolute strength is below this.

    Returns:
        At most ``top_k`` pairs sorted by descending absolute strength.
        Self-pairs (``feature_a == feature_b``) are filtered out.
    """
    out: list[InteractionPair] = []
    for a, b, strength in pairs:
        if a == b:
            continue
        if abs(float(strength)) < min_strength:
            continue
        out.append(InteractionPair(a, b, float(strength)))
    out.sort(key=lambda p: abs(p.interaction_strength), reverse=True)
    return out[:top_k]


def synthesize_interaction_features(
    X: pd.DataFrame,
    pairs: list[InteractionPair],
) -> pd.DataFrame:
    """Append three derived columns per interaction pair to a copy of X.

    Args:
        X: Feature DataFrame.
        pairs: Selected interaction pairs.

    Returns:
        New DataFrame with extra columns named:

        - ``__ix__{a}__x__{b}`` — product
        - ``__ix__{a}__hh__{b}`` — high-high quadrant indicator (0/1)
        - ``__ix__{a}__xor__{b}`` — mixed-quadrant XOR indicator (0/1)

        The ``__ix__`` prefix lets downstream filters distinguish
        derived from original columns. Pairs whose features are
        missing or non-numeric are skipped with a warning.
    """
    augmented = X.copy()
    for pair in pairs:
        a, b = pair.feature_a, pair.feature_b
        if a not in X.columns or b not in X.columns:
            log.warning(
                "ive.interaction.skip_missing_feature",
                feature_a=a,
                feature_b=b,
            )
            continue
        col_a = pd.to_numeric(X[a], errors="coerce")
        col_b = pd.to_numeric(X[b], errors="coerce")
        if col_a.isna().all() or col_b.isna().all():
            log.warning(
                "ive.interaction.skip_non_numeric",
                feature_a=a,
                feature_b=b,
            )
            continue

        product_col = f"__ix__{a}__x__{b}"
        hh_col = f"__ix__{a}__hh__{b}"
        xor_col = f"__ix__{a}__xor__{b}"

        med_a = float(col_a.median())
        med_b = float(col_b.median())

        product = (col_a * col_b).astype(float).fillna(0.0)
        hh = (((col_a > med_a) & (col_b > med_b)).astype(int)).fillna(0)
        # XOR: exactly one of the two is above its median.
        xor = (((col_a > med_a) ^ (col_b > med_b)).astype(int)).fillna(0)

        augmented[product_col] = product
        augmented[hh_col] = hh
        augmented[xor_col] = xor

    return augmented


def is_interaction_column(name: str) -> bool:
    """True when ``name`` is one of the synthesized interaction columns."""
    return name.startswith("__ix__")


__all__ = [
    "InteractionPair",
    "is_interaction_column",
    "select_top_interactions",
    "synthesize_interaction_features",
]
