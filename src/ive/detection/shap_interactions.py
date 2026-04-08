"""
SHAP Interaction Analysis.

Computes SHAP values and SHAP interaction values to identify:
    1. Which features most influence model predictions
    2. Which feature *pairs* have significant joint effects (interactions)

These insights guide subgroup discovery and latent variable naming in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class SHAPResult:
    """Container for SHAP analysis outputs."""

    shap_values: np.ndarray[Any, Any]  # (n_samples, n_features)
    shap_interaction_values: np.ndarray[Any, Any] | None = (
        None  # (n_samples, n_features, n_features)
    )
    mean_abs_shap: dict[str, float] = field(default_factory=dict)
    top_interaction_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)


class SHAPInteractionAnalyzer:
    """
    Computes SHAP values and feature interaction effects.

    For large datasets, computation is performed on a random subsample
    to keep runtime tractable.
    """

    def __init__(self, sample_size: int = 500) -> None:
        """
        Args:
            sample_size: Maximum number of samples to use for SHAP (for speed).
        """
        self.sample_size = sample_size

    def compute(
        self,
        model: object,  # IVEModel instance (XGBoostIVEModel preferred)
        X: np.ndarray[Any, Any],
        feature_names: list[str],
        compute_interactions: bool = True,
    ) -> SHAPResult:
        """
        Compute SHAP values (and optionally interaction values).

        Args:
            model: A fitted IVEModel instance.
            X: Feature matrix (n_samples, n_features).
            feature_names: Names corresponding to columns of X.
            compute_interactions: If True, also compute interaction matrix.

        Returns:
            SHAPResult with main values and top interaction pairs.

        """
        n = min(self.sample_size, len(X))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n, replace=False)
        X_sample = X[idx]

        log.info("ive.shap.compute", n_samples=n, n_features=X.shape[1])

        try:
            shap_values = model.get_shap_values(X_sample)  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("ive.shap.compute_failed", error=str(exc))
            shap_values = np.zeros_like(X_sample)

        mean_abs = {
            fname: float(np.mean(np.abs(shap_values[:, i])))
            for i, fname in enumerate(feature_names)
        }

        interaction_values: np.ndarray[Any, Any] | None = None
        top_pairs: list[tuple[str, str, float]] = []

        if compute_interactions and hasattr(model, 'get_shap_interaction_values'):
            try:
                interaction_values = model.get_shap_interaction_values(X_sample)  # type: ignore[union-attr]
                top_pairs = self._rank_interaction_pairs(interaction_values, feature_names, top_k=10)
            except Exception as exc:
                log.warning("ive.shap.interactions_failed", error=str(exc))

        return SHAPResult(
            shap_values=shap_values,
            shap_interaction_values=interaction_values,
            mean_abs_shap=mean_abs,
            top_interaction_pairs=top_pairs,
            feature_names=feature_names,
        )

    def _rank_interaction_pairs(
        self,
        interaction_values: np.ndarray[Any, Any],
        feature_names: list[str],
        top_k: int = 10,
    ) -> list[tuple[str, str, float]]:
        """
        Rank feature pairs by mean absolute SHAP interaction strength.

        Args:
            interaction_values: Shape (n_samples, n_features, n_features).
            feature_names: Feature name list.
            top_k: Return only the top-k pairs.

        Returns:
            List of (feature_a, feature_b, interaction_strength) tuples.

        """
        # Average absolute interaction values over samples
        # interaction_values shape: (n_samples, n_features, n_features)
        mean_interaction = np.mean(np.abs(interaction_values), axis=0)

        # Zero out diagonal (self-interaction is not meaningful for pairs)
        np.fill_diagonal(mean_interaction, 0.0)

        n_features = len(feature_names)
        pairs: list[tuple[str, str, float]] = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                strength = float(mean_interaction[i, j])
                if strength > 0:
                    pairs.append((feature_names[i], feature_names[j], strength))

        # Sort by interaction strength descending
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_k]
