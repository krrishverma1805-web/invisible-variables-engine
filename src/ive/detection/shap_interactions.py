"""
SHAP Interaction Analysis.

Computes SHAP values and SHAP interaction values to identify:
    1. Which features most influence model predictions
    2. Which feature *pairs* have significant joint effects (interactions)

These insights guide subgroup discovery and latent variable naming in Phase 4.
"""

from __future__ import annotations
from typing import Any

from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class SHAPResult:
    """Container for SHAP analysis outputs."""

    shap_values: np.ndarray[Any, Any]  # (n_samples, n_features)
    shap_interaction_values: np.ndarray[Any, Any] | None = None  # (n_samples, n_features, n_features)
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

        TODO:
            - Subsample X if len(X) > self.sample_size (random seed for reproducibility)
            - Call model.get_shap_values(X_sample) → shap_values
            - If compute_interactions and hasattr(model, 'get_shap_interaction_values'):
                  interactions = model.get_shap_interaction_values(X_sample)
                  top_pairs = self._rank_interaction_pairs(interactions, feature_names)
            - Compute mean_abs_shap per feature
        """
        n = min(self.sample_size, len(X))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n, replace=False)
        X_sample = X[idx]

        log.info("ive.shap.compute", n_samples=n, n_features=X.shape[1])

        # TODO: Call model.get_shap_values(X_sample)
        shap_values = np.zeros_like(X_sample)  # placeholder

        mean_abs = {
            fname: float(np.mean(np.abs(shap_values[:, i])))
            for i, fname in enumerate(feature_names)
        }

        interaction_values: np.ndarray[Any, Any] | None = None
        top_pairs: list[tuple[str, str, float]] = []

        # TODO: Compute interactions for XGBoost models
        # if compute_interactions and hasattr(model, 'get_shap_interaction_values'):
        #     interaction_values = model.get_shap_interaction_values(X_sample)
        #     top_pairs = self._rank_interaction_pairs(interaction_values, feature_names, top_k=10)

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

        TODO:
            - Average |interaction_values| over samples: mean_interaction[i, j]
            - Zero out diagonal (self-interaction)
            - Return top_k pairs by mean_interaction strength
        """
        # TODO: Implement pair ranking
        return []
