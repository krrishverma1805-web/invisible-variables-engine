"""
HDBSCAN Clustering for Residual Space Analysis.

Clusters samples in the joint (residual, feature) space to find dense
groups of samples that share similar systematic error patterns. These
clusters are strong candidates for latent variable subgroups.

HDBSCAN is preferred over K-Means because:
    - Does not require specifying k in advance
    - Handles noise points (label = -1) gracefully
    - Finds clusters of variable density
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class ClusteringResult:
    """Output of HDBSCAN clustering."""

    labels: np.ndarray  # -1 = noise, 0..k = cluster id
    n_clusters: int = 0
    noise_fraction: float = 0.0
    cluster_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    validity_score: float = 0.0  # DBCV score (-1 to 1)


class HDBSCANClusterer:
    """
    Clusters the residual-feature space using HDBSCAN.

    The input is constructed by concatenating normalised residuals with
    normalised feature values, so that both contribute to cluster shape.
    """

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        cluster_selection_method: str = "eom",
        metric: str = "euclidean",
    ) -> None:
        """
        Args:
            min_cluster_size: Minimum number of samples to form a cluster.
            min_samples: Robustness parameter (defaults to min_cluster_size).
            cluster_selection_method: 'eom' (excess of mass) or 'leaf'.
            metric: Distance metric for HDBSCAN.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self._clusterer: object | None = None

    def fit(
        self,
        residuals: np.ndarray,
        feature_matrix: np.ndarray,
        residual_weight: float = 2.0,
    ) -> ClusteringResult:
        """
        Fit HDBSCAN on the combined residual-feature space.

        Args:
            residuals: Shape (n_samples,) — OOF residuals.
            feature_matrix: Shape (n_samples, n_features) — preprocessed features.
            residual_weight: Multiplier for residual dimension to boost its
                influence on cluster formation.

        Returns:
            ClusteringResult with cluster labels and statistics.

        TODO:
            - Import hdbscan
            - Normalise residuals and feature_matrix to zero mean, unit variance
            - Concatenate: X = np.hstack([residuals_scaled * residual_weight, features_scaled])
            - self._clusterer = hdbscan.HDBSCAN(min_cluster_size=..., ...)
            - self._clusterer.fit(X)
            - labels = self._clusterer.labels_
            - Compute cluster_stats per cluster (mean residual, std, coverage)
            - Compute validity_score via hdbscan.validity.validity_index(X, labels)
        """
        from sklearn.preprocessing import StandardScaler

        log.info(
            "ive.clustering.fit",
            n_samples=len(residuals),
            min_cluster_size=self.min_cluster_size,
        )

        # Normalise and combine residuals with features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_matrix)

        residuals_2d = residuals.reshape(-1, 1)
        residuals_scaled = StandardScaler().fit_transform(residuals_2d) * residual_weight
        combined = np.hstack([residuals_scaled, features_scaled])

        # TODO: Replace placeholder with actual HDBSCAN
        # import hdbscan
        # self._clusterer = hdbscan.HDBSCAN(
        #     min_cluster_size=self.min_cluster_size,
        #     min_samples=self.min_samples,
        #     cluster_selection_method=self.cluster_selection_method,
        #     metric=self.metric,
        # )
        # self._clusterer.fit(combined)
        # labels = self._clusterer.labels_

        # Placeholder: all noise until implemented
        labels = np.full(len(residuals), -1, dtype=int)

        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        n_clusters = len(unique)
        noise_fraction = float(np.mean(labels == -1))

        cluster_stats: dict[int, dict[str, float]] = {}
        for cid, count in zip(unique, counts, strict=False):
            mask = labels == cid
            cluster_stats[int(cid)] = {
                "size": int(count),
                "mean_residual": float(np.mean(residuals[mask])),
                "std_residual": float(np.std(residuals[mask])),
                "coverage_pct": float(count / len(residuals) * 100),
            }

        log.info(
            "ive.clustering.done",
            n_clusters=n_clusters,
            noise_pct=round(noise_fraction * 100, 1),
        )

        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            noise_fraction=noise_fraction,
            cluster_stats=cluster_stats,
        )
