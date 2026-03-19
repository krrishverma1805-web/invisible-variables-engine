"""
HDBSCAN Clustering — Phase 3 Detection Engine.

Clusters the **worst-performing** samples (highest absolute residuals) to
discover dense groups that share similar feature-space structure.  These
clusters are strong candidates for latent variable subgroups because they
represent samples the model systematically fails on *and* that are
geometrically close in feature space.

Algorithm
---------
1. Retain only the top-20 % absolute-error samples (≥ 80th percentile).
2. Isolate numeric features and apply ``StandardScaler``.
3. Fit ``hdbscan.HDBSCAN`` on the scaled feature matrix.
4. For each cluster (label ≠ −1), record:
   • cluster center (unscaled feature means),
   • mean & std of absolute residuals,
   • sample count.

HDBSCAN is preferred over K-Means because it does not require specifying
*k* in advance, handles noise points (label = −1), and finds clusters of
variable density.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.preprocessing import StandardScaler

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
_HIGH_ERROR_PERCENTILE: int = 80  # keep the worst 20 %
_MIN_SAMPLES_FOR_CLUSTERING: int = 30  # below this, clustering is pointless


# ---------------------------------------------------------------------------
# Legacy dataclass kept for backward-compatibility with __init__.py exports.
# ---------------------------------------------------------------------------


@dataclass
class ClusteringResult:
    """Output of the legacy :class:`HDBSCANClusterer.fit` method."""

    labels: np.ndarray
    n_clusters: int = 0
    noise_fraction: float = 0.0
    cluster_stats: dict[int, dict[str, float]] = field(default_factory=dict)
    validity_score: float = 0.0


# ---------------------------------------------------------------------------
# New Phase-3 class requested by the user brief
# ---------------------------------------------------------------------------


class HDBSCANClustering:
    """Cluster high-error samples to discover latent-variable subgroups.

    Focuses exclusively on the *worst* 20 % of prediction errors,
    standardises numeric features, and runs HDBSCAN to find dense pockets
    of jointly-similar high-error samples.

    Args:
        high_error_percentile: Percentile threshold above which samples are
                               retained for clustering (default 80).
        min_samples_for_clustering: Minimum filtered samples required to
                                    attempt clustering (default 30).
    """

    def __init__(
        self,
        high_error_percentile: int = _HIGH_ERROR_PERCENTILE,
        min_samples_for_clustering: int = _MIN_SAMPLES_FOR_CLUSTERING,
    ) -> None:
        self.high_error_percentile = high_error_percentile
        self.min_samples_for_clustering = min_samples_for_clustering

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        X: pd.DataFrame,
        abs_residuals: np.ndarray,
        min_cluster_size: int = 15,
    ) -> list[dict[str, Any]]:
        """Discover clusters among high-error samples.

        Args:
            X:                Feature DataFrame (n_samples × n_features).
            abs_residuals:    Absolute OOF residuals, shape ``(n_samples,)``.
            min_cluster_size: Minimum samples for HDBSCAN to form a cluster.

        Returns:
            List of cluster pattern dicts sorted by ``mean_error``
            descending.  Each dict contains:

            * ``pattern_type``   — always ``"cluster"``
            * ``cluster_id``     — integer label (0-based)
            * ``sample_count``   — number of samples in the cluster
            * ``mean_error``     — mean absolute residual within the cluster
            * ``std_error``      — std of absolute residuals
            * ``cluster_center`` — dict mapping numeric column name → mean
              (in original, unscaled units)

            Returns ``[]`` if there are too few samples, no numeric columns,
            or HDBSCAN finds no clusters.
        """
        import hdbscan as _hdbscan

        abs_residuals = np.asarray(abs_residuals, dtype=np.float64)

        if len(X) != len(abs_residuals):
            raise ValueError(
                f"X has {len(X)} rows but abs_residuals has {len(abs_residuals)} entries."
            )

        if len(abs_residuals) == 0:
            log.info("ive.clustering.skip_empty")
            return []

        # ── Step 2: Filter to the worst 20 % ──────────────────────────
        threshold = float(np.percentile(abs_residuals, self.high_error_percentile))
        high_mask = abs_residuals >= threshold

        X_high = X.loc[high_mask].reset_index(drop=True)
        abs_high = abs_residuals[high_mask]

        if len(X_high) < self.min_samples_for_clustering:
            log.info(
                "ive.clustering.too_few_samples",
                n_filtered=len(X_high),
                threshold=self.min_samples_for_clustering,
            )
            return []

        # ── Step 4: Isolate numeric columns ────────────────────────────
        numeric_cols = X_high.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            log.info("ive.clustering.no_numeric_columns")
            return []

        X_numeric = X_high[numeric_cols].copy()

        # ── Step 5: Scale + NaN → 0 ───────────────────────────────────
        scaler = StandardScaler()
        scaled = scaler.fit_transform(X_numeric.values.astype(np.float64))
        np.nan_to_num(scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        log.info(
            "ive.clustering.fit",
            n_samples=len(scaled),
            n_features=len(numeric_cols),
            min_cluster_size=min_cluster_size,
        )

        # ── Step 6: Run HDBSCAN ────────────────────────────────────────
        clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
        )
        clusterer.fit(scaled)
        labels: np.ndarray = clusterer.labels_

        unique_labels = set(labels.tolist())
        unique_labels.discard(-1)

        if not unique_labels:
            log.info("ive.clustering.no_clusters_found")
            return []

        # ── Step 7: Build result dicts with hardened acceptance ────────
        #
        # Compute global mean error across the entire high-error subset.
        # Only keep clusters that show a *meaningful* uplift vs the
        # global error baseline — this eliminates noise-derived clusters.
        # -----------------------------------------------------------------
        global_mean_error = float(np.mean(abs_high))
        patterns: list[dict[str, Any]] = []

        for cluster_id in sorted(unique_labels):
            cluster_mask = labels == cluster_id
            n_in_cluster = int(cluster_mask.sum())

            if n_in_cluster == 0:
                continue

            # Gate 1: minimum cluster size
            if n_in_cluster < min_cluster_size:
                log.debug(
                    "ive.clustering.reject_small",
                    cluster_id=cluster_id,
                    size=n_in_cluster,
                    min_size=min_cluster_size,
                )
                continue

            # Unscaled feature means → cluster center
            cluster_center: dict[str, float] = {}
            for col in numeric_cols:
                col_values = X_numeric.loc[cluster_mask, col].values.astype(float)
                col_values = col_values[~np.isnan(col_values)]
                cluster_center[col] = float(np.mean(col_values)) if len(col_values) > 0 else 0.0

            # Gate 5: non-empty center
            if not cluster_center:
                log.debug(
                    "ive.clustering.reject_empty_center",
                    cluster_id=cluster_id,
                )
                continue

            # Abs-residual statistics
            cluster_errors = abs_high[cluster_mask]
            mean_error = float(np.mean(cluster_errors))
            std_error = float(np.std(cluster_errors, ddof=0)) if len(cluster_errors) > 1 else 0.0

            # Gate 2: cluster must outperform global baseline
            if mean_error <= global_mean_error:
                log.debug(
                    "ive.clustering.reject_below_global",
                    cluster_id=cluster_id,
                    mean_error=round(mean_error, 4),
                    global_mean_error=round(global_mean_error, 4),
                )
                continue

            # Gate 3: error_lift >= 1.10
            error_lift = mean_error / max(global_mean_error, 1e-9)
            if error_lift < 1.10:
                log.debug(
                    "ive.clustering.reject_low_lift",
                    cluster_id=cluster_id,
                    error_lift=round(error_lift, 4),
                )
                continue

            # Gate 4: minimum absolute difference
            abs_diff = mean_error - global_mean_error
            min_abs_diff = max(0.01, 0.05 * global_mean_error)
            if abs_diff < min_abs_diff:
                log.debug(
                    "ive.clustering.reject_small_diff",
                    cluster_id=cluster_id,
                    abs_diff=round(abs_diff, 4),
                    min_abs_diff=round(min_abs_diff, 4),
                )
                continue

            patterns.append(
                {
                    "pattern_type": "cluster",
                    "cluster_id": int(cluster_id),
                    "sample_count": n_in_cluster,
                    "mean_error": mean_error,
                    "std_error": std_error,
                    "cluster_center": cluster_center,
                    "error_lift": round(error_lift, 4),
                    "global_mean_error": round(global_mean_error, 4),
                }
            )

        # Sort by mean_error descending, then sample_count descending
        patterns.sort(key=lambda p: (-p["mean_error"], -p["sample_count"]))

        log.info(
            "ive.clustering.complete",
            n_clusters=len(patterns),
            noise_samples=int((labels == -1).sum()),
            global_mean_error=round(global_mean_error, 4),
        )
        return patterns


# ---------------------------------------------------------------------------
# Legacy class kept for backward-compatibility with __init__.py exports.
# ---------------------------------------------------------------------------


class HDBSCANClusterer:
    """Legacy clusterer for the residual-feature space using HDBSCAN.

    Retained for backward compatibility with existing callers and
    ``__init__.py`` exports.  New callers should use
    :class:`HDBSCANClustering` instead.
    """

    def __init__(
        self,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        cluster_selection_method: str = "eom",
        metric: str = "euclidean",
    ) -> None:
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
        """Fit HDBSCAN on the combined residual-feature space.

        Args:
            residuals:       Shape ``(n_samples,)`` — OOF residuals.
            feature_matrix:  Shape ``(n_samples, n_features)`` — preprocessed.
            residual_weight: Multiplier for the residual dimension to boost
                             its influence on cluster formation.

        Returns:
            :class:`ClusteringResult` with labels and per-cluster statistics.
        """
        import hdbscan as _hdbscan

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

        # Handle remaining NaN/inf after scaling
        np.nan_to_num(combined, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        self._clusterer = _hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_method=self.cluster_selection_method,
            metric=self.metric,
        )
        self._clusterer.fit(combined)
        labels: np.ndarray = self._clusterer.labels_

        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        n_clusters = len(unique)
        noise_fraction = float(np.mean(labels == -1))

        cluster_stats: dict[int, dict[str, float]] = {}
        for cid, count in zip(unique, counts, strict=False):
            mask = labels == cid
            cluster_stats[int(cid)] = {
                "size": int(count),
                "mean_residual": float(np.mean(residuals[mask])),
                "std_residual": float(np.std(residuals[mask], ddof=0)),
                "coverage_pct": float(count / len(residuals) * 100),
            }

        # Attempt DBCV validity score
        validity = 0.0
        if n_clusters >= 2:
            try:
                from hdbscan.validity import validity_index

                validity = float(validity_index(combined, labels))
            except Exception:
                log.debug("ive.clustering.validity_index_failed")

        log.info(
            "ive.clustering.done",
            n_clusters=n_clusters,
            noise_pct=round(noise_fraction * 100, 1),
            validity=round(validity, 4),
        )

        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            noise_fraction=noise_fraction,
            cluster_stats=cluster_stats,
            validity_score=validity,
        )
