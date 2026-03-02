"""Unit tests for HDBSCAN clustering module."""

from __future__ import annotations

from ive.detection.clustering import ClusteringResult, HDBSCANClusterer


class TestHDBSCANClusterer:
    def test_fit_returns_clustering_result(self, small_X_y, residuals_array) -> None:
        """fit() should return a ClusteringResult."""
        X, _ = small_X_y
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        result = clusterer.fit(residuals_array, X)
        assert isinstance(result, ClusteringResult)

    def test_labels_correct_length(self, small_X_y, residuals_array) -> None:
        """Cluster labels should have one entry per sample."""
        X, _ = small_X_y
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        result = clusterer.fit(residuals_array, X)
        assert len(result.labels) == len(residuals_array)

    def test_noise_fraction_between_0_and_1(self, small_X_y, residuals_array) -> None:
        """noise_fraction should be in [0, 1]."""
        X, _ = small_X_y
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        result = clusterer.fit(residuals_array, X)
        assert 0.0 <= result.noise_fraction <= 1.0

    def test_cluster_stats_keys_match_labels(self, small_X_y, residuals_array) -> None:
        """cluster_stats keys should match unique non-noise labels."""
        X, _ = small_X_y
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        result = clusterer.fit(residuals_array, X)
        unique_non_noise = set(result.labels[result.labels >= 0].tolist())
        assert set(result.cluster_stats.keys()) == unique_non_noise or len(unique_non_noise) == 0
