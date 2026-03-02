"""Unit tests for SHAP interaction analysis module."""

from __future__ import annotations

import pytest

from ive.detection.shap_interactions import SHAPInteractionAnalyzer, SHAPResult
from ive.models.linear_model import LinearIVEModel


@pytest.fixture
def fitted_linear_model(small_X_y):
    X, y = small_X_y
    model = LinearIVEModel()
    model.fit(X, y)
    return model


class TestSHAPInteractionAnalyzer:
    def test_compute_returns_shap_result(self, fitted_linear_model, small_X_y) -> None:
        """compute() should return a SHAPResult."""
        X, _ = small_X_y
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(fitted_linear_model, X, ["x1", "x2"], compute_interactions=False)
        assert isinstance(result, SHAPResult)

    def test_shap_values_shape_matches_input(self, fitted_linear_model, small_X_y) -> None:
        """shap_values should have shape (n_samples, n_features)."""
        X, _ = small_X_y
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(fitted_linear_model, X, ["x1", "x2"])
        assert result.shap_values.shape[1] == 2

    def test_mean_abs_shap_keys_match_feature_names(self, fitted_linear_model, small_X_y) -> None:
        """mean_abs_shap keys should match provided feature names."""
        X, _ = small_X_y
        feature_names = ["x1", "x2"]
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(fitted_linear_model, X, feature_names)
        assert set(result.mean_abs_shap.keys()) == set(feature_names)

    def test_sample_size_respected(self, fitted_linear_model, small_X_y) -> None:
        """SHAP analysis should subsample to at most sample_size rows."""
        X, _ = small_X_y
        sample_size = 10
        analyzer = SHAPInteractionAnalyzer(sample_size=sample_size)
        result = analyzer.compute(fitted_linear_model, X, ["x1", "x2"])
        assert result.shap_values.shape[0] <= sample_size
