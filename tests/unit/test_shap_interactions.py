"""Unit tests for SHAPInteractionAnalyzer.

Aligned with current public API:
  - SHAPInteractionAnalyzer(sample_size=50)
  - compute(model, X: np.ndarray, feature_names, compute_interactions=True) -> SHAPResult
  - SHAPResult attributes: shap_values, mean_abs_shap, shap_interaction_values,
                           feature_names, top_interaction_pairs

Note: small_X_y returns (np.ndarray, np.ndarray) — shape (200, 4).
      We use a 2-feature slice so SHAP shapes are predictable.
"""

from __future__ import annotations

import pytest

from ive.detection.shap_interactions import SHAPInteractionAnalyzer, SHAPResult
from ive.models.linear_model import LinearIVEModel


@pytest.fixture
def fitted_linear_model(small_X_y):
    """Fit a LinearIVEModel on the first 2 features of small_X_y."""
    X, y = small_X_y
    X2 = X[:, :2]
    model = LinearIVEModel()
    model.fit(X2, y)
    return model, X2


class TestSHAPInteractionAnalyzer:
    def test_compute_returns_shap_result(self, fitted_linear_model) -> None:
        """compute() must return a SHAPResult instance."""
        model, X2 = fitted_linear_model
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(model, X2, ["x1", "x2"], compute_interactions=False)
        assert isinstance(result, SHAPResult)

    def test_shap_values_shape_matches_input(self, fitted_linear_model) -> None:
        """shap_values must have shape (n_samples, n_features) where n_features=2."""
        model, X2 = fitted_linear_model
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(model, X2, ["x1", "x2"], compute_interactions=False)
        assert result.shap_values.ndim == 2
        assert result.shap_values.shape[1] == 2

    def test_mean_abs_shap_keys_match_feature_names(self, fitted_linear_model) -> None:
        """mean_abs_shap keys must exactly match the supplied feature names."""
        model, X2 = fitted_linear_model
        feature_names = ["x1", "x2"]
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(model, X2, feature_names, compute_interactions=False)
        assert set(result.mean_abs_shap.keys()) == set(feature_names)

    def test_sample_size_respected(self, fitted_linear_model) -> None:
        """SHAP analysis should subsample to at most sample_size rows."""
        model, X2 = fitted_linear_model
        sample_size = 10
        analyzer = SHAPInteractionAnalyzer(sample_size=sample_size)
        result = analyzer.compute(model, X2, ["x1", "x2"], compute_interactions=False)
        assert result.shap_values.shape[0] <= sample_size

    def test_mean_abs_shap_values_are_non_negative(self, fitted_linear_model) -> None:
        """All mean_abs_shap values must be >= 0 (they are absolute values)."""
        model, X2 = fitted_linear_model
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(model, X2, ["x1", "x2"], compute_interactions=False)
        for feat, val in result.mean_abs_shap.items():
            assert val >= 0.0, f"mean_abs_shap['{feat}'] = {val} is negative"

    def test_feature_names_stored_on_result(self, fitted_linear_model) -> None:
        """SHAPResult must expose feature_names matching what was passed in."""
        model, X2 = fitted_linear_model
        feature_names = ["alpha", "beta"]
        analyzer = SHAPInteractionAnalyzer(sample_size=50)
        result = analyzer.compute(model, X2, feature_names, compute_interactions=False)
        assert list(result.feature_names) == feature_names
