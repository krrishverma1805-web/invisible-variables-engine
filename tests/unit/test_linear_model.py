"""Unit tests for LinearIVEModel."""

from __future__ import annotations

import numpy as np
import pytest

from ive.models.linear_model import LinearIVEModel


class TestLinearIVEModel:
    def test_model_name(self) -> None:
        assert LinearIVEModel().model_name == "linear"

    def test_fit_and_predict(self, small_X_y) -> None:
        """fit then predict should return array of correct shape."""
        X, y = small_X_y
        model = LinearIVEModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_before_fit_raises(self, small_X_y) -> None:
        """predict() before fit() should raise RuntimeError."""
        X, _ = small_X_y
        model = LinearIVEModel()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_get_feature_importance_returns_dict(self, small_X_y) -> None:
        """get_feature_importance() should return a dict summing to ~1.0."""
        X, y = small_X_y
        model = LinearIVEModel()
        model.fit(X, y)
        fi = model.get_feature_importance()
        assert isinstance(fi, dict)
        assert abs(sum(fi.values()) - 1.0) < 1e-6

    def test_get_shap_values_shape(self, small_X_y) -> None:
        """get_shap_values() should return array matching input shape."""
        X, y = small_X_y
        model = LinearIVEModel()
        model.fit(X, y)
        shap = model.get_shap_values(X)
        assert shap.shape == X.shape

    def test_regularisation_alpha(self) -> None:
        """Alpha parameter should be stored on the model."""
        model = LinearIVEModel(alpha=10.0)
        assert model.alpha == 10.0

    def test_is_fitted_flag(self, small_X_y) -> None:
        X, y = small_X_y
        model = LinearIVEModel()
        assert not model.is_fitted
        model.fit(X, y)
        assert model.is_fitted
