"""Unit tests for XGBoostIVEModel."""

from __future__ import annotations

import pytest

pytest.importorskip("xgboost", reason="xgboost not installed")

from ive.models.xgboost_model import XGBoostIVEModel


class TestXGBoostIVEModel:
    def test_model_name(self) -> None:
        assert XGBoostIVEModel().model_name == "xgboost"

    def test_fit_and_predict(self, small_X_y) -> None:
        X, y = small_X_y
        model = XGBoostIVEModel(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)

    def test_predict_before_fit_raises(self, small_X_y) -> None:
        X, _ = small_X_y
        model = XGBoostIVEModel()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_feature_importance_non_empty(self, small_X_y) -> None:
        X, y = small_X_y
        model = XGBoostIVEModel(n_estimators=10)
        model.fit(X, y)
        fi = model.get_feature_importance()
        assert isinstance(fi, dict)
        assert len(fi) > 0

    def test_is_fitted_after_fit(self, small_X_y) -> None:
        X, y = small_X_y
        model = XGBoostIVEModel(n_estimators=5)
        assert not model.is_fitted
        model.fit(X, y)
        assert model.is_fitted

    def test_get_params_returns_dict(self) -> None:
        model = XGBoostIVEModel(n_estimators=50, max_depth=3)
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3
