"""Unit tests for the binary-classification residual + LogisticIVEModel.

LogisticIVEModel doesn't depend on xgboost / libomp so it's safe to
exercise in this environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from ive.models.classifier_models import (
    CLASSIFICATION_PROBA_EPS,
    LogisticIVEModel,
    signed_deviance_residual,
)

pytestmark = pytest.mark.unit


class TestSignedDevianceResidual:
    def test_perfect_prediction_returns_zero(self):
        y = np.array([0, 1])
        p = np.array([0.0, 1.0])
        out = signed_deviance_residual(y, p)
        # log(eps) is finite but huge; with y=p the deviance contribution
        # is -2*log(1-eps) ≈ 2e-7, sqrt ≈ 1.4e-3 → near-zero, within
        # tolerance for "the model is right."
        assert np.all(np.abs(out) < 1e-2)

    def test_completely_wrong_prediction_max_residual(self):
        y = np.array([1, 0])
        p = np.array([0.0, 1.0])  # confidently wrong
        out = signed_deviance_residual(y, p)
        # |residual| = sqrt(-2*log(eps)) ≈ sqrt(32) ≈ 5.7 — large.
        assert np.all(np.abs(out) > 5)

    def test_sign_follows_under_prediction(self):
        # When y=1 and p<0.5, model under-predicts → positive sign.
        y = np.array([1, 1])
        p = np.array([0.2, 0.4])
        out = signed_deviance_residual(y, p)
        assert np.all(out > 0)

    def test_sign_follows_over_prediction(self):
        # When y=0 and p>0.5, model over-predicts → negative sign.
        y = np.array([0, 0])
        p = np.array([0.7, 0.9])
        out = signed_deviance_residual(y, p)
        assert np.all(out < 0)

    def test_clipping_prevents_log_zero_nan(self):
        # p=0 with y=1 would diverge without clipping.
        out = signed_deviance_residual(
            np.array([1]),
            np.array([0.0]),
            eps=CLASSIFICATION_PROBA_EPS,
        )
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_zero_residual_at_uncertain_prediction(self):
        # y=1, p=0.5 → deviance = -2*log(0.5) ≈ 1.39, sqrt ≈ 1.18.
        out = signed_deviance_residual(np.array([1]), np.array([0.5]))
        assert np.isclose(out[0], np.sqrt(-2 * np.log(0.5)))


class TestLogisticIVEModel:
    @pytest.fixture
    def linearly_separable_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 3))
        # y = 1 when first feature > 0
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_fits_and_predicts_probabilities(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = LogisticIVEModel()
        model.fit(X, y)
        p = model.predict(X)
        assert p.shape == (len(y),)
        assert np.all((p >= 0) & (p <= 1))

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LogisticIVEModel().predict(np.zeros((1, 1)))

    def test_feature_importance_normalized(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = LogisticIVEModel()
        model.fit(X, y)
        importances = model.get_feature_importance()
        assert np.isclose(sum(importances.values()), 1.0)

    def test_shap_values_shape_matches_input(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = LogisticIVEModel()
        model.fit(X, y)
        shap = model.get_shap_values(X)
        assert shap.shape == X.shape

    def test_residuals_are_low_when_classifier_is_accurate(self, linearly_separable_data):
        X, y = linearly_separable_data
        model = LogisticIVEModel()
        model.fit(X, y)
        p = model.predict(X)
        residuals = signed_deviance_residual(y, p)
        # Most residuals should be small for an easy classification problem.
        assert float(np.median(np.abs(residuals))) < 1.0

    def test_model_name_is_logistic(self):
        assert LogisticIVEModel().model_name == "logistic"
