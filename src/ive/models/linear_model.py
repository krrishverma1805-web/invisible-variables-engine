"""
Linear IVE Model (Ridge Regression).

Implements the IVEModel interface using scikit-learn's Ridge regressor.
Linear models serve as fast baselines whose residuals are straightforward
to interpret — high residuals from a linear model strongly indicate that
a non-linear latent structure exists in the data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from ive.models.base_model import IVEModel

log = structlog.get_logger(__name__)


class LinearIVEModel(IVEModel):
    """
    Ridge regression model wrapped in the IVEModel interface.

    Uses L2 regularisation to handle multicollinear feature spaces.
    SHAP values for linear models are exact (coefficients × feature values).
    """

    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        """
        Args:
            alpha: Regularisation strength (L2). Higher = more regularisation.
            random_state: Random seed for reproducibility.
        """
        self.alpha = alpha
        self.random_state = random_state
        self._model: Any = None
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def model_name(self) -> str:
        return "linear"

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        """
        Fit Ridge regression on the training data.

        TODO:
            - from sklearn.linear_model import Ridge
            - self._model = Ridge(alpha=self.alpha)
            - self._model.fit(X, y)
            - self._fitted = True
        """
        from sklearn.linear_model import Ridge

        log.debug("ive.linear_model.fit", n_samples=X.shape[0], n_features=X.shape[1])
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return Ridge predictions."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predict().")
        return self._model.predict(X)  # type: ignore[no-any-return]

    def get_feature_importance(self) -> dict[str, float]:
        """
        Return |coefficient| as a proxy for feature importance.

        TODO:
            - Normalise absolute coefficients to [0, 1]
            - Map to self._feature_names
        """
        if not self._fitted or self._model is None:
            return {}
        coefficients = np.abs(self._model.coef_)
        total = coefficients.sum()
        if total == 0:
            return {}
        normalised = coefficients / total
        names = self._feature_names or [f"f{i}" for i in range(len(normalised))]
        return dict(zip(names, normalised.tolist(), strict=False))

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute exact SHAP values for a linear model.

        For linear models: shap_i = coef_i * (x_i - mean_i)

        TODO:
            - Use shap.LinearExplainer for exact values
            - Or compute directly: shap_values = (X - X.mean(axis=0)) * self._model.coef_
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before get_shap_values().")
        # Approximate SHAP via coefficient × feature deviation
        # TODO: Replace with shap.LinearExplainer for accuracy
        feature_means = np.zeros(X.shape[1])
        return (X - feature_means) * self._model.coef_

    def get_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "model_type": "ridge"}
