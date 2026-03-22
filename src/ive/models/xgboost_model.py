"""
XGBoost IVE Model.

Implements the IVEModel interface using XGBoost's gradient boosting.
XGBoost is the primary model for residual extraction because:
    1. It can capture non-linear patterns that linear models miss
    2. TreeExplainer provides exact SHAP values and interaction values
    3. It naturally handles missing values and mixed feature types
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from ive.models.base_model import IVEModel

log = structlog.get_logger(__name__)


class XGBoostIVEModel(IVEModel):
    """
    XGBoost regressor/classifier wrapped in the IVEModel interface.

    Defaults are tuned for medium-sized datasets and reasonable training time.
    All hyperparameters can be overridden.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self._model: Any = None
        self._explainer: Any = None
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def model_name(self) -> str:
        return "xgboost"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit XGBoost on the training data.

        TODO:
            - from xgboost import XGBRegressor
            - self._model = XGBRegressor(**params, tree_method='hist', enable_categorical=True)
            - self._model.fit(X, y, verbose=False)
            - self._fitted = True
        """
        from xgboost import XGBRegressor

        log.debug("ive.xgboost_model.fit", n_samples=X.shape[0], n_features=X.shape[1])
        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            verbosity=0,
        )
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return XGBoost predictions."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predict().")
        return self._model.predict(X)  # type: ignore[no-any-return]

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importances (weight/gain) sorted descending."""
        # Safely get the underlying model object (handles both self.model and self._model)
        model_obj = getattr(self, "model", getattr(self, "_model", None))

        if model_obj is None or not hasattr(model_obj, "feature_importances_"):
            return {}

        raw_importances = model_obj.feature_importances_

        # Safely get feature names
        names = getattr(self, "feature_names_in_", getattr(self, "_feature_names_in", None))
        if names is None:
            names = [f"feature_{i}" for i in range(len(raw_importances))]

        imp_dict = {
            str(name): float(imp) for name, imp in zip(names, raw_importances, strict=False)
        }
        return dict(sorted(imp_dict.items(), key=lambda item: item[1], reverse=True))

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute exact SHAP values using TreeExplainer.

        TODO:
            - import shap
            - self._explainer = shap.TreeExplainer(self._model)
            - return self._explainer.shap_values(X)
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before get_shap_values().")
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer.shap_values(X)  # type: ignore[no-any-return]

    def get_shap_interaction_values(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SHAP interaction values (n_samples, n_features, n_features).

        These reveal pairwise feature interactions not captured by main SHAP effects.

        TODO:
            - Initialise self._explainer if None
            - return self._explainer.shap_interaction_values(X)
        """
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer.shap_interaction_values(X)  # type: ignore[no-any-return]

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
