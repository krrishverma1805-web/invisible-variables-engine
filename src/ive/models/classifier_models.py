"""Classification IVE models (binary only).

Per plan §B5 + §170, multiclass classification routes to ``XGBClassifier``
for prediction but **skips residual-based detection** (the multinomial
deviance is harder to interpret as a per-row residual signal). This
module ships only the binary path, which is the common case.

**Residuals for classification.** ``predict()`` returns positive-class
probabilities. The residual surfaced to detection is the **signed
deviance residual**:

    sign(y - p) * sqrt(-2 * (y * log(p_safe) + (1-y) * log(1-p_safe)))

with ``p_safe = clip(p, eps, 1 - eps)`` (default eps = 1e-7) to avoid
``log(0)`` NaNs when the model predicts confidently. The ``CLASSIFICATION_PROBA_EPS``
constant is exposed in ``ive.utils.statistics`` so any other code
computing classification residuals stays consistent.

Plan reference: §B5, §170 (provisional effect-size threshold).
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import structlog

from ive.models.base_model import IVEModel

log = structlog.get_logger(__name__)


CLASSIFICATION_PROBA_EPS = 1e-7


def signed_deviance_residual(
    y_true: np.ndarray[Any, Any],
    y_proba: np.ndarray[Any, Any],
    *,
    eps: float = CLASSIFICATION_PROBA_EPS,
) -> np.ndarray[Any, Any]:
    """Compute signed deviance residuals for binary classification.

    Args:
        y_true: 0/1 integer target.
        y_proba: Positive-class probability ``p ∈ (0, 1)``.
        eps: Numerical-safety clip applied to ``p`` before the log.

    Returns:
        Array of the same shape as ``y_true``. Sign is positive when the
        true class is 1 and the model is unconfident; negative when the
        true class is 0 and the model is unconfident.
    """
    # Coerce inputs robustly — accept lists, pandas Series, etc.
    y_arr = np.asarray(y_true, dtype=float)
    p_arr = np.asarray(y_proba, dtype=float)
    if y_arr.shape != p_arr.shape:
        raise ValueError(
            f"signed_deviance_residual: shape mismatch y_true={y_arr.shape} vs y_proba={p_arr.shape}"
        )
    p = np.clip(p_arr, eps, 1.0 - eps)
    deviance = -2.0 * (y_arr * np.log(p) + (1.0 - y_arr) * np.log(1.0 - p))
    sign = np.where(y_arr >= p, 1.0, -1.0)
    return cast(np.ndarray[Any, Any], sign * np.sqrt(np.maximum(deviance, 0.0)))


# ── XGBoost binary classifier ───────────────────────────────────────────────


class XGBoostClassifierIVEModel(IVEModel):
    """``xgboost.XGBClassifier`` wrapped in the IVE model interface.

    ``predict()`` returns the **positive-class probability** (not the hard
    label) so residuals are well-defined. Use
    :func:`signed_deviance_residual` to convert ``(y_true, predict(X))``
    into the residual array detection consumes.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model: Any = None
        self._feature_names: list[str] = []
        self._fitted = False

    @property
    def model_name(self) -> str:
        return "xgboost_classifier"

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        from xgboost import XGBClassifier

        log.debug(
            "ive.xgboost_classifier.fit",
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )
        self._model.fit(X, y)
        self._fitted = True

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the positive-class probability so residuals are well-defined."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predict().")
        proba = self._model.predict_proba(X)
        return cast(np.ndarray[Any, Any], proba[:, 1])

    def get_feature_importance(self) -> dict[str, float]:
        if not self._fitted or self._model is None:
            return {}
        importances = self._model.feature_importances_
        names = self._feature_names or [f"f{i}" for i in range(len(importances))]
        return dict(zip(names, importances.tolist(), strict=False))

    def get_shap_values(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        import shap

        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before get_shap_values().")
        explainer = shap.TreeExplainer(self._model)
        return cast(np.ndarray[Any, Any], explainer.shap_values(X))

    def get_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "model_type": "xgboost_classifier",
        }


# ── Logistic regression ─────────────────────────────────────────────────────


class LogisticIVEModel(IVEModel):
    """``sklearn.linear_model.LogisticRegression`` wrapped in IVEModel.

    Linear-model counterpart to ``XGBoostClassifierIVEModel``. Fast,
    interpretable; serves as a baseline whose residuals indicate that a
    non-linear classification structure exists.
    """

    def __init__(
        self,
        C: float = 1.0,  # noqa: N803 — sklearn convention
        random_state: int = 42,
        max_iter: int = 1000,
    ) -> None:
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self._model: Any = None
        self._feature_names: list[str] = []
        self._fitted = False
        self._training_mean: np.ndarray[Any, Any] | None = None

    @property
    def model_name(self) -> str:
        return "logistic"

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        from sklearn.linear_model import LogisticRegression

        log.debug("ive.logistic.fit", n_samples=X.shape[0], n_features=X.shape[1])
        self._model = LogisticRegression(
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self._model.fit(X, y)
        self._training_mean = np.mean(X, axis=0)
        self._fitted = True

    def predict(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the positive-class probability."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predict().")
        return cast(np.ndarray[Any, Any], self._model.predict_proba(X)[:, 1])

    def get_feature_importance(self) -> dict[str, float]:
        if not self._fitted or self._model is None:
            return {}
        coefficients = np.abs(self._model.coef_).ravel()
        total = coefficients.sum()
        if total == 0:
            return {}
        normalised = coefficients / total
        names = self._feature_names or [f"f{i}" for i in range(len(normalised))]
        return dict(zip(names, normalised.tolist(), strict=False))

    def get_shap_values(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model must be fitted before get_shap_values().")
        if self._training_mean is None:
            raise RuntimeError("Training mean not available — model must be fitted.")
        # Linear SHAP on logits: coef * (x - mean). Returns a per-feature
        # contribution to the log-odds, consistent with downstream detection.
        coef = self._model.coef_.ravel()
        return cast(np.ndarray[Any, Any], (X - self._training_mean) * coef)

    def get_params(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "model_type": "logistic",
        }


# ── Problem-type detector ───────────────────────────────────────────────────


def detect_problem_type(
    y: np.ndarray[Any, Any],
    *,
    user_override: str | None = None,
) -> str:
    """Resolve the problem type for an experiment target.

    Per plan §B5 (corrected, conservative heuristic):

    - If ``user_override`` is provided, it always wins.
    - Auto-classify only when **all** of:
      - dtype is bool, integer (signed/unsigned), or pandas-extension boolean,
      - ``nunique <= 10``,
      - values are non-negative,
      - ``n_rows / nunique >= 30`` (so 1–5 star ratings on 80 rows stay regression).
    - Binary if ``nunique == 2`` AND values are in ``{0, 1}``.
    - Multiclass if ``nunique > 2`` AND values are sequential non-negative integers.
    - All other cases fall back to ``"regression"``.

    For ambiguous low-cardinality integer targets, a future caller should
    log ``problem_type_inferred_ambiguous`` so users see the inference and
    can override via experiment config.

    Returns:
        One of ``"regression" | "binary" | "multiclass"``.
    """
    if user_override:
        if user_override not in ("regression", "binary", "multiclass"):
            raise ValueError(
                f"user_override must be one of regression/binary/multiclass; got {user_override!r}"
            )
        return user_override

    # Defensive shape coercion — IVE downstream code assumes 1D targets.
    # A (n, 1) array would otherwise route through np.unique correctly but
    # silently break per-row residual computation downstream.
    if y.ndim > 1:
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        else:
            return "regression"

    if y.size == 0:
        return "regression"

    kind = y.dtype.kind  # 'b' bool, 'i' int, 'u' uint, 'f' float, 'O' object…
    if kind not in {"b", "i", "u"}:
        return "regression"

    unique = np.unique(y)
    nunique = int(unique.shape[0])
    if nunique < 2:
        return "regression"
    if nunique > 10:
        return "regression"
    if (unique < 0).any():
        return "regression"
    if y.shape[0] / nunique < 30:
        return "regression"

    if nunique == 2 and {int(v) for v in unique} <= {0, 1}:
        return "binary"

    expected = np.arange(nunique)
    if np.array_equal(unique.astype(int), expected):
        return "multiclass"

    # Low-cardinality integers that aren't 0/1 binary or sequential —
    # safer to treat as regression than misroute.
    return "regression"


__all__ = [
    "CLASSIFICATION_PROBA_EPS",
    "LogisticIVEModel",
    "XGBoostClassifierIVEModel",
    "detect_problem_type",
    "signed_deviance_residual",
]
