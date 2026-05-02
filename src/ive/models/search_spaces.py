"""Per-model hyperparameter search spaces (Phase B1).

Plan reference: §B1 + §98. Each space is a flat dict of ``ParamSpec``
entries; the HPO module evaluates each spec via Optuna's trial.suggest_*
API. Pinning ``n_estimators`` at a large value is intentional — XGBoost's
native early stopping handles the tree-count axis far more efficiently
than tuning it as a hyperparameter would (per plan §98).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FloatRange:
    """Continuous float parameter, optionally on a log scale."""

    low: float
    high: float
    log: bool = False


@dataclass(frozen=True)
class IntRange:
    """Discrete integer parameter."""

    low: int
    high: int


@dataclass(frozen=True)
class CategoricalChoice:
    """Pick one of a fixed set of values."""

    choices: tuple[Any, ...]


ParamSpec = FloatRange | IntRange | CategoricalChoice


# Linear (Ridge regression) — single tuning axis on a log scale.
LINEAR_SEARCH_SPACE: dict[str, ParamSpec] = {
    "alpha": FloatRange(low=1e-3, high=100.0, log=True),
}

# Logistic regression — same single-axis shape as linear.
LOGISTIC_SEARCH_SPACE: dict[str, ParamSpec] = {
    "C": FloatRange(low=1e-3, high=100.0, log=True),
}

# XGBoost regressor — n_estimators is fixed at a large value (plan §98);
# early stopping handles the tree-count search inside each trial.
XGBOOST_REGRESSOR_SEARCH_SPACE: dict[str, ParamSpec] = {
    "max_depth": IntRange(low=3, high=10),
    "learning_rate": FloatRange(low=0.01, high=0.3, log=True),
    "subsample": FloatRange(low=0.6, high=1.0),
    "colsample_bytree": FloatRange(low=0.6, high=1.0),
    "reg_alpha": FloatRange(low=0.0, high=10.0),
    "reg_lambda": FloatRange(low=0.0, high=10.0),
    "min_child_weight": IntRange(low=1, high=20),
    "gamma": FloatRange(low=0.0, high=5.0),
}

# XGBoost classifier — same shape, different objective.
XGBOOST_CLASSIFIER_SEARCH_SPACE: dict[str, ParamSpec] = dict(XGBOOST_REGRESSOR_SEARCH_SPACE)


# Pinned hyperparameters that are NOT tuned but are passed to the
# constructor unchanged (plan §98).
PINNED_HYPERPARAMS: dict[str, dict[str, Any]] = {
    "xgboost_regressor": {"n_estimators": 2000},
    "xgboost_classifier": {"n_estimators": 2000},
}


def get_search_space(model_type: str, problem_type: str) -> dict[str, ParamSpec]:
    """Return the search space for a ``(model_type, problem_type)`` pair.

    Raises:
        ValueError: when the combination is unsupported (mirrors
            ``resolve_model_class`` so the pipeline gets a uniform error
            surface).
    """
    if model_type == "linear":
        if problem_type == "regression":
            return LINEAR_SEARCH_SPACE
        if problem_type == "binary":
            return LOGISTIC_SEARCH_SPACE
        raise ValueError(
            f"Linear models do not support problem_type={problem_type!r}."
        )
    if model_type == "xgboost":
        if problem_type == "regression":
            return XGBOOST_REGRESSOR_SEARCH_SPACE
        return XGBOOST_CLASSIFIER_SEARCH_SPACE
    raise ValueError(f"Unknown model_type {model_type!r}.")


def get_pinned_hyperparams(model_type: str, problem_type: str) -> dict[str, Any]:
    """Return hyperparameters that should always be passed to the constructor."""
    if model_type == "xgboost":
        if problem_type == "regression":
            return dict(PINNED_HYPERPARAMS["xgboost_regressor"])
        return dict(PINNED_HYPERPARAMS["xgboost_classifier"])
    return {}


__all__ = [
    "LINEAR_SEARCH_SPACE",
    "LOGISTIC_SEARCH_SPACE",
    "PINNED_HYPERPARAMS",
    "XGBOOST_CLASSIFIER_SEARCH_SPACE",
    "XGBOOST_REGRESSOR_SEARCH_SPACE",
    "CategoricalChoice",
    "FloatRange",
    "IntRange",
    "ParamSpec",
    "get_pinned_hyperparams",
    "get_search_space",
]
