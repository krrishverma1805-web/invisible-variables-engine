"""Model dispatch — picks the right ``IVEModel`` class for a problem type.

Per plan §B5 + §170: the pipeline calls ``resolve_model_class(model_type, problem_type)``
to get the constructor, instantiates it with deployment defaults, and runs
the existing CV loop. This keeps the pipeline ignorant of which specific
classifier the user wanted.
"""

from __future__ import annotations

from typing import Literal

from ive.models.base_model import IVEModel
from ive.models.classifier_models import LogisticIVEModel, XGBoostClassifierIVEModel
from ive.models.linear_model import LinearIVEModel
from ive.models.xgboost_model import XGBoostIVEModel

ProblemType = Literal["regression", "binary", "multiclass"]


def resolve_model_class(model_type: str, problem_type: str) -> type[IVEModel]:
    """Return the IVEModel subclass for a ``(model_type, problem_type)`` pair.

    Args:
        model_type: One of ``"linear" | "xgboost"``.
        problem_type: One of ``"regression" | "binary" | "multiclass"``.

    Raises:
        ValueError: when the combination isn't supported (currently only
            multiclass + linear, since LogisticRegression's binary-only
            shape doesn't fit the IVE residual model — multiclass routes
            to XGBoost only).
    """
    if model_type not in ("linear", "xgboost"):
        raise ValueError(
            f"Unknown model_type {model_type!r}; expected one of linear/xgboost."
        )
    if problem_type not in ("regression", "binary", "multiclass"):
        raise ValueError(
            f"Unknown problem_type {problem_type!r}; expected one of regression/binary/multiclass."
        )
    if problem_type == "regression":
        if model_type == "linear":
            return LinearIVEModel
        return XGBoostIVEModel
    if problem_type == "binary":
        if model_type == "linear":
            return LogisticIVEModel
        return XGBoostClassifierIVEModel
    # multiclass: only XGBoost classifier (per plan §B5 — multiclass skips
    # residual-based detection but still trains predictively).
    if model_type == "linear":
        raise ValueError(
            "Multiclass classification with the linear model is not supported. "
            "Configure model_types=['xgboost'] for multiclass experiments."
        )
    return XGBoostClassifierIVEModel


__all__ = ["ProblemType", "resolve_model_class"]
