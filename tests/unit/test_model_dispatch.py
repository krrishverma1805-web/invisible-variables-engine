"""Unit tests for ive.models.dispatch.resolve_model_class."""

from __future__ import annotations

import pytest

from ive.models.classifier_models import LogisticIVEModel, XGBoostClassifierIVEModel
from ive.models.dispatch import resolve_model_class
from ive.models.linear_model import LinearIVEModel
from ive.models.xgboost_model import XGBoostIVEModel

pytestmark = pytest.mark.unit


class TestResolveModelClass:
    def test_regression_linear(self):
        assert resolve_model_class("linear", "regression") is LinearIVEModel

    def test_regression_xgboost(self):
        assert resolve_model_class("xgboost", "regression") is XGBoostIVEModel

    def test_binary_linear_routes_to_logistic(self):
        assert resolve_model_class("linear", "binary") is LogisticIVEModel

    def test_binary_xgboost_routes_to_classifier(self):
        assert resolve_model_class("xgboost", "binary") is XGBoostClassifierIVEModel

    def test_multiclass_xgboost_routes_to_classifier(self):
        assert resolve_model_class("xgboost", "multiclass") is XGBoostClassifierIVEModel

    def test_multiclass_linear_raises(self):
        with pytest.raises(ValueError, match="Multiclass classification with the linear"):
            resolve_model_class("linear", "multiclass")

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            resolve_model_class("knn", "regression")

    def test_unknown_problem_type_raises(self):
        with pytest.raises(ValueError, match="Unknown problem_type"):
            resolve_model_class("linear", "ranking")
