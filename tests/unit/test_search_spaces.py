"""Unit tests for ive.models.search_spaces."""

from __future__ import annotations

import pytest

from ive.models.search_spaces import (
    LINEAR_SEARCH_SPACE,
    LOGISTIC_SEARCH_SPACE,
    XGBOOST_CLASSIFIER_SEARCH_SPACE,
    XGBOOST_REGRESSOR_SEARCH_SPACE,
    FloatRange,
    IntRange,
    get_pinned_hyperparams,
    get_search_space,
)

pytestmark = pytest.mark.unit


class TestSearchSpaceShape:
    def test_linear_has_only_alpha_log_scale(self):
        assert set(LINEAR_SEARCH_SPACE.keys()) == {"alpha"}
        spec = LINEAR_SEARCH_SPACE["alpha"]
        assert isinstance(spec, FloatRange)
        assert spec.log is True

    def test_logistic_has_only_C_log_scale(self):
        assert set(LOGISTIC_SEARCH_SPACE.keys()) == {"C"}
        assert LOGISTIC_SEARCH_SPACE["C"].log is True

    def test_xgboost_regressor_has_no_n_estimators(self):
        # Per plan §98, n_estimators is pinned, not tuned.
        assert "n_estimators" not in XGBOOST_REGRESSOR_SEARCH_SPACE

    def test_xgboost_regressor_has_expected_axes(self):
        expected = {
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "min_child_weight",
            "gamma",
        }
        assert set(XGBOOST_REGRESSOR_SEARCH_SPACE.keys()) == expected

    def test_xgboost_classifier_mirrors_regressor(self):
        assert (
            set(XGBOOST_CLASSIFIER_SEARCH_SPACE.keys())
            == set(XGBOOST_REGRESSOR_SEARCH_SPACE.keys())
        )

    def test_int_ranges_are_int_specs(self):
        assert isinstance(XGBOOST_REGRESSOR_SEARCH_SPACE["max_depth"], IntRange)
        assert isinstance(XGBOOST_REGRESSOR_SEARCH_SPACE["min_child_weight"], IntRange)


class TestGetSearchSpace:
    def test_linear_regression(self):
        assert get_search_space("linear", "regression") is LINEAR_SEARCH_SPACE

    def test_linear_binary_routes_to_logistic(self):
        assert get_search_space("linear", "binary") is LOGISTIC_SEARCH_SPACE

    def test_linear_multiclass_raises(self):
        with pytest.raises(ValueError):
            get_search_space("linear", "multiclass")

    def test_xgboost_regression(self):
        assert get_search_space("xgboost", "regression") is XGBOOST_REGRESSOR_SEARCH_SPACE

    def test_xgboost_binary(self):
        assert get_search_space("xgboost", "binary") is XGBOOST_CLASSIFIER_SEARCH_SPACE

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError):
            get_search_space("knn", "regression")


class TestPinnedHyperparams:
    def test_xgboost_regression_pins_n_estimators(self):
        pinned = get_pinned_hyperparams("xgboost", "regression")
        assert pinned["n_estimators"] == 2000

    def test_xgboost_classifier_pins_n_estimators(self):
        pinned = get_pinned_hyperparams("xgboost", "binary")
        assert pinned["n_estimators"] == 2000

    def test_linear_has_no_pinned(self):
        assert get_pinned_hyperparams("linear", "regression") == {}

    def test_returns_fresh_dict(self):
        a = get_pinned_hyperparams("xgboost", "regression")
        b = get_pinned_hyperparams("xgboost", "regression")
        a["n_estimators"] = 999
        assert b["n_estimators"] == 2000  # mutation isolated
