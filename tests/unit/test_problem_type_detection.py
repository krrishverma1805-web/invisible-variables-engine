"""Unit tests for ive.models.classifier_models.detect_problem_type.

The heuristic is deliberately conservative (per plan §B5): only auto-
classifies when *all* of (integer/bool dtype, nunique<=10, non-negative,
n_rows/nunique>=30) hold. Everything else stays regression so we don't
silently misroute count-valued targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from ive.models.classifier_models import detect_problem_type

pytestmark = pytest.mark.unit


class TestUserOverride:
    def test_user_override_always_wins(self):
        # Even an obviously-binary array stays regression with override.
        y = np.array([0, 1, 1, 0, 1, 0])
        assert detect_problem_type(y, user_override="regression") == "regression"

    def test_user_override_validates_value(self):
        with pytest.raises(ValueError):
            detect_problem_type(np.array([1.0]), user_override="bogus")

    @pytest.mark.parametrize("override", ["regression", "binary", "multiclass"])
    def test_user_override_accepts_valid_values(self, override):
        y = np.array([1.0, 2.0, 3.0])
        assert detect_problem_type(y, user_override=override) == override


class TestAutoBinary:
    def test_balanced_binary_zero_one_int(self):
        y = np.repeat([0, 1], 50)  # 100 rows, 2 classes
        assert detect_problem_type(y) == "binary"

    def test_imbalanced_binary_still_routes(self):
        # 95/5 split, 100 rows → ratio 50, well over the 30 threshold.
        y = np.array([0] * 95 + [1] * 5)
        assert detect_problem_type(y) == "binary"

    def test_binary_bool_dtype(self):
        y = np.array([True, False] * 50)
        assert detect_problem_type(y) == "binary"

    def test_few_rows_binary_falls_back(self):
        # 20 rows / 2 classes = 10 ratio, under threshold → regression
        y = np.array([0, 1] * 10)
        assert detect_problem_type(y) == "regression"


class TestAutoMulticlass:
    def test_three_class_sequential(self):
        y = np.repeat(np.arange(3), 50)  # 150 rows, classes 0,1,2
        assert detect_problem_type(y) == "multiclass"

    def test_ten_class_sequential(self):
        y = np.repeat(np.arange(10), 30)  # 300 rows, exactly the threshold
        assert detect_problem_type(y) == "multiclass"

    def test_eleven_classes_falls_back(self):
        # nunique > 10 → regression (count target territory).
        y = np.repeat(np.arange(11), 30)
        assert detect_problem_type(y) == "regression"

    def test_non_sequential_classes_falls_back(self):
        # Star-rating-style {1,2,3,4,5} aren't sequential from 0 → regression.
        y = np.repeat(np.array([1, 2, 3, 4, 5]), 30)
        assert detect_problem_type(y) == "regression"


class TestRegressionFallbacks:
    def test_float_dtype_always_regression(self):
        y = np.array([0.0, 1.0] * 50)
        # Even though only 2 unique values, float dtype → regression.
        assert detect_problem_type(y) == "regression"

    def test_negative_values_force_regression(self):
        y = np.array([-1, 0, 1] * 50)
        assert detect_problem_type(y) == "regression"

    def test_single_unique_value(self):
        y = np.array([1, 1, 1, 1])
        assert detect_problem_type(y) == "regression"

    def test_empty_array(self):
        assert detect_problem_type(np.array([], dtype=int)) == "regression"

    def test_object_dtype_falls_back(self):
        y = np.array(["a", "b", "a"], dtype=object)
        assert detect_problem_type(y) == "regression"
