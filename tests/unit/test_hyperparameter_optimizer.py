"""Unit tests for ive.models.hyperparameter_optimizer.optimize.

These exercise the LinearIVEModel + Ridge path so we don't need xgboost
(libomp env issue on this box). The Optuna API is the same regardless.
"""

from __future__ import annotations

import numpy as np
import pytest

from ive.models.hyperparameter_optimizer import HPOResult, optimize
from ive.models.linear_model import LinearIVEModel
from ive.models.search_spaces import LINEAR_SEARCH_SPACE

pytestmark = pytest.mark.unit


def _factory(params):
    return LinearIVEModel(alpha=params["alpha"])


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 4))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.3
    return X, y


class TestOptimizeBasics:
    def test_returns_hpo_result(self, regression_data):
        X, y = regression_data
        result = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=5,
            timeout_seconds=60.0,
            cv_strategy="kfold",
            inner_cv_splits=3,
            seed=42,
        )
        assert isinstance(result, HPOResult)
        assert "alpha" in result.best_params
        assert result.n_trials == 5
        assert len(result.search_history) == 5
        assert result.elapsed_seconds > 0

    def test_best_params_within_search_space(self, regression_data):
        X, y = regression_data
        result = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=10,
            timeout_seconds=60.0,
            cv_strategy="kfold",
            inner_cv_splits=3,
            seed=42,
        )
        alpha = result.best_params["alpha"]
        spec = LINEAR_SEARCH_SPACE["alpha"]
        assert spec.low <= alpha <= spec.high


class TestDeterminism:
    def test_same_seed_same_best_params(self, regression_data):
        X, y = regression_data
        a = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=8,
            timeout_seconds=60.0,
            cv_strategy="kfold",
            seed=42,
        )
        b = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=8,
            timeout_seconds=60.0,
            cv_strategy="kfold",
            seed=42,
        )
        # Same seed → same trial trajectory → same best_params.
        assert a.best_params == b.best_params

    def test_different_seeds_explore_different_points(self, regression_data):
        X, y = regression_data
        a = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=8,
            cv_strategy="kfold",
            seed=42,
        )
        b = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=8,
            cv_strategy="kfold",
            seed=99,
        )
        # The trial histories should differ even if the best happens to coincide.
        assert a.search_history != b.search_history


class TestBudget:
    def test_n_trials_respected(self, regression_data):
        X, y = regression_data
        result = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=3,
            timeout_seconds=60.0,
            cv_strategy="kfold",
            seed=42,
        )
        assert result.n_trials == 3

    def test_timeout_caps_runtime(self, regression_data):
        X, y = regression_data
        # Very tight budget — should return whatever fits in 0.1s.
        result = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=10_000,
            timeout_seconds=0.1,
            cv_strategy="kfold",
            seed=42,
        )
        assert result.elapsed_seconds <= 5.0  # generous to absorb sklearn warmup


class TestTimeSeriesCompatibility:
    def test_timeseries_strategy_is_passed_through(self, regression_data):
        X, y = regression_data
        time_index = np.arange(len(y))
        # Should not raise — inner CV runs with TimeSeriesSplit.
        result = optimize(
            model_factory=_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=2,
            timeout_seconds=60.0,
            cv_strategy="timeseries",
            time_index=time_index,
            seed=42,
        )
        assert result.n_trials == 2
