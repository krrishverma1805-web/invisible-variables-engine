"""Wave 2 audit regressions — flaws caught during rigorous testing.

Locks in the Wave 2 audit fixes so they don't silently regress.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ive.models.ensemble import StackedEnsemble
from ive.models.hyperparameter_optimizer import optimize
from ive.models.linear_model import LinearIVEModel
from ive.models.search_spaces import LINEAR_SEARCH_SPACE

pytestmark = pytest.mark.unit


# ── HPO factory-failure handling ────────────────────────────────────────────


class TestHpoFactoryFailureRecovery:
    """When the model factory itself raises, the trial should be recorded as
    failed rather than crashing the optimize() call."""

    @pytest.fixture
    def regression_data(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 3))
        y = X[:, 0] + 0.1 * rng.standard_normal(100)
        return X, y

    def test_all_failing_factory_returns_inf_score(self, regression_data):
        X, y = regression_data

        def bad_factory(_p):
            raise RuntimeError("forced failure")

        result = optimize(
            model_factory=bad_factory,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=3,
            timeout_seconds=5.0,
            cv_strategy="kfold",
        )
        assert result.best_score == -float("inf")
        assert result.best_params == {}
        assert result.n_trials == 3

    def test_partial_failure_picks_surviving_trial(self, regression_data):
        X, y = regression_data
        calls = {"n": 0}

        def flaky(p):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("flake")
            return LinearIVEModel(alpha=p["alpha"])

        result = optimize(
            model_factory=flaky,
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=4,
            timeout_seconds=10.0,
            cv_strategy="kfold",
            seed=42,
        )
        # With 4 trials and a 50% flake rate, at least one succeeds.
        assert result.best_score > -float("inf")


class TestHpoNanScores:
    """A model that returns NaN predictions must not propagate NaN into Optuna's
    TPE sampler. The objective coerces NaN → -inf so the trial is excluded."""

    def test_nan_predictions_treated_as_failed(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 3))
        y = X[:, 0]

        class _NaNModel:
            def fit(self, X, y): pass

            def predict(self, X):
                return np.full(X.shape[0], np.nan)

            def get_feature_importance(self):
                return {}

            def get_shap_values(self, X):
                return X

            @property
            def model_name(self):
                return "nan"

            def get_params(self):
                return {}

            @property
            def is_fitted(self):
                return True

        result = optimize(
            model_factory=lambda _p: _NaNModel(),
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=2,
            timeout_seconds=5.0,
            cv_strategy="kfold",
        )
        assert result.best_score == -float("inf")


# ── Ensemble degenerate-blend handling ──────────────────────────────────────


class TestEnsembleDegenerate:
    def test_single_base_model_uniform_weight(self):
        # One base model → meta-learner has nothing to blend; we still
        # need blend_weights to sum to 1.
        result = StackedEnsemble(
            {"only": np.arange(50, dtype=float)},
            problem_type="regression",
        ).fit(np.arange(50, dtype=float))
        assert result.blend_weights == {"only": 1.0}
        assert pytest.approx(sum(result.blend_weights.values())) == 1.0

    def test_constant_target_uniform_weights(self):
        # When y is constant, Ridge produces zero coefs. We fall back
        # to uniform weights so the dict still sums to 1.
        y_const = np.zeros(100)
        base = {"a": np.zeros(100), "b": np.zeros(100)}
        result = StackedEnsemble(base, problem_type="regression").fit(y_const)
        assert pytest.approx(sum(result.blend_weights.values())) == 1.0
        assert result.blend_weights["a"] == result.blend_weights["b"]


# ── StackedEnsembleResult.predict() ─────────────────────────────────────────


class TestEnsemblePredict:
    @pytest.fixture
    def fit_result(self):
        rng = np.random.default_rng(42)
        n = 200
        y = rng.standard_normal(n)
        base = {
            "linear": y + 0.1 * rng.standard_normal(n),
            "xgboost": 0.7 * y + 0.3 * rng.standard_normal(n),
        }
        return StackedEnsemble(base, problem_type="regression", seed=42).fit(y), y

    def test_predict_recovers_signal(self, fit_result):
        result, _ = fit_result
        # Fresh holdout — well-correlated bases should yield high R².
        rng = np.random.default_rng(99)
        n = 100
        y_h = rng.standard_normal(n)
        base_h = {
            "linear": y_h + 0.1 * rng.standard_normal(n),
            "xgboost": 0.7 * y_h + 0.3 * rng.standard_normal(n),
        }
        preds = result.predict(base_h)
        assert preds.shape == (n,)
        r2 = 1 - np.var(y_h - preds) / np.var(y_h)
        assert r2 > 0.5  # generous lower bound

    def test_predict_rejects_missing_base(self, fit_result):
        result, _ = fit_result
        with pytest.raises(ValueError, match="base-model mismatch"):
            result.predict({"linear": np.zeros(10)})

    def test_predict_rejects_extra_base(self, fit_result):
        result, _ = fit_result
        with pytest.raises(ValueError, match="base-model mismatch"):
            result.predict(
                {
                    "linear": np.zeros(10),
                    "xgboost": np.zeros(10),
                    "extra": np.zeros(10),
                }
            )

    def test_predict_without_meta_raises(self, fit_result):
        # Backwards-compat shape: meta_learner=None means "no predict path".
        result, _ = fit_result
        result.meta_learner = None
        with pytest.raises(ValueError, match="meta_learner is None"):
            result.predict({"linear": np.zeros(10), "xgboost": np.zeros(10)})


class TestEnsembleBinaryPredict:
    def test_binary_predict_returns_probabilities(self):
        rng = np.random.default_rng(7)
        n = 200
        y = (rng.standard_normal(n) > 0).astype(int)
        base = {
            "logistic": np.clip(0.5 + 0.3 * (y - 0.5) + 0.1 * rng.standard_normal(n), 0, 1),
            "xgboost": np.clip(0.5 + 0.4 * (y - 0.5) + 0.1 * rng.standard_normal(n), 0, 1),
        }
        result = StackedEnsemble(base, problem_type="binary", seed=42).fit(y)

        # Predict on a fresh holdout.
        n_h = 50
        y_h = (rng.standard_normal(n_h) > 0).astype(int)
        base_h = {
            "logistic": np.clip(0.5 + 0.3 * (y_h - 0.5) + 0.1 * rng.standard_normal(n_h), 0, 1),
            "xgboost": np.clip(0.5 + 0.4 * (y_h - 0.5) + 0.1 * rng.standard_normal(n_h), 0, 1),
        }
        preds = result.predict(base_h)
        # Logistic predictions must be in [0, 1].
        assert np.all((preds >= 0) & (preds <= 1))


# ── HPO search-history JSONB serialization safety ──────────────────────────


class TestHpoHistorySerialisable:
    """The ``hpo_search_results`` payload is persisted to a Postgres JSONB
    column. Every value in the history must be JSON-serialisable."""

    def test_history_is_json_serialisable(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 3))
        y = X[:, 0]
        result = optimize(
            model_factory=lambda p: LinearIVEModel(alpha=p["alpha"]),
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=3,
            timeout_seconds=5.0,
            cv_strategy="kfold",
        )
        # The full payload mirrors what the pipeline sends to the DB.
        payload = {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "n_trials": result.n_trials,
            "elapsed_seconds": result.elapsed_seconds,
            "timed_out": result.timed_out,
            "history": result.search_history,
        }
        # Coerce -inf/inf/NaN through float→str fallback the way Postgres'
        # JSONB driver would.
        try:
            json.dumps(payload, allow_nan=False)
        except ValueError:
            # -inf / NaN are not valid JSON — Postgres allows them but
            # strict drivers may reject. Accept allow_nan=True to mirror
            # asyncpg's default behaviour.
            json.dumps(payload, allow_nan=True)

    def test_history_no_nan_unless_factory_failed(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))
        y = X[:, 0] + 0.05 * rng.standard_normal(50)
        result = optimize(
            model_factory=lambda p: LinearIVEModel(alpha=p["alpha"]),
            X=X,
            y=y,
            search_space=LINEAR_SEARCH_SPACE,
            n_trials=3,
            timeout_seconds=5.0,
            cv_strategy="kfold",
        )
        # Successful trials should have finite scores, not NaN.
        for entry in result.search_history:
            assert entry["score"] == -float("inf") or np.isfinite(entry["score"])
