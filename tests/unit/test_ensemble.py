"""Unit tests for ive.models.ensemble.StackedEnsemble.

The crucial property is leak-free meta predictions: the meta-fold's
training never sees the row it later predicts.
"""

from __future__ import annotations

import numpy as np
import pytest

from ive.models.ensemble import StackedEnsemble, StackedEnsembleResult

pytestmark = pytest.mark.unit


def _two_correlated_oof(n: int = 200, seed: int = 42):
    """Two base-model OOF arrays — both correlated with y, slightly
    different miscalibrations so the meta-learner has work to do."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    base_a = y + 0.1 * rng.standard_normal(n)
    base_b = 0.5 * y + 0.4 * rng.standard_normal(n)
    return {"linear": base_a, "xgboost": base_b}, y


class TestConstructorValidation:
    def test_empty_oof_raises(self):
        with pytest.raises(ValueError, match="at least one base"):
            StackedEnsemble({}, problem_type="regression")

    def test_multiclass_raises(self):
        with pytest.raises(ValueError, match="multiclass"):
            StackedEnsemble({"a": np.zeros(10)}, problem_type="multiclass")

    def test_unknown_problem_type_raises(self):
        with pytest.raises(ValueError, match="Unknown problem_type"):
            StackedEnsemble({"a": np.zeros(10)}, problem_type="ranking")


class TestRegressionMeta:
    def test_returns_result_with_correct_shape(self):
        base, y = _two_correlated_oof()
        ens = StackedEnsemble(base, problem_type="regression", seed=42)
        result = ens.fit(y)
        assert isinstance(result, StackedEnsembleResult)
        assert result.oof_predictions.shape == (len(y),)
        assert result.oof_residuals.shape == (len(y),)
        assert result.meta_kind == "ridge"

    def test_blend_weights_sum_to_one(self):
        base, y = _two_correlated_oof()
        result = StackedEnsemble(base, problem_type="regression").fit(y)
        s = sum(result.blend_weights.values())
        assert pytest.approx(s, abs=1e-9) == 1.0

    def test_meta_learner_coefs_present(self):
        base, y = _two_correlated_oof()
        result = StackedEnsemble(base, problem_type="regression").fit(y)
        assert set(result.meta_learner_coefs.keys()) == {"linear", "xgboost"}

    def test_residuals_match_y_minus_predictions(self):
        base, y = _two_correlated_oof()
        result = StackedEnsemble(base, problem_type="regression").fit(y)
        # Residual is y - oof for the rows that were predicted.
        finite = np.isfinite(result.oof_predictions)
        diff = y[finite] - result.oof_predictions[finite]
        assert np.allclose(diff, result.oof_residuals[finite])

    def test_ensemble_rmse_close_to_or_better_than_best_base(self):
        """The cross-fitted meta blend shouldn't be much worse than
        the best base model — a sanity check that the meta-learner is
        actually doing something useful."""
        base, y = _two_correlated_oof(n=500)
        result = StackedEnsemble(base, problem_type="regression", seed=42).fit(y)

        def _rmse(pred):
            return float(np.sqrt(np.mean((y - pred) ** 2)))

        base_rmses = [_rmse(p) for p in base.values()]
        ens_rmse = _rmse(result.oof_predictions)
        # Generous tolerance — within 1.5× of the best base. The exact
        # gap depends on signal strength; this just guards against gross
        # regressions where meta is wildly worse.
        assert ens_rmse <= 1.5 * min(base_rmses)


class TestBinaryMeta:
    def test_uses_logistic_meta(self):
        rng = np.random.default_rng(7)
        n = 200
        y = (rng.standard_normal(n) > 0).astype(int)
        # Use probabilities, not logits.
        base = {
            "logistic": np.clip(0.5 + 0.3 * (y - 0.5) + 0.1 * rng.standard_normal(n), 0, 1),
            "xgboost": np.clip(0.5 + 0.4 * (y - 0.5) + 0.1 * rng.standard_normal(n), 0, 1),
        }
        result = StackedEnsemble(base, problem_type="binary", seed=42).fit(y)
        assert result.meta_kind == "logistic"
        # Meta predictions should be probabilities in [0, 1].
        finite = np.isfinite(result.oof_predictions)
        assert np.all(
            (result.oof_predictions[finite] >= 0)
            & (result.oof_predictions[finite] <= 1)
        )


class TestNanRobustness:
    def test_nan_rows_stay_nan_in_oof(self):
        # Simulate TimeSeriesSplit's first-chunk NaN in one base model;
        # the ensemble's OOF should also be NaN for those rows.
        rng = np.random.default_rng(0)
        n = 100
        y = rng.standard_normal(n)
        base_a = y + 0.1 * rng.standard_normal(n)
        base_a[:10] = np.nan
        base_b = 0.7 * y + 0.3 * rng.standard_normal(n)
        result = StackedEnsemble(
            {"a": base_a, "b": base_b},
            problem_type="regression",
            seed=42,
        ).fit(y)
        # Rows 0-9 had NaN in base_a → meta features are non-finite
        # → ensemble OOF is NaN there.
        assert np.isnan(result.oof_predictions[:10]).all()
        # Rows 10+ should mostly be finite (modulo K-fold coverage).
        assert np.isfinite(result.oof_predictions[10:]).any()


class TestDeterminism:
    def test_same_seed_same_oof(self):
        base, y = _two_correlated_oof()
        a = StackedEnsemble(base, problem_type="regression", seed=42).fit(y)
        b = StackedEnsemble(base, problem_type="regression", seed=42).fit(y)
        # Same seed → identical fold assignments → identical OOF preds.
        finite = np.isfinite(a.oof_predictions) & np.isfinite(b.oof_predictions)
        assert np.allclose(a.oof_predictions[finite], b.oof_predictions[finite])
