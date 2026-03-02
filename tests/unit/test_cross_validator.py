"""Unit tests for CrossValidator."""

from __future__ import annotations

import numpy as np

from ive.models.cross_validator import CrossValidator, CVResult
from ive.models.linear_model import LinearIVEModel


class TestCrossValidator:
    def test_fit_returns_cv_result(self, small_X_y) -> None:
        """fit() should return a CVResult dataclass."""
        X, y = small_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert isinstance(result, CVResult)

    def test_oof_predictions_correct_length(self, small_X_y) -> None:
        """OOF predictions should have the same length as y."""
        X, y = small_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert len(result.oof_predictions) == len(y)

    def test_oof_residuals_correct_length(self, small_X_y) -> None:
        """OOF residuals should have the same length as y."""
        X, y = small_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert len(result.oof_residuals) == len(y)

    def test_residuals_equal_y_minus_preds(self, small_X_y) -> None:
        """residuals == y - oof_predictions by definition."""
        X, y = small_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        np.testing.assert_allclose(result.oof_residuals, y - result.oof_predictions)

    def test_n_fold_scores_matches_n_splits(self, small_X_y) -> None:
        """fold_scores list should have one entry per CV fold."""
        X, y = small_X_y
        n_splits = 4
        cv = CrossValidator(LinearIVEModel(), n_splits=n_splits)
        result = cv.fit(X, y)
        assert len(result.fold_scores) == n_splits

    def test_fitted_models_count(self, small_X_y) -> None:
        """fitted_models list should contain one model per fold."""
        X, y = small_X_y
        n_splits = 3
        cv = CrossValidator(LinearIVEModel(), n_splits=n_splits)
        result = cv.fit(X, y)
        assert len(result.fitted_models) == n_splits
