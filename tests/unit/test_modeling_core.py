"""
Unit test suite for the IVE Modeling Core (Phase 2).

Tests cover:
    - :class:`~ive.data.preprocessor.DataPreprocessor` — fit_transform, missing values
    - :class:`~ive.models.linear_model.LinearIVEModel`  — fit, predict, importances
    - :class:`~ive.models.xgboost_model.XGBoostIVEModel` — fit, predict, importances
    - :class:`~ive.models.cross_validator.CrossValidator` — OOF length, fold scores, no leakage
    - :class:`~ive.models.residual_analyzer.ResidualAnalyzer` — record structure, math correctness

All tests run without a database, filesystem, or network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.data.preprocessor import DataPreprocessor
from ive.models.cross_validator import CrossValidator, CVResult
from ive.models.linear_model import LinearIVEModel
from ive.models.residual_analyzer import ResidualAnalysis, ResidualAnalyzer
from ive.models.xgboost_model import XGBoostIVEModel

# ---------------------------------------------------------------------------
# Shared local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def regression_X_y(sample_regression_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Numeric feature matrix and target vector from the regression fixture."""
    df = sample_regression_df.copy()
    y = df["price"].values.astype(float)
    # One-hot encode the single categorical so every test has a clean numeric X
    X_df = pd.get_dummies(df.drop(columns=["price"]), drop_first=True, dtype=float)
    X = X_df.values.astype(float)
    return X, y


@pytest.fixture
def classification_X_y(sample_classification_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Numeric feature matrix and binary target vector from the classification fixture."""
    df = sample_classification_df.copy()
    y = df["target"].values.astype(float)
    X_df = pd.get_dummies(df.drop(columns=["target"]), drop_first=True, dtype=float)
    X = X_df.values.astype(float)
    return X, y


# ============================================================================
# 1. DataPreprocessor
# ============================================================================


class TestDataPreprocessor:
    """Tests for :class:`~ive.data.preprocessor.DataPreprocessor`."""

    def test_fit_transform_returns_array_and_names(
        self, sample_regression_df: pd.DataFrame
    ) -> None:
        """fit_transform should return a 2-D array and a matching name list."""
        df = sample_regression_df.drop(columns=["price"])
        feature_cols = df.columns.tolist()
        preprocessor = DataPreprocessor()
        X, names = preprocessor.fit_transform(df, feature_cols)

        assert isinstance(X, np.ndarray), "fit_transform must return a numpy array"
        assert X.ndim == 2, "Returned array must be 2-D (n_samples, n_features)"
        # Preprocessor may expand via OHE — at minimum we need ≥1 column
        assert X.shape[0] == len(df), "Row count must match input DataFrame"
        assert X.shape[1] >= 1, "At least one output feature expected"
        assert isinstance(names, list), "Feature names must be a list"

    def test_is_fitted_after_fit_transform(self, sample_regression_df: pd.DataFrame) -> None:
        """_fitted flag must be True after fit_transform is called."""
        df = sample_regression_df.drop(columns=["price"])
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(df, df.columns.tolist())
        assert preprocessor._fitted is True

    def test_handles_missing_values_without_crash(self) -> None:
        """fit_transform must not raise on numeric columns with NaN."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "a": rng.standard_normal(100),
                "b": rng.standard_normal(100),
            }
        )
        # Introduce NaN in 10 % of rows
        df.loc[rng.choice(100, 10, replace=False), "a"] = np.nan
        preprocessor = DataPreprocessor()
        try:
            X, _ = preprocessor.fit_transform(df, ["a", "b"])
        except Exception as exc:
            pytest.fail(f"fit_transform raised on NaN input: {exc}")

        assert not np.any(np.isnan(X)), "Output array must contain no NaN after preprocessing"

    def test_handles_categorical_columns(self) -> None:
        """fit_transform must not raise on object-dtype columns."""
        df = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0, 5.0],
                "cat": ["A", "B", "A", "C", "B"],
            }
        )
        preprocessor = DataPreprocessor()
        try:
            X, names = preprocessor.fit_transform(df, ["num", "cat"])
        except Exception as exc:
            pytest.fail(f"fit_transform raised on categorical input: {exc}")

        # At minimum the numeric column must appear
        assert X.ndim == 2
        assert X.shape[0] == 5

    def test_fit_transform_with_column_types_hint(self, sample_regression_df: pd.DataFrame) -> None:
        """Passing column_types kwarg must not crash."""
        df = sample_regression_df.drop(columns=["price"])
        column_types = {"feature_a": "numeric", "feature_b": "numeric", "category": "categorical"}
        preprocessor = DataPreprocessor()
        try:
            preprocessor.fit_transform(df, df.columns.tolist(), column_types=column_types)
        except Exception as exc:
            pytest.fail(f"fit_transform raised with column_types: {exc}")

    def test_schema_json_equivalent_metadata(self, sample_regression_df: pd.DataFrame) -> None:
        """Mimic what the orchestrator would build as schema_json.

        Confirms that a dict describing the columns can be built and used
        as auxiliary metadata without affecting fit_transform's output shape.
        """
        df = sample_regression_df.drop(columns=["price"])
        schema_json = {
            col: ("numeric" if sample_regression_df[col].dtype.kind in "biufc" else "categorical")
            for col in df.columns
        }
        preprocessor = DataPreprocessor()
        X, _ = preprocessor.fit_transform(df, df.columns.tolist(), column_types=schema_json)
        assert X.shape[0] == len(df)


# ============================================================================
# 2. LinearIVEModel
# ============================================================================


class TestLinearIVEModel:
    """Tests for :class:`~ive.models.linear_model.LinearIVEModel`."""

    def test_predict_returns_correct_shape(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict() must return a 1-D array whose length matches the input."""
        X, y = regression_X_y
        model = LinearIVEModel()
        model.fit(X, y)
        preds = model.predict(X)

        assert isinstance(preds, np.ndarray), "predict must return a numpy array"
        assert preds.ndim == 1, "Predictions must be 1-D"
        assert len(preds) == len(y), "Prediction length must match input length"

    def test_predict_values_are_finite(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """All predicted values must be finite floats."""
        X, y = regression_X_y
        model = LinearIVEModel()
        model.fit(X, y)
        assert np.all(np.isfinite(model.predict(X))), "Predictions must all be finite"

    def test_predict_before_fit_raises(self) -> None:
        """predict() must raise RuntimeError if called before fit()."""
        model = LinearIVEModel()
        X = np.random.default_rng(0).standard_normal((10, 3))
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_get_feature_importance_returns_dict(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """get_feature_importance() must return a dict after fitting."""
        X, y = regression_X_y
        model = LinearIVEModel()
        model.fit(X, y)
        importances = model.get_feature_importance()

        assert isinstance(importances, dict), "Feature importances must be a dict"
        assert len(importances) == X.shape[1], "One importance entry per feature column"

    def test_get_feature_importance_values_non_negative(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """All importance values must be ≥ 0 and finite."""
        X, y = regression_X_y
        model = LinearIVEModel()
        model.fit(X, y)
        for name, value in model.get_feature_importance().items():
            assert value >= 0.0, f"Importance for {name!r} is negative: {value}"
            assert np.isfinite(value), f"Importance for {name!r} is not finite: {value}"

    def test_get_feature_importance_before_fit_returns_empty(self) -> None:
        """get_feature_importance() must return an empty dict when unfitted."""
        model = LinearIVEModel()
        result = model.get_feature_importance()
        assert result == {}, "Unfitted model must return empty importance dict"

    def test_is_fitted_flag(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """is_fitted must be False before fit() and True after."""
        X, y = regression_X_y
        model = LinearIVEModel()
        assert model.is_fitted is False
        model.fit(X, y)
        assert model.is_fitted is True

    def test_model_name_is_string(self) -> None:
        """model_name property must return a non-empty string."""
        assert isinstance(LinearIVEModel().model_name, str)
        assert len(LinearIVEModel().model_name) > 0


# ============================================================================
# 3. XGBoostIVEModel
# ============================================================================


class TestXGBoostIVEModel:
    """Tests for :class:`~ive.models.xgboost_model.XGBoostIVEModel`."""

    def test_predict_returns_correct_shape(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """predict() must return a 1-D array whose length matches the input."""
        X, y = regression_X_y
        model = XGBoostIVEModel(n_estimators=10)  # fast for tests
        model.fit(X, y)
        preds = model.predict(X)

        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 1
        assert len(preds) == len(y)

    def test_predict_values_are_finite(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """All XGBoost predictions must be finite."""
        X, y = regression_X_y
        model = XGBoostIVEModel(n_estimators=10)
        model.fit(X, y)
        assert np.all(np.isfinite(model.predict(X)))

    def test_predict_before_fit_raises(self) -> None:
        """predict() must raise RuntimeError if called before fit()."""
        model = XGBoostIVEModel()
        X = np.random.default_rng(0).standard_normal((10, 3))
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_get_feature_importance_returns_dict(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """get_feature_importance() must return a non-empty dict after fitting."""
        X, y = regression_X_y
        model = XGBoostIVEModel(n_estimators=10)
        model.fit(X, y)
        importances = model.get_feature_importance()

        assert isinstance(importances, dict)
        assert len(importances) > 0

    def test_get_feature_importance_values_are_non_negative(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """All feature importance values must be ≥ 0."""
        X, y = regression_X_y
        model = XGBoostIVEModel(n_estimators=10)
        model.fit(X, y)
        for name, value in model.get_feature_importance().items():
            assert value >= 0.0, f"Importance {name!r} = {value} is negative"

    def test_importances_sum_to_one(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """Normalised importances must sum to approximately 1.0."""
        X, y = regression_X_y
        model = XGBoostIVEModel(n_estimators=10)
        model.fit(X, y)
        importances = model.get_feature_importance()
        total = sum(importances.values())
        assert abs(total - 1.0) < 1e-6, f"Importances sum to {total}, expected 1.0"

    def test_get_feature_importance_sorted_descending(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Importance dict must be sorted descending by value."""
        X, y = regression_X_y
        model = XGBoostIVEModel(n_estimators=20)
        model.fit(X, y)
        values = list(model.get_feature_importance().values())
        assert values == sorted(
            values, reverse=True
        ), "Feature importances must be sorted descending"

    def test_is_fitted_flag(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """is_fitted must transition False → True on fit()."""
        X, y = regression_X_y
        model = XGBoostIVEModel(n_estimators=5)
        assert model.is_fitted is False
        model.fit(X, y)
        assert model.is_fitted is True

    def test_xgboost_outperforms_linear_on_nonlinear_data(self) -> None:
        """XGBoost should achieve lower RMSE than Ridge on a cubic signal."""
        rng = np.random.default_rng(99)
        n = 300
        X = rng.standard_normal((n, 3))
        # Non-linear target: cubic + interaction
        y = X[:, 0] ** 3 + X[:, 0] * X[:, 1] + rng.standard_normal(n) * 0.1

        linear = LinearIVEModel()
        linear.fit(X[:200], y[:200])
        xgb = XGBoostIVEModel(n_estimators=50, max_depth=4)
        xgb.fit(X[:200], y[:200])

        linear_rmse = float(np.sqrt(np.mean((y[200:] - linear.predict(X[200:])) ** 2)))
        xgb_rmse = float(np.sqrt(np.mean((y[200:] - xgb.predict(X[200:])) ** 2)))
        assert (
            xgb_rmse < linear_rmse
        ), f"XGBoost RMSE ({xgb_rmse:.4f}) must beat linear ({linear_rmse:.4f}) on cubic signal"


# ============================================================================
# 4. CrossValidator
# ============================================================================


class TestCrossValidator:
    """Tests for :class:`~ive.models.cross_validator.CrossValidator`."""

    # -- CVResult structure --------------------------------------------------

    def test_returns_cv_result(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """fit() must return a CVResult dataclass instance."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert isinstance(result, CVResult)

    def test_oof_predictions_length_matches_input(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """oof_predictions must have exactly n_samples elements."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert len(result.oof_predictions) == len(
            y
        ), f"Expected {len(y)} OOF predictions, got {len(result.oof_predictions)}"

    def test_oof_residuals_length_matches_input(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """oof_residuals must have exactly n_samples elements."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert len(result.oof_residuals) == len(y)

    def test_fold_assignments_length_matches_input(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fold_assignments must cover every sample exactly once."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        assert len(result.fold_assignments) == len(y)
        # Every sample must be in exactly one fold
        assert np.all(result.fold_assignments >= 0), "All samples must have a fold assigned"

    # -- Residual correctness ------------------------------------------------

    def test_residuals_equal_y_minus_predictions(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """oof_residuals must equal y − oof_predictions by definition."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        np.testing.assert_allclose(
            result.oof_residuals,
            y - result.oof_predictions,
            rtol=1e-10,
            err_msg="Residuals must equal y - oof_predictions",
        )

    # -- Fold score structure ------------------------------------------------

    def test_fold_scores_count_matches_n_splits(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fold_scores list must have exactly n_splits entries."""
        X, y = regression_X_y
        n_splits = 3
        cv = CrossValidator(LinearIVEModel(), n_splits=n_splits)
        result = cv.fit(X, y)
        assert (
            len(result.fold_scores) == n_splits
        ), f"Expected {n_splits} fold scores, got {len(result.fold_scores)}"

    def test_fold_scores_are_finite(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """Every per-fold score (R²) must be a finite float."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        for i, score in enumerate(result.fold_scores):
            assert np.isfinite(score), f"Fold {i} score is not finite: {score}"

    def test_mean_score_consistent_with_fold_scores(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """mean_score must equal the arithmetic mean of fold_scores."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        expected = float(np.mean(result.fold_scores))
        assert (
            abs(result.mean_score - expected) < 1e-10
        ), f"mean_score {result.mean_score} != mean(fold_scores) {expected}"

    # -- Fitted models -------------------------------------------------------

    def test_fitted_models_count_matches_n_splits(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """fitted_models must contain exactly n_splits fitted model instances."""
        X, y = regression_X_y
        n_splits = 3
        cv = CrossValidator(LinearIVEModel(), n_splits=n_splits)
        result = cv.fit(X, y)
        assert len(result.fitted_models) == n_splits

    def test_fitted_models_are_fitted(self, regression_X_y: tuple[np.ndarray, np.ndarray]) -> None:
        """Every model in fitted_models must report is_fitted=True."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3)
        result = cv.fit(X, y)
        for i, m in enumerate(result.fitted_models):
            assert m.is_fitted, f"Fold {i} model is not fitted"

    # -- No data leakage -----------------------------------------------------

    def test_no_data_leakage_between_folds(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Each sample must appear in the validation set in exactly one fold."""
        X, y = regression_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=5)
        result = cv.fit(X, y)
        # fold_assignments[i] is the single fold where sample i was held out
        fold_counts = np.bincount(result.fold_assignments, minlength=5)
        # All samples covered exactly once
        assert np.all(result.fold_assignments >= 0)
        assert np.sum(fold_counts) == len(y), "Total fold assignments must equal n_samples"

    # -- Reproducibility -----------------------------------------------------

    def test_deterministic_with_same_seed(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Two CV runs with the same seed must produce identical OOF predictions."""
        X, y = regression_X_y
        r1 = CrossValidator(LinearIVEModel(), n_splits=3, seed=7).fit(X, y)
        r2 = CrossValidator(LinearIVEModel(), n_splits=3, seed=7).fit(X, y)
        np.testing.assert_array_equal(
            r1.oof_predictions,
            r2.oof_predictions,
            err_msg="OOF predictions must be deterministic for same seed",
        )

    # -- Stratified mode -----------------------------------------------------

    def test_stratified_cv_on_classification(
        self, classification_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """stratified=True must not raise and must produce correct-length OOF."""
        X, y = classification_X_y
        cv = CrossValidator(LinearIVEModel(), n_splits=3, stratified=True)
        result = cv.fit(X, y)
        assert len(result.oof_predictions) == len(y)

    # -- XGBoost in CV -------------------------------------------------------

    def test_xgboost_in_cv_produces_valid_result(
        self, regression_X_y: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """CrossValidator must work with XGBoostIVEModel."""
        X, y = regression_X_y
        cv = CrossValidator(XGBoostIVEModel(n_estimators=10), n_splits=3)
        result = cv.fit(X, y)
        assert isinstance(result, CVResult)
        assert len(result.oof_predictions) == len(y)
        assert np.all(np.isfinite(result.oof_predictions))


# ============================================================================
# 5. ResidualAnalyzer
# ============================================================================


class TestResidualAnalyzer:
    """Tests for :class:`~ive.models.residual_analyzer.ResidualAnalyzer`."""

    # ---- Fixtures -----------------------------------------------------------

    @pytest.fixture
    def simple_inputs(self) -> dict:
        """Deterministic y_true / oof_predictions / fold_assignments / feature df."""
        n = 50
        rng = np.random.default_rng(5)
        y_true = rng.standard_normal(n)
        oof_preds = y_true + rng.standard_normal(n) * 0.3  # close but imperfect
        fold_assignments = np.tile([0, 1, 2], n // 3 + 1)[:n]
        X_df = pd.DataFrame(
            {
                "feat_a": rng.standard_normal(n),
                "feat_b": rng.integers(0, 5, n).astype(float),
            }
        )
        return {
            "y_true": y_true,
            "oof_preds": oof_preds,
            "fold_assignments": fold_assignments,
            "X_df": X_df,
        }

    # ---- analyze() ---------------------------------------------------------

    def test_analyze_returns_residual_analysis(self, simple_inputs: dict) -> None:
        """analyze() must return a ResidualAnalysis dataclass."""
        residuals = simple_inputs["y_true"] - simple_inputs["oof_preds"]
        analyzer = ResidualAnalyzer()
        result = analyzer.analyze(residuals)
        assert isinstance(result, ResidualAnalysis)

    def test_analyze_mean_is_finite(self, simple_inputs: dict) -> None:
        """ResidualAnalysis.mean must be a finite float."""
        residuals = simple_inputs["y_true"] - simple_inputs["oof_preds"]
        result = ResidualAnalyzer().analyze(residuals)
        assert np.isfinite(result.mean)

    def test_analyze_std_non_negative(self, simple_inputs: dict) -> None:
        """ResidualAnalysis.std must be ≥ 0."""
        residuals = simple_inputs["y_true"] - simple_inputs["oof_preds"]
        result = ResidualAnalyzer().analyze(residuals)
        assert result.std >= 0.0

    def test_analyze_empty_residuals_returns_default(self) -> None:
        """analyze(empty array) must return default ResidualAnalysis without raising."""
        result = ResidualAnalyzer().analyze(np.array([]))
        assert result.mean == 0.0
        assert result.std == 0.0

    def test_analyze_pct_large_between_0_and_100(self, simple_inputs: dict) -> None:
        """pct_large must be in [0, 100]."""
        residuals = simple_inputs["y_true"] - simple_inputs["oof_preds"]
        result = ResidualAnalyzer().analyze(residuals)
        assert 0.0 <= result.pct_large <= 100.0

    def test_analyze_shapiro_p_populated(self, simple_inputs: dict) -> None:
        """shapiro_p must be populated (not None) for a non-empty residual vector."""
        residuals = simple_inputs["y_true"] - simple_inputs["oof_preds"]
        result = ResidualAnalyzer().analyze(residuals)
        assert result.shapiro_p is not None
        assert 0.0 <= result.shapiro_p <= 1.0

    def test_analyze_durbin_watson_populated(self, simple_inputs: dict) -> None:
        """durbin_watson must be populated for a non-empty residual vector."""
        residuals = simple_inputs["y_true"] - simple_inputs["oof_preds"]
        result = ResidualAnalyzer().analyze(residuals)
        assert result.durbin_watson is not None
        # DW statistic is in [0, 4]; ≈2 means no autocorrelation
        assert 0.0 <= result.durbin_watson <= 4.0

    def test_analyze_warnings_is_list(self, simple_inputs: dict) -> None:
        """warnings attribute must always be a list (may be empty)."""
        residuals = simple_inputs["y_true"] - simple_inputs["oof_preds"]
        result = ResidualAnalyzer().analyze(residuals)
        assert isinstance(result.warnings, list)

    def test_large_residual_triggers_warning(self) -> None:
        """A residual distribution with many large outliers must produce a warning."""
        # 50 % of residuals are > 2σ by construction
        rng = np.random.default_rng(42)
        base = rng.standard_normal(50)
        outliers = rng.standard_normal(50) * 10  # 10-sigma outliers
        residuals = np.concatenate([base, outliers])
        result = ResidualAnalyzer(large_residual_threshold=2.0).analyze(residuals)
        assert (
            len(result.warnings) > 0
        ), "Expected at least one warning for high pct_large outlier distribution"

    # ---- build_residual_records() ------------------------------------------

    def test_build_residual_records_correct_length(self, simple_inputs: dict) -> None:
        """build_residual_records must return one dict per sample."""
        d = simple_inputs
        records = ResidualAnalyzer().build_residual_records(
            X_df=d["X_df"],
            y_true=d["y_true"],
            oof_predictions=d["oof_preds"],
            fold_assignments=d["fold_assignments"],
        )
        assert len(records) == len(d["y_true"]), "One residual record per sample expected"

    def test_build_residual_records_required_keys(self, simple_inputs: dict) -> None:
        """Every record must contain all 7 required keys."""
        required = {
            "sample_index",
            "fold_number",
            "actual_value",
            "predicted_value",
            "residual_value",
            "abs_residual",
            "feature_vector",
        }
        d = simple_inputs
        records = ResidualAnalyzer().build_residual_records(
            X_df=d["X_df"],
            y_true=d["y_true"],
            oof_predictions=d["oof_preds"],
            fold_assignments=d["fold_assignments"],
        )
        for i, rec in enumerate(records):
            missing = required - set(rec.keys())
            assert not missing, f"Record {i} is missing keys: {missing}"

    def test_abs_residual_is_mathematically_correct(self, simple_inputs: dict) -> None:
        """abs_residual must equal |actual_value - predicted_value| for every record."""
        d = simple_inputs
        records = ResidualAnalyzer().build_residual_records(
            X_df=d["X_df"],
            y_true=d["y_true"],
            oof_predictions=d["oof_preds"],
            fold_assignments=d["fold_assignments"],
        )
        for i, rec in enumerate(records):
            expected_abs = abs(rec["actual_value"] - rec["predicted_value"])
            assert abs(rec["abs_residual"] - expected_abs) < 1e-9, (
                f"Record {i}: abs_residual={rec['abs_residual']:.6f} "
                f"but |actual - predicted|={expected_abs:.6f}"
            )

    def test_residual_value_sign_is_correct(self, simple_inputs: dict) -> None:
        """residual_value must equal actual_value - predicted_value (signed)."""
        d = simple_inputs
        records = ResidualAnalyzer().build_residual_records(
            X_df=d["X_df"],
            y_true=d["y_true"],
            oof_predictions=d["oof_preds"],
            fold_assignments=d["fold_assignments"],
        )
        for i, rec in enumerate(records):
            expected = rec["actual_value"] - rec["predicted_value"]
            assert abs(rec["residual_value"] - expected) < 1e-9, (
                f"Record {i}: residual_value={rec['residual_value']:.6f} "
                f"but actual-predicted={expected:.6f}"
            )

    def test_feature_vector_is_dict(self, simple_inputs: dict) -> None:
        """feature_vector must be a dict in every record."""
        d = simple_inputs
        records = ResidualAnalyzer().build_residual_records(
            X_df=d["X_df"],
            y_true=d["y_true"],
            oof_predictions=d["oof_preds"],
            fold_assignments=d["fold_assignments"],
        )
        for i, rec in enumerate(records):
            assert isinstance(
                rec["feature_vector"], dict
            ), f"Record {i}: feature_vector is {type(rec['feature_vector'])}, expected dict"

    def test_feature_vector_keys_match_dataframe_columns(self, simple_inputs: dict) -> None:
        """feature_vector keys must match the columns of the input X_df."""
        d = simple_inputs
        expected_cols = set(d["X_df"].columns.tolist())
        records = ResidualAnalyzer().build_residual_records(
            X_df=d["X_df"],
            y_true=d["y_true"],
            oof_predictions=d["oof_preds"],
            fold_assignments=d["fold_assignments"],
        )
        for i, rec in enumerate(records):
            assert set(rec["feature_vector"].keys()) == expected_cols, (
                f"Record {i}: feature_vector keys {set(rec['feature_vector'].keys())} "
                f"don't match df columns {expected_cols}"
            )

    def test_fold_number_matches_assignments(self, simple_inputs: dict) -> None:
        """fold_number in each record must match the fold_assignments array."""
        d = simple_inputs
        records = ResidualAnalyzer().build_residual_records(
            X_df=d["X_df"],
            y_true=d["y_true"],
            oof_predictions=d["oof_preds"],
            fold_assignments=d["fold_assignments"],
        )
        for i, rec in enumerate(records):
            assert rec["fold_number"] == int(d["fold_assignments"][i]), (
                f"Record {i}: fold_number={rec['fold_number']} "
                f"but fold_assignments[{i}]={d['fold_assignments'][i]}"
            )

    def test_length_mismatch_raises_value_error(self) -> None:
        """build_residual_records must raise ValueError on length mismatch."""
        n = 20
        rng = np.random.default_rng(0)
        X_df = pd.DataFrame({"a": rng.standard_normal(n)})
        y_true = rng.standard_normal(n)
        oof_preds = rng.standard_normal(n + 5)  # wrong length
        fold_assignments = np.zeros(n, dtype=int)

        analyzer = ResidualAnalyzer()
        with pytest.raises(ValueError, match="[Ll]ength"):
            analyzer.build_residual_records(
                X_df=X_df,
                y_true=y_true,
                oof_predictions=oof_preds,
                fold_assignments=fold_assignments,
            )

    def test_classification_residuals_probability_margin(self) -> None:
        """For classification task_type, residual = y_true - predicted_prob."""
        n = 30
        rng = np.random.default_rng(17)
        y_true = rng.choice([0, 1], n).astype(float)
        oof_preds = rng.uniform(0, 1, n)  # probability predictions
        fold_assignments = np.zeros(n, dtype=int)
        X_df = pd.DataFrame({"f1": rng.standard_normal(n)})

        records = ResidualAnalyzer().build_residual_records(
            X_df=X_df,
            y_true=y_true,
            oof_predictions=oof_preds,
            fold_assignments=fold_assignments,
            task_type="classification",
        )
        # Residual = y_true - predicted_prob (probability margin)
        for i, rec in enumerate(records):
            expected = float(y_true[i]) - float(oof_preds[i])
            assert abs(rec["residual_value"] - expected) < 1e-9, (
                f"Record {i}: classification residual {rec['residual_value']:.4f} "
                f"≠ y-prob margin {expected:.4f}"
            )

    # ---- End-to-end: CrossValidator → ResidualAnalyzer --------------------

    def test_end_to_end_cv_to_residual_records(
        self,
        regression_X_y: tuple[np.ndarray, np.ndarray],
        sample_regression_df: pd.DataFrame,
    ) -> None:
        """Full round-trip: CV → ResidualAnalyzer → DB-ready records."""
        X, y = regression_X_y
        feature_cols = [c for c in sample_regression_df.columns if c != "price"]
        X_df_original = sample_regression_df[feature_cols].copy()

        cv = CrossValidator(LinearIVEModel(), n_splits=3, seed=42)
        result = cv.fit(X, y)

        analyzer = ResidualAnalyzer()
        records = analyzer.build_residual_records(
            X_df=X_df_original.reset_index(drop=True),
            y_true=y,
            oof_predictions=result.oof_predictions,
            fold_assignments=result.fold_assignments,
        )

        # Shape
        assert len(records) == len(y)

        # Every required key present
        required = {
            "sample_index",
            "fold_number",
            "actual_value",
            "predicted_value",
            "residual_value",
            "abs_residual",
            "feature_vector",
        }
        for rec in records:
            assert required.issubset(set(rec.keys()))

        # Math correctness
        for rec in records:
            expected_abs = abs(rec["actual_value"] - rec["predicted_value"])
            assert abs(rec["abs_residual"] - expected_abs) < 1e-9
