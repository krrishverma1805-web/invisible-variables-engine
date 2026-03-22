"""Unit tests for DataPreprocessor.

Aligned with current public API:
  - DataPreprocessor(scaler_type='standard')
  - fit_transform(df, feature_columns, column_types=None) -> (np.ndarray, list[str])
  - transform(df, feature_columns) -> (np.ndarray, list[str])  — raises if not fitted
  - _fitted: bool attribute
"""

from __future__ import annotations

import numpy as np
import pytest

from ive.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    def test_fit_transform_returns_array(self, simple_regression_df) -> None:
        """fit_transform() should return a 2D numpy array and matching feature list."""
        preprocessor = DataPreprocessor()
        X, names = preprocessor.fit_transform(simple_regression_df, ["x1", "x2"])
        assert isinstance(X, np.ndarray)
        assert X.shape == (len(simple_regression_df), 2)
        assert isinstance(names, list)
        assert len(names) == 2

    def test_transform_before_fit_raises(self, simple_regression_df) -> None:
        """transform() before fit_transform() should raise RuntimeError."""
        preprocessor = DataPreprocessor()
        with pytest.raises(RuntimeError, match="must be fit"):
            preprocessor.transform(simple_regression_df, ["x1"])

    def test_is_fitted_flag(self, simple_regression_df) -> None:
        """_fitted should be False before and True after fit_transform()."""
        preprocessor = DataPreprocessor()
        assert preprocessor._fitted is False
        preprocessor.fit_transform(simple_regression_df, ["x1", "x2"])
        assert preprocessor._fitted is True

    def test_scaler_type_stored(self) -> None:
        """scaler_type parameter should be stored on the instance."""
        preprocessor = DataPreprocessor(scaler_type="minmax")
        assert preprocessor.scaler_type == "minmax"

    def test_output_is_float64(self, simple_regression_df) -> None:
        """fit_transform() output should be float64 numpy array."""
        preprocessor = DataPreprocessor()
        X, _ = preprocessor.fit_transform(simple_regression_df, ["x1", "x2"])
        assert X.dtype == np.float64

    def test_feature_names_match_columns(self, simple_regression_df) -> None:
        """Returned feature names must match the requested columns."""
        preprocessor = DataPreprocessor()
        _, names = preprocessor.fit_transform(simple_regression_df, ["x1", "x2"])
        assert names == ["x1", "x2"]
