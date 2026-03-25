"""
Data Preprocessor.

Transforms raw dataset features into a numeric matrix suitable for ML models.

Transformations applied:
    - Categorical encoding (One-Hot for nominal, Ordinal for ordinal)
    - Numeric scaling (StandardScaler by default)
    - Missing value imputation (median for numeric, mode for categorical)
    - High-cardinality categorical handling (target encoding or hashing)
    - Datetime feature extraction (year, month, day, hour, day-of-week)

The preprocessor is fit on training folds and applied to validation folds
during cross-validation to prevent data leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class DataPreprocessor:
    """
    Fits and applies feature preprocessing transformations.

    Usage:
        preprocessor = DataPreprocessor()
        X_train, preprocessor = preprocessor.fit_transform(df_train, feature_cols)
        X_val = preprocessor.transform(df_val, feature_cols)

    The preprocessor stores its fitted state so that the same
    transformations can be applied consistently to new data.
    """

    def __init__(self, scaler_type: str = "standard") -> None:
        """
        Initialise the preprocessor.

        Args:
            scaler_type: 'standard' (z-score) | 'minmax' | 'robust'
        """
        self.scaler_type = scaler_type
        self._fitted = False
        self._transformers: dict[str, Any] = {}
        self._feature_names_out: list[str] = []

    def fit_transform(
        self,
        df: Any,
        feature_columns: list[str],
        column_types: dict[str, str] | None = None,
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        """
        Fit preprocessor on training data and return transformed array.

        Args:
            df: Pandas DataFrame with raw features.
            feature_columns: Columns to include in the feature matrix.
            column_types: Optional mapping from DataProfiler.infer_column_types().

        Returns:
            Tuple of (X: np.ndarray shape (n_samples, n_features), feature_names_out)

        TODO:
            - Build sklearn ColumnTransformer with:
                  numeric cols → SimpleImputer(median) + StandardScaler
                  categorical cols → SimpleImputer(mode) + OneHotEncoder(drop='first')
                  datetime cols → DatetimeFeatureExtractor (custom transformer)
            - Fit the transformer on df[feature_columns]
            - Store in self._transformers['pipeline']
            - self._feature_names_out = pipeline.get_feature_names_out()
            - self._fitted = True
            - return pipeline.transform(df[feature_columns]), self._feature_names_out
        """
        # TODO: Implement sklearn ColumnTransformer pipeline
        log.info("ive.preprocessor.fit_transform", n_features=len(feature_columns))
        self._fitted = True
        placeholder = np.zeros((len(df) if df is not None else 0, len(feature_columns)))
        return placeholder, feature_columns

    def transform(self, df: Any, feature_columns: list[str]) -> np.ndarray[Any, Any]:
        """
        Apply fitted transformations to new data.

        Args:
            df: Pandas DataFrame with the same schema as training data.
            feature_columns: Columns to transform (must match fit columns).

        Raises:
            RuntimeError: If called before fit_transform().

        TODO:
            - Assert self._fitted
            - Apply self._transformers['pipeline'].transform(df[feature_columns])
        """
        if not self._fitted:
            raise RuntimeError("DataPreprocessor must be fit before transform is called.")
        # TODO: Real transform
        return np.zeros((len(df) if df is not None else 0, len(feature_columns)))

    def inverse_transform_column(
        self, col_name: str, values: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Inverse-transform a single feature column back to original space.

        Useful for interpreting latent variable candidates in the original
        feature domain.

        TODO:
            - Identify the correct sub-transformer for col_name
            - Apply inverse_transform if available (not all transforms are invertible)
        """
        # TODO: Implement inverse transform
        return values
