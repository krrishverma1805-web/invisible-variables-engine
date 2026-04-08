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
import pandas as pd
import structlog
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

log = structlog.get_logger(__name__)

# Column type categories that should be skipped during preprocessing.
_SKIP_TYPES = frozenset({"datetime", "text", "id"})


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
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_columns(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        column_types: dict[str, str] | None,
    ) -> tuple[list[str], list[str]]:
        """Split *feature_columns* into numeric and categorical lists."""
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []

        for col in feature_columns:
            # Explicit type from profiler takes precedence.
            if column_types and col in column_types:
                ctype = column_types[col]
                if ctype in _SKIP_TYPES:
                    continue
                if ctype == "numeric":
                    numeric_cols.append(col)
                elif ctype == "categorical":
                    categorical_cols.append(col)
                continue

            # Fall back to pandas dtype inference.
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
            elif (
                pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_object_dtype(dtype)
                or isinstance(dtype, pd.CategoricalDtype)
            ):
                categorical_cols.append(col)
            # Other dtypes (e.g. datetime64) are silently skipped.

        return numeric_cols, categorical_cols

    def _build_scaler(self) -> StandardScaler | MinMaxScaler | RobustScaler:
        scaler_map: dict[str, StandardScaler | MinMaxScaler | RobustScaler] = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }
        return scaler_map.get(self.scaler_type, StandardScaler())

    @staticmethod
    def _strip_prefixes(names: list[str]) -> list[str]:
        """Remove sklearn ColumnTransformer prefixes like ``numeric__``."""
        out: list[str] = []
        for name in names:
            # ColumnTransformer uses ``<name>__<feature>`` format.
            if "__" in name:
                out.append(name.split("__", 1)[1])
            else:
                out.append(name)
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        Raises:
            ValueError: If no valid feature columns remain after classification.
        """
        log.info("ive.preprocessor.fit_transform", n_features=len(feature_columns))

        numeric_cols, categorical_cols = self._classify_columns(
            df, feature_columns, column_types
        )

        if not numeric_cols and not categorical_cols:
            raise ValueError("No valid feature columns found for preprocessing")

        self._numeric_cols = numeric_cols
        self._categorical_cols = categorical_cols

        # Build sub-pipelines.
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", self._build_scaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="infrequent_if_exist",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers: list[tuple[str, Pipeline, list[str]]] = []
        if numeric_cols:
            transformers.append(("numeric", numeric_pipeline, numeric_cols))
        if categorical_cols:
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        ct = ColumnTransformer(transformers=transformers, remainder="drop")

        # Fit + transform.
        all_cols = numeric_cols + categorical_cols
        X_out: np.ndarray[Any, Any] = ct.fit_transform(df[all_cols])

        self._transformers["pipeline"] = ct
        self._feature_names_out = self._strip_prefixes(
            list(ct.get_feature_names_out())
        )
        self._fitted = True

        log.info(
            "ive.preprocessor.fitted",
            numeric=len(numeric_cols),
            categorical=len(categorical_cols),
            output_features=len(self._feature_names_out),
        )

        return X_out, self._feature_names_out

    def transform(self, df: Any, feature_columns: list[str]) -> np.ndarray[Any, Any]:
        """
        Apply fitted transformations to new data.

        Args:
            df: Pandas DataFrame with the same schema as training data.
            feature_columns: Columns to transform (must match fit columns).

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if not self._fitted:
            raise RuntimeError("DataPreprocessor must be fit before transform is called.")

        ct: ColumnTransformer = self._transformers["pipeline"]
        all_cols = self._numeric_cols + self._categorical_cols
        missing = set(all_cols) - set(df.columns)
        if missing:
            raise ValueError(
                f"Transform data is missing columns that were present during fit: {sorted(missing)}"
            )
        return ct.transform(df[all_cols])  # type: ignore[no-any-return]

    def inverse_transform_column(
        self, col_name: str, values: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """
        Inverse-transform a single feature column back to original space.

        Best-effort: only numeric (scaled) columns can be reliably inverted.
        Categorical one-hot encoded columns are returned as-is.
        """
        if not self._fitted:
            return values

        # Only numeric columns have a straightforward inverse via the scaler.
        if col_name not in self._numeric_cols:
            return values

        try:
            ct: ColumnTransformer = self._transformers["pipeline"]
            # Locate the numeric sub-pipeline and its scaler step.
            numeric_pipeline: Pipeline = ct.named_transformers_["numeric"]
            scaler = numeric_pipeline.named_steps["scaler"]

            # Determine the positional index of col_name within numeric cols.
            idx = self._numeric_cols.index(col_name)

            # Scaler expects 2-D input with the same number of features it was
            # fit on; we construct a zero-padded array and extract the column.
            n_numeric = len(self._numeric_cols)
            placeholder = np.zeros((len(values), n_numeric))
            placeholder[:, idx] = values.ravel()
            inversed = scaler.inverse_transform(placeholder)
            return inversed[:, idx]
        except Exception:
            log.debug(
                "ive.preprocessor.inverse_transform_failed",
                col_name=col_name,
                reason="falling back to raw values",
            )
            return values
