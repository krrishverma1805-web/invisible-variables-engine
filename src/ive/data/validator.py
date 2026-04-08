"""
Data Validator.

Validates dataset schema and quality before pipeline execution.
Raises descriptive exceptions for unrecoverable issues and emits
warnings for recoverable problems (high missingness, etc.).

Validation checks:
    - Target column exists in DataFrame
    - Target column has no nulls (required for supervised modelling)
    - Minimum rows and columns requirements
    - No fully-duplicate rows (warn only)
    - No zero-variance columns (will be dropped with warning)
    - Data type sanity checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

_MIN_ROWS = 50
_MIN_FEATURES = 2


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)


class DataValidator:
    """
    Validates a dataset before the IVE pipeline begins processing.

    Raises:
        ValueError: If a critical validation error is found.
    """

    def validate(self, df: Any, target_column: str) -> ValidationResult:
        """
        Validate the DataFrame against IVE requirements.

        Args:
            df: Pandas DataFrame to validate.
            target_column: Name of the target/label column.

        Returns:
            ValidationResult with errors and warnings.

        Raises:
            ValueError: If any critical validation check fails.

        TODO:
            - Check target_column in df.columns → error if missing
            - Check df[target_column].isnull().sum() == 0 → error if any nulls
            - Check len(df) >= _MIN_ROWS → error if too few rows
            - Check (len(df.columns) - 1) >= _MIN_FEATURES → error if too few features
            - Check for zero-variance columns → add to dropped_columns, warn
            - Check for 100% duplicate rows → warn
            - Log final ValidationResult
        """
        errors: list[str] = []
        warnings: list[str] = []
        dropped_columns: list[str] = []

        if df is None:
            errors.append("DataFrame is None")
            return ValidationResult(is_valid=False, errors=errors)

        # Check target column presence
        if target_column not in df.columns:
            errors.append(
                f"Target column '{target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)[:10]}"
            )

        # Check target column has no nulls (only if column exists)
        if target_column in df.columns:
            null_count = int(df[target_column].isnull().sum())
            if null_count > 0:
                errors.append(
                    f"Target column has {null_count} null values. "
                    "All target values must be non-null."
                )

        # Check minimum rows
        if len(df) < _MIN_ROWS:
            errors.append(
                f"Dataset has {len(df)} rows, minimum required is {_MIN_ROWS}."
            )

        # Check minimum features (excluding target column)
        num_features = len(df.columns) - 1
        if num_features < _MIN_FEATURES:
            errors.append(
                f"Dataset has {num_features} features, minimum required is {_MIN_FEATURES}."
            )

        # Check for zero-variance columns
        for col in df.select_dtypes(include="number").columns:
            if df[col].nunique() <= 1:
                dropped_columns.append(col)
                warnings.append(
                    f"Column '{col}' has zero variance and will be dropped."
                )

        # Check for duplicate rows
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            pct = round(dup_count / len(df) * 100, 1)
            warnings.append(
                f"Dataset contains {dup_count} duplicate rows ({pct}% of data)."
            )

        is_valid = len(errors) == 0
        if not is_valid:
            for err in errors:
                log.error("ive.validator.error", error=err)
            raise ValueError(f"Dataset validation failed: {'; '.join(errors)}")

        for warn in warnings:
            log.warning("ive.validator.warning", warning=warn)

        log.info("ive.validator.passed", warnings=len(warnings))
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=warnings,
            dropped_columns=dropped_columns,
        )

    def check_target_suitability(
        self,
        target: Any,
        task_type: str = "auto",
    ) -> str:
        """
        Determine whether the target is suitable for regression or classification.

        Returns:
            'regression' or 'classification'

        TODO:
            - If task_type == 'auto': heuristically detect from target distribution
              (continuous float → regression, integer/string ≤ 20 unique → classification)
            - Validate the detected task type is supported
        """
        _VALID_TYPES = ("regression", "classification")

        if task_type != "auto":
            if task_type not in _VALID_TYPES:
                raise ValueError(
                    f"Unsupported task_type '{task_type}'. "
                    f"Must be one of {_VALID_TYPES} or 'auto'."
                )
            return task_type

        # Auto-detect from target series
        target_series = pd.Series(target) if not isinstance(target, pd.Series) else target

        if (
            target_series.dtype == "bool"
            or pd.api.types.is_object_dtype(target_series)
            or pd.api.types.is_string_dtype(target_series)
        ):
            return "classification"

        if pd.api.types.is_float_dtype(target_series) and target_series.nunique() > 20:
            return "regression"

        if pd.api.types.is_integer_dtype(target_series) and target_series.nunique() <= 20:
            return "classification"

        return "regression"
