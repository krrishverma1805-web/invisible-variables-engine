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

        # TODO: Implement remaining checks (see docstring)

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
        # TODO: Detect task type from target column
        return "regression"
