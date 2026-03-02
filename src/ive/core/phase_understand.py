"""
Phase 1 — Understand.

Profiles the dataset to understand its structure before modelling:
- Column type inference (continuous, categorical, ordinal, datetime, text)
- Univariate statistics (mean, std, skew, kurtosis, quantiles)
- Missing value analysis
- Pairwise correlation matrix and multicollinearity detection
- Target variable distribution analysis

Outputs written to PipelineContext:
    ctx.df              — cleaned DataFrame
    ctx.column_types    — dict mapping column → inferred type
    ctx.profile         — serialisable profile dict
    ctx.feature_columns — list of usable feature column names
    ctx.target_series   — target column as pandas Series
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import structlog

from ive.core.pipeline import PhaseResult, PipelineContext

log = structlog.get_logger(__name__)


class PhaseBase(ABC):
    """Abstract base class for all pipeline phases."""

    @abstractmethod
    def get_phase_name(self) -> str:
        """Return the canonical phase identifier string."""
        ...

    @abstractmethod
    async def execute(self, ctx: PipelineContext) -> PhaseResult:
        """Execute this phase, mutating ctx with outputs."""
        ...


class PhaseUnderstand(PhaseBase):
    """
    Phase 1: Understand the dataset.

    Runs data ingestion, profiling, and validation before any modelling.
    """

    def get_phase_name(self) -> str:
        return "understand"

    async def execute(self, ctx: PipelineContext) -> PhaseResult:
        """
        Load the dataset, profile it, and populate the pipeline context.

        Steps:
            1. Load dataset from ctx.data_path via DataIngestion
            2. Validate schema and target column presence
            3. Profile columns (types, stats, missing values)
            4. Compute correlation matrix; flag multicollinearity
            5. Separate target from features; store in ctx

        TODO:
            - Instantiate DataIngestion and call .load(ctx.data_path)
            - Call DataProfiler.profile(df) → store result in ctx.profile
            - Call DataValidator.validate(df, ctx.config.target_column)
            - Call DataPreprocessor.identify_column_types(df)
        """
        start = time.monotonic()
        log.info("ive.phase.understand.start", experiment_id=str(ctx.experiment_id))

        # TODO: Load data
        # from ive.data.ingestion import DataIngestion
        # ingestion = DataIngestion()
        # ctx.df = await ingestion.load(ctx.data_path)

        # TODO: Profile data
        # from ive.data.profiler import DataProfiler
        # profiler = DataProfiler()
        # ctx.profile = profiler.profile(ctx.df)
        # ctx.column_types = profiler.infer_column_types(ctx.df)

        # TODO: Validate data
        # from ive.data.validator import DataValidator
        # validator = DataValidator()
        # validator.validate(ctx.df, target_column=ctx.config.target_column)

        # TODO: Split target from features
        # ctx.target_series = ctx.df[ctx.config.target_column]
        # ctx.feature_columns = [c for c in ctx.df.columns if c != ctx.config.target_column]

        elapsed = time.monotonic() - start
        return PhaseResult(
            phase_name="understand",
            success=True,
            duration_seconds=elapsed,
            metadata={"column_count": 0, "row_count": 0},
        )
