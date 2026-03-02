"""
Integration test — End-to-End Pipeline.

Tests the full four-phase IVE pipeline using synthetic datasets
to verify that each phase receives and produces correct data.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
import pytest_asyncio

from tests.fixtures.synthetic_datasets import make_regression_with_latent


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_context_populated_after_phases(tmp_path) -> None:
    """
    Smoke test: running all four phases should not raise and should
    populate PipelineContext latent_variables after the engine runs.

    TODO (once implementations are complete):
        - Create a real CSV from make_regression_with_latent()
        - Build an ExperimentConfig
        - Instantiate IVEEngine and await engine.run(...)
        - Assert len(result.latent_variables) >= 0
        - Assert result.elapsed_seconds > 0
    """
    from ive.core.engine import IVEEngine
    from ive.api.v1.schemas.experiment_schemas import ExperimentConfig

    pytest.skip("Phases not yet implemented — skipping until core logic added")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_understand_phase_populates_context(tmp_path) -> None:
    """
    Phase 1 (Understand) should load the dataset and populate ctx.df.

    TODO:
        - Write CSV to tmp_path
        - Build PipelineContext
        - Instantiate PhaseUnderstand and call execute(ctx)
        - Assert ctx.df is not None
        - Assert ctx.column_types is not empty
    """
    pytest.skip("Phase understand not yet implemented — skipping")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_phase_produces_residuals(tmp_path) -> None:
    """
    Phase 2 (Model) should populate ctx.residuals after execution.

    TODO:
        - Run Phase 1 first to populate ctx.df
        - Run Phase 2
        - Assert ctx.residuals.shape == (n_samples,)
    """
    pytest.skip("Phase model not yet implemented — skipping")
