"""
Latent Variables API Endpoints.

Read-only endpoints for accessing discovered latent variables.
Latent variables are created by the worker pipeline — they cannot be
created directly via the API.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, HTTPException, status

from ive.api.v1.schemas.latent_variable_schemas import (
    LatentVariableDetailResponse,
    LatentVariableExplanationResponse,
    LatentVariableListResponse,
)

log = structlog.get_logger(__name__)

router = APIRouter()


@router.get(
    "/experiments/{experiment_id}/latent-variables",
    response_model=LatentVariableListResponse,
    summary="List latent variables for an experiment",
)
async def list_latent_variables(experiment_id: uuid.UUID) -> LatentVariableListResponse:
    """
    Return all discovered latent variables for a completed experiment.

    Results are ordered by rank (highest confidence first).

    TODO:
        - Verify experiment exists and status == 'completed'
        - Call LatentVariableRepo.list_by_experiment(experiment_id)
    """
    # TODO: Replace with real DB query
    return LatentVariableListResponse(items=[], experiment_id=experiment_id)


@router.get(
    "/{lv_id}",
    response_model=LatentVariableDetailResponse,
    summary="Get latent variable details",
)
async def get_latent_variable(lv_id: uuid.UUID) -> LatentVariableDetailResponse:
    """
    Get full details for a single latent variable.

    Includes: candidate features, validation results, SHAP importance.

    TODO:
        - Call LatentVariableRepo.get_by_id(lv_id)
        - Raise 404 if not found
    """
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Latent variable '{lv_id}' not found",
    )


@router.get(
    "/{lv_id}/explanation",
    response_model=LatentVariableExplanationResponse,
    summary="Get natural language explanation",
)
async def get_explanation(lv_id: uuid.UUID) -> LatentVariableExplanationResponse:
    """
    Return the natural language explanation for a latent variable.

    TODO:
        - Call LatentVariableRepo.get_by_id(lv_id)
        - Return the explanation field and supporting evidence
        - Raise 404 if not found
    """
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Latent variable '{lv_id}' not found",
    )
