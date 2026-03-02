"""
Latent Variable Endpoints — Invisible Variables Engine.

STUB — Phase 2.

All endpoints return HTTP 501 (Not Implemented) until the construct phase
of the IVE pipeline is completed in Phase 2.

Routes (nested under /experiments/{id}/latent-variables):
    GET /experiments/{experiment_id}/latent-variables/
    GET /experiments/{experiment_id}/latent-variables/{variable_id}
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db, get_pagination
from ive.api.v1.schemas.latent_variable_schemas import (
    LatentVariableDetail,
    LatentVariableListResponse,
)
from ive.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_NOT_IMPLEMENTED = "Latent variables are not yet available — coming in Phase 2."


@router.get(
    "/{experiment_id}/latent-variables",
    response_model=LatentVariableListResponse,
    summary="List latent variables for an experiment (Phase 2)",
    description=(
        "Return all latent variables discovered by the IVE construct phase "
        "for the given experiment.  Optionally filter by status."
    ),
)
async def list_latent_variables(
    experiment_id: UUID,
    lv_status: str | None = Query(
        None,
        alias="status",
        description="Filter by status: candidate | validated | rejected",
    ),
    pagination: dict = Depends(get_pagination),
    db: AsyncSession = Depends(get_db),
) -> LatentVariableListResponse:
    """[STUB] List latent variables for an experiment."""
    log.info("latent_variables.list.stub", experiment_id=str(experiment_id))
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get(
    "/{experiment_id}/latent-variables/{variable_id}",
    response_model=LatentVariableDetail,
    summary="Get latent variable detail (Phase 2)",
    description="Return the full detail of a single latent variable, including construction rule and bootstrap statistics.",
)
async def get_latent_variable(
    experiment_id: UUID,
    variable_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> LatentVariableDetail:
    """[STUB] Return full detail for a single latent variable."""
    log.info(
        "latent_variables.get.stub",
        experiment_id=str(experiment_id),
        variable_id=str(variable_id),
    )
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)
