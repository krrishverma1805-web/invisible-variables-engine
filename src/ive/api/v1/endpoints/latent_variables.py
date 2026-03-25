"""
Latent Variable Endpoints — Invisible Variables Engine.

Routes (standalone — cross-experiment):
    GET /latent-variables/                    — List all, paginated, filterable
    GET /latent-variables/{variable_id}       — Single variable detail

Experiment-scoped routes live in experiments.py:
    GET /experiments/{experiment_id}/latent-variables
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db, get_pagination
from ive.api.v1.schemas.latent_variable_schemas import (
    LatentVariableListResponse,
    LatentVariableResponse,
)
from ive.db.models import LatentVariable
from ive.db.repositories.latent_variable_repo import LatentVariableRepository
from ive.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# GET /latent-variables/  — Cross-experiment list
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=LatentVariableListResponse,
    summary="List latent variables (all experiments)",
)
async def list_latent_variables(
    lv_status: str | None = Query(
        None,
        alias="status",
        description="Filter by status: candidate | validated | rejected",
    ),
    experiment_id: UUID | None = Query(None, description="Filter by experiment UUID"),
    pagination: dict[str, Any] = Depends(get_pagination),
    db: AsyncSession = Depends(get_db),
) -> LatentVariableListResponse:
    """Return a paginated list of latent variables across all experiments.

    Supports optional filtering by ``status`` and ``experiment_id``.
    """
    stmt = select(LatentVariable).order_by(LatentVariable.importance_score.desc())

    if lv_status is not None:
        stmt = stmt.where(LatentVariable.status == lv_status)
    if experiment_id is not None:
        stmt = stmt.where(LatentVariable.experiment_id == experiment_id)

    # Count total before pagination
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    skip = pagination["skip"]
    limit = pagination["limit"]
    paged = await db.execute(stmt.offset(skip).limit(limit))
    rows = list(paged.scalars().all())

    return LatentVariableListResponse(
        variables=[LatentVariableResponse.model_validate(r) for r in rows],
        total=total,
        skip=skip,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# GET /latent-variables/{variable_id}  — Single variable detail
# ---------------------------------------------------------------------------


@router.get(
    "/{variable_id}",
    response_model=LatentVariableResponse,
    summary="Get latent variable detail",
)
async def get_latent_variable(
    variable_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> LatentVariableResponse:
    """Return full detail for a single latent variable. Returns 404 if not found."""
    lv_repo = LatentVariableRepository(db, LatentVariable)
    variable = await lv_repo.get_by_id(variable_id)
    if variable is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Latent variable {variable_id} not found.",
        )
    return LatentVariableResponse.model_validate(variable)
