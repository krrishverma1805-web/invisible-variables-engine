"""
Experiment Endpoints — Invisible Variables Engine.

STUB — Phase 2.

All endpoints return HTTP 501 (Not Implemented) until the experiment
pipeline (Celery tasks, IVE engine phases) is wired in Phase 2.

Routes:
    POST   /experiments/                          — Create experiment
    GET    /experiments/                          — List experiments
    GET    /experiments/{experiment_id}           — Experiment detail
    GET    /experiments/{experiment_id}/progress  — Progress poll
    POST   /experiments/{experiment_id}/cancel    — Cancel
    DELETE /experiments/{experiment_id}           — Delete
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db, get_pagination
from ive.api.v1.schemas.experiment_schemas import (
    ExperimentCreateRequest,
    ExperimentCreateResponse,
    ExperimentListResponse,
    ExperimentProgressResponse,
    ExperimentResponse,
)
from ive.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_NOT_IMPLEMENTED = "Experiments are not yet implemented — coming in Phase 2."


@router.post(
    "/",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ExperimentCreateResponse,
    summary="Create an experiment (Phase 2)",
)
async def create_experiment(
    request: ExperimentCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> ExperimentCreateResponse:
    """[STUB] Queue an experiment for the IVE pipeline."""
    log.info("experiments.create.stub", dataset_id=str(request.dataset_id))
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get(
    "/",
    response_model=ExperimentListResponse,
    summary="List experiments (Phase 2)",
)
async def list_experiments(
    dataset_id: UUID | None = Query(None, description="Filter by dataset"),
    experiment_status: str | None = Query(None, alias="status"),
    pagination: dict = Depends(get_pagination),
    db: AsyncSession = Depends(get_db),
) -> ExperimentListResponse:
    """[STUB] Return paginated list of experiments."""
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment detail (Phase 2)",
)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """[STUB] Return full experiment detail."""
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.get(
    "/{experiment_id}/progress",
    response_model=ExperimentProgressResponse,
    summary="Poll experiment progress (Phase 2)",
)
async def get_experiment_progress(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentProgressResponse:
    """[STUB] Return lightweight progress for the polling loop."""
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


@router.post(
    "/{experiment_id}/cancel",
    response_model=ExperimentResponse,
    summary="Cancel an experiment (Phase 2)",
)
async def cancel_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """[STUB] Revoke the Celery task and mark the experiment as cancelled."""
    raise HTTPException(status_code=501, detail=_NOT_IMPLEMENTED)


from fastapi import Response, status

@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete experiment. Returns 204 on success."""
    
    repo = ExperimentRepository(db)
    experiment = await repo.get_by_id(experiment_id)

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Experiment not found",
        )

    await repo.delete(experiment_id)

    return Response(status_code=status.HTTP_204_NO_CONTENT)
