"""
Experiment Endpoints — Invisible Variables Engine.

Routes:
    POST   /experiments/                                — Create & queue experiment
    GET    /experiments/                                — List experiments (paginated)
    GET    /experiments/{experiment_id}                 — Full experiment detail
    GET    /experiments/{experiment_id}/progress        — Lightweight progress poll
    GET    /experiments/{experiment_id}/patterns        — Error patterns discovered
    GET    /experiments/{experiment_id}/latent-variables — Latent variables
    POST   /experiments/{experiment_id}/cancel          — Cancel & revoke task
    DELETE /experiments/{experiment_id}                 — Hard delete
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db, get_pagination
from ive.api.v1.schemas.experiment_schemas import (
    ErrorPatternResponse,
    ExperimentCreate,
    ExperimentCreateResponse,
    ExperimentListResponse,
    ExperimentProgressResponse,
    ExperimentResponse,
)
from ive.api.v1.schemas.latent_variable_schemas import (
    LatentVariableListResponse,
    LatentVariableResponse,
)
from ive.db.models import Dataset, Experiment, LatentVariable
from ive.db.repositories.dataset_repo import DatasetRepository
from ive.db.repositories.experiment_repo import ExperimentRepository
from ive.db.repositories.latent_variable_repo import LatentVariableRepository
from ive.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /experiments/  — Create and queue
# ---------------------------------------------------------------------------


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=ExperimentCreateResponse,
    summary="Create and queue an experiment",
)
async def create_experiment(
    request: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
) -> ExperimentCreateResponse:
    """Queue a new IVE experiment for the given dataset.

    Validates the dataset exists, creates an ``Experiment`` DB row with
    ``status="queued"``, dispatches the Celery task, and saves the
    ``celery_task_id`` back to the row.

    Returns 404 if the dataset does not exist.
    """
    # Validate dataset exists
    ds_repo = DatasetRepository(db, Dataset)
    dataset = await ds_repo.get_by_id(request.dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset {request.dataset_id} not found.",
        )

    # Create experiment row
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.create(
        dataset_id=request.dataset_id,
        config_json=request.config,
        status="queued",
        progress_pct=0,
    )
    await db.flush()

    # Dispatch Celery task
    from ive.workers.tasks import run_experiment as celery_run

    task = celery_run.delay(str(experiment.id), request.config)

    # Persist task id
    await exp_repo.update(experiment.id, celery_task_id=task.id)

    log.info(
        "experiments.create",
        experiment_id=str(experiment.id),
        dataset_id=str(request.dataset_id),
        celery_task_id=task.id,
    )

    return ExperimentCreateResponse(
        id=experiment.id,
        status=experiment.status,
        celery_task_id=task.id,
        message="Experiment queued successfully.",
    )


# ---------------------------------------------------------------------------
# GET /experiments/  — List
# ---------------------------------------------------------------------------


@router.get(
    "/",
    response_model=ExperimentListResponse,
    summary="List experiments (paginated)",
)
async def list_experiments(
    dataset_id: UUID | None = Query(None, description="Filter by dataset UUID"),
    experiment_status: str | None = Query(None, alias="status", description="Filter by status"),
    pagination: dict = Depends(get_pagination),
    db: AsyncSession = Depends(get_db),
) -> ExperimentListResponse:
    """Return a paginated list of experiments with optional filters."""
    stmt = select(Experiment).order_by(Experiment.created_at.desc())

    if dataset_id is not None:
        stmt = stmt.where(Experiment.dataset_id == dataset_id)
    if experiment_status is not None:
        stmt = stmt.where(Experiment.status == experiment_status)

    # Count total before pagination
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total: int = (await db.execute(count_stmt)).scalar_one()

    skip = pagination["skip"]
    limit = pagination["limit"]
    paged = await db.execute(stmt.offset(skip).limit(limit))
    rows = list(paged.scalars().all())

    return ExperimentListResponse(
        experiments=[ExperimentResponse.model_validate(r) for r in rows],
        total=total,
        skip=skip,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}  — Detail
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment detail",
)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Return full detail for a single experiment. Returns 404 if not found."""
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )
    return ExperimentResponse.model_validate(experiment)


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/progress  — Progress poll
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/progress",
    response_model=ExperimentProgressResponse,
    summary="Poll experiment progress",
)
async def get_experiment_progress(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentProgressResponse:
    """Lightweight progress endpoint for polling / WebSocket clients."""
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )
    return ExperimentProgressResponse.model_validate(experiment)


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/patterns  — Error patterns
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/patterns",
    response_model=list[ErrorPatternResponse],
    summary="Get error patterns for an experiment",
)
async def get_experiment_patterns(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[ErrorPatternResponse]:
    """Return all error patterns discovered for the given experiment."""
    # Verify experiment exists
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    patterns = await exp_repo.get_error_patterns(experiment_id)
    return [ErrorPatternResponse.model_validate(p) for p in patterns]


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/latent-variables  — Latent variables
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/latent-variables",
    response_model=LatentVariableListResponse,
    summary="List latent variables for an experiment",
)
async def get_experiment_latent_variables(
    experiment_id: UUID,
    lv_status: str | None = Query(
        None,
        alias="status",
        description="Filter by status: candidate | validated | rejected",
    ),
    pagination: dict = Depends(get_pagination),
    db: AsyncSession = Depends(get_db),
) -> LatentVariableListResponse:
    """Return all latent variables discovered for the given experiment."""
    # Verify experiment exists
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    lv_repo = LatentVariableRepository(db, LatentVariable)
    all_vars = await lv_repo.get_by_experiment(experiment_id, status=lv_status)

    skip = pagination["skip"]
    limit = pagination["limit"]
    paged = all_vars[skip : skip + limit]

    return LatentVariableListResponse(
        variables=[LatentVariableResponse.model_validate(v) for v in paged],
        total=len(all_vars),
        skip=skip,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# POST /experiments/{experiment_id}/cancel  — Cancel
# ---------------------------------------------------------------------------


@router.post(
    "/{experiment_id}/cancel",
    response_model=ExperimentResponse,
    summary="Cancel a running experiment",
)
async def cancel_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Revoke the Celery task and mark the experiment as cancelled."""
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    if experiment.status not in ("queued", "running"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel experiment with status '{experiment.status}'.",
        )

    # Revoke Celery task if we have the task id
    if experiment.celery_task_id:
        from ive.workers.tasks import cancel_experiment as celery_cancel

        celery_cancel.delay(
            task_id=experiment.celery_task_id,
            experiment_id=str(experiment_id),
        )

    updated = await exp_repo.mark_cancelled(experiment_id)
    log.info("experiments.cancel", experiment_id=str(experiment_id))
    return ExperimentResponse.model_validate(updated)


# ---------------------------------------------------------------------------
# DELETE /experiments/{experiment_id}  — Hard delete
# ---------------------------------------------------------------------------


@router.delete(
    "/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an experiment",
)
async def delete_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Permanently delete an experiment and all its child records (cascade)."""
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    await exp_repo.delete(experiment_id)
    log.info("experiments.delete", experiment_id=str(experiment_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
