"""
Experiments API Endpoints.

Create and manage analysis experiments. Starting an experiment queues a
Celery task that runs the four-phase IVE pipeline.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, HTTPException, Query, status

from ive.api.v1.schemas.experiment_schemas import (
    ExperimentCreateRequest,
    ExperimentCreateResponse,
    ExperimentDetailResponse,
    ExperimentListResponse,
)

log = structlog.get_logger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=ExperimentCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a new experiment",
)
async def create_experiment(body: ExperimentCreateRequest) -> ExperimentCreateResponse:
    """
    Queue a new IVE analysis experiment.

    Steps:
        1. Validate that the dataset exists and is in 'profiled' status
        2. Create the experiment record in the DB (status='queued')
        3. Enqueue workers.tasks.run_experiment.delay(experiment_id)
        4. Return 202 Accepted with experiment_id and task_id

    TODO:
        - Call DatasetRepo.get_by_id() to verify dataset exists
        - Call ExperimentRepo.create()
        - Call workers.tasks.run_experiment.apply_async()
    """
    experiment_id = uuid.uuid4()
    task_id = str(uuid.uuid4())  # TODO: real Celery task ID

    log.info(
        "ive.experiment.queued",
        experiment_id=str(experiment_id),
        dataset_id=str(body.dataset_id),
    )

    return ExperimentCreateResponse(
        id=experiment_id,
        dataset_id=body.dataset_id,
        name=body.name,
        status="queued",
        task_id=task_id,
    )


@router.get("", response_model=ExperimentListResponse, summary="List experiments")
async def list_experiments(
    dataset_id: Annotated[uuid.UUID | None, Query()] = None,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ExperimentListResponse:
    """
    List experiments with optional filters.

    TODO:
        - Call ExperimentRepo.list() with filters
        - Support filtering by dataset_id and status
    """
    return ExperimentListResponse(items=[], total=0, page=page, page_size=page_size)


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse, summary="Get experiment")
async def get_experiment(experiment_id: uuid.UUID) -> ExperimentDetailResponse:
    """
    Retrieve experiment details including phase progress.

    TODO:
        - Call ExperimentRepo.get_by_id()
        - Include phase breakdown and latest events
        - Raise 404 if not found
    """
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Experiment '{experiment_id}' not found",
    )


@router.delete(
    "/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Cancel/delete experiment"
)
async def delete_experiment(experiment_id: uuid.UUID) -> None:
    """
    Cancel a running experiment or delete a completed one.

    TODO:
        - If status=='running': revoke the Celery task
        - Call ExperimentRepo.delete()
        - Return 204 on success
    """
    log.info("ive.experiment.delete", experiment_id=str(experiment_id))
