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

    GET    /experiments/{experiment_id}/report                  — Full JSON report
    GET    /experiments/{experiment_id}/summary                 — Compact summary JSON
    GET    /experiments/{experiment_id}/patterns/export         — Patterns CSV download
    GET    /experiments/{experiment_id}/latent-variables/export — LVs CSV download
"""

from __future__ import annotations

import io
import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db, get_pagination
from ive.api.v1.schemas.experiment_schemas import (
    ErrorPatternResponse,
    ExperimentCreate,
    ExperimentCreateResponse,
    ExperimentEventResponse,
    ExperimentEventsListResponse,
    ExperimentListResponse,
    ExperimentProgressResponse,
    ExperimentResponse,
    ExperimentSummaryResponse,
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
from ive.utils.reporting import (
    build_full_report,
    latent_variables_to_csv,
    patterns_to_csv,
)

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
# GET /experiments/{experiment_id}/events  — Execution log
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/events",
    response_model=ExperimentEventsListResponse,
    summary="Get the execution event log for an experiment",
)
async def get_experiment_events(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentEventsListResponse:
    """Return the chronological audit log of pipeline lifecycle events.

    Each event records a discrete milestone (e.g. ``dataset_loaded``,
    ``modeling_completed``) along with a human-readable message and
    optional metadata captured at the time the event occurred.

    Events are sorted oldest-first so they read like a sequential log.
    If the experiment has not yet produced any events (e.g. it is still
    queued), an empty list is returned.
    """
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    events = await exp_repo.get_events(experiment_id)
    return ExperimentEventsListResponse(
        experiment_id=experiment_id,
        events=[ExperimentEventResponse.model_validate(e) for e in events],
        total=len(events),
    )


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


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/summary  — Compact summary JSON
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/summary",
    response_model=ExperimentSummaryResponse,
    summary="Get compact experiment summary",
    tags=["Reporting"],
)
async def get_experiment_summary(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ExperimentSummaryResponse:
    """Return headline, counts, top findings and recommendations for a completed experiment.

    The summary is reconstructed live from current DB state using
    :class:`ExplanationGenerator` so it is always fresh and does not
    require a dedicated summary table.

    Returns 404 if the experiment does not exist.
    """
    from ive.construction.explanation_generator import ExplanationGenerator

    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    # Fetch patterns and latent variables
    patterns = await exp_repo.get_error_patterns(experiment_id)
    patterns_dicts = [_orm_to_dict(p) for p in patterns]

    lv_repo = LatentVariableRepository(db, LatentVariable)
    lvs = await lv_repo.get_by_experiment(experiment_id)
    lv_dicts = [_orm_to_dict(v) for v in lvs]

    # Reconstruct summary live
    dataset_name = str(experiment_id)[:8]
    target_column = ""
    ds_repo = DatasetRepository(db, Dataset)
    dataset = await ds_repo.get_by_id(experiment.dataset_id)
    if dataset is not None:
        dataset_name = getattr(dataset, "name", dataset_name)
        target_column = getattr(dataset, "target_column", "")

    explainer = ExplanationGenerator()
    summary = explainer.generate_experiment_summary(
        patterns=patterns_dicts,
        candidates=lv_dicts,
        dataset_name=dataset_name,
        target_column=target_column,
    )

    log.info("experiments.summary", experiment_id=str(experiment_id))
    return ExperimentSummaryResponse(**summary)


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/report  — Full JSON report
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/report",
    summary="Download full experiment report as JSON",
    tags=["Reporting"],
)
async def get_experiment_report(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Return the complete experiment report as a downloadable JSON file.

    The response includes:

    * ``experiment`` — full experiment metadata
    * ``dataset``    — dataset metadata
    * ``patterns``   — all detected error patterns
    * ``latent_variables`` — all latent variable candidates
    * ``summary``    — executive summary from ExplanationGenerator

    Content-Type is ``application/json`` with a ``Content-Disposition``
    header prompting a file download.

    Returns 404 if the experiment does not exist.
    """
    from ive.construction.explanation_generator import ExplanationGenerator

    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    # Serialise experiment
    exp_dict = ExperimentResponse.model_validate(experiment).model_dump(mode="json")

    # Fetch and serialise dataset
    dataset_dict: dict = {}
    ds_repo = DatasetRepository(db, Dataset)
    dataset = await ds_repo.get_by_id(experiment.dataset_id)
    if dataset is not None:
        dataset_dict = {
            "id": str(dataset.id),
            "name": getattr(dataset, "name", ""),
            "target_column": getattr(dataset, "target_column", ""),
            "row_count": getattr(dataset, "row_count", None),
            "col_count": getattr(dataset, "col_count", None),
        }

    # Fetch patterns and latent variables
    patterns = await exp_repo.get_error_patterns(experiment_id)
    patterns_dicts = [_orm_to_dict(p) for p in patterns]

    lv_repo = LatentVariableRepository(db, LatentVariable)
    lvs = await lv_repo.get_by_experiment(experiment_id)
    lv_dicts = [_orm_to_dict(v) for v in lvs]

    # Build summary
    dataset_name = dataset_dict.get("name") or str(experiment_id)[:8]
    target_column = dataset_dict.get("target_column", "")
    explainer = ExplanationGenerator()
    summary = explainer.generate_experiment_summary(
        patterns=patterns_dicts,
        candidates=lv_dicts,
        dataset_name=dataset_name,
        target_column=target_column,
    )

    report = build_full_report(
        experiment=exp_dict,
        dataset=dataset_dict,
        patterns=patterns_dicts,
        latent_variables=lv_dicts,
        summary=summary,
    )

    filename = f"experiment_{str(experiment_id)[:8]}_report.json"
    log.info("experiments.report", experiment_id=str(experiment_id))
    return Response(
        content=json.dumps(report, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/patterns/export  — Patterns CSV
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/patterns/export",
    summary="Download patterns as CSV",
    tags=["Reporting"],
)
async def export_patterns_csv(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream a CSV download of all error patterns for the given experiment.

    Columns: ``pattern_type``, ``column_name``, ``effect_size``,
    ``p_value``, ``adjusted_p_value``, ``sample_count``,
    ``mean_residual``, ``std_residual``.

    Returns an empty CSV (header only) when no patterns exist.
    Returns 404 if the experiment does not exist.
    """
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    patterns = await exp_repo.get_error_patterns(experiment_id)
    patterns_dicts = [_orm_to_dict(p) for p in patterns]

    csv_str = patterns_to_csv(patterns_dicts)
    filename = f"experiment_{str(experiment_id)[:8]}_patterns.csv"

    log.info(
        "experiments.export_patterns",
        experiment_id=str(experiment_id),
        n_patterns=len(patterns_dicts),
    )
    return StreamingResponse(
        io.StringIO(csv_str),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /experiments/{experiment_id}/latent-variables/export  — LVs CSV
# ---------------------------------------------------------------------------


@router.get(
    "/{experiment_id}/latent-variables/export",
    summary="Download latent variables as CSV",
    tags=["Reporting"],
)
async def export_latent_variables_csv(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream a CSV download of all latent variables for the given experiment.

    Columns: ``name``, ``status``, ``stability_score``,
    ``bootstrap_presence_rate``, ``importance_score``,
    ``description``, ``explanation_text``.

    Returns an empty CSV (header only) when no latent variables exist.
    Returns 404 if the experiment does not exist.
    """
    exp_repo = ExperimentRepository(db, Experiment)
    experiment = await exp_repo.get_by_id(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found.",
        )

    lv_repo = LatentVariableRepository(db, LatentVariable)
    lvs = await lv_repo.get_by_experiment(experiment_id)
    lv_dicts = [_orm_to_dict(v) for v in lvs]

    csv_str = latent_variables_to_csv(lv_dicts)
    filename = f"experiment_{str(experiment_id)[:8]}_latent_variables.csv"

    log.info(
        "experiments.export_latent_variables",
        experiment_id=str(experiment_id),
        n_variables=len(lv_dicts),
    )
    return StreamingResponse(
        io.StringIO(csv_str),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _orm_to_dict(obj: object) -> dict:
    """Serialise a SQLAlchemy ORM instance to a plain dict.

    Uses ``__dict__`` and strips the SQLAlchemy internal ``_sa_instance_state``
    key.  UUID and datetime fields are converted to strings so the result is
    always JSON-serialisable.

    Args:
        obj: SQLAlchemy ORM model instance.

    Returns:
        Plain dict with string-serialised UUID/datetime values.
    """
    import datetime
    from uuid import UUID as _UUID

    raw = {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    out: dict = {}
    for k, v in raw.items():
        if isinstance(v, _UUID):
            out[k] = str(v)
        elif isinstance(v, datetime.datetime):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out
