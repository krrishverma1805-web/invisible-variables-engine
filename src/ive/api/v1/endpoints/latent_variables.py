"""
Latent Variable Endpoints — Invisible Variables Engine.

Routes (standalone — cross-experiment):
    GET  /latent-variables/                   — List all, paginated, filterable
    GET  /latent-variables/{variable_id}      — Single variable detail
    POST /latent-variables/apply              — Apply validated variables to new data

Experiment-scoped routes live in experiments.py:
    GET /experiments/{experiment_id}/latent-variables
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
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


# ---------------------------------------------------------------------------
# POST /latent-variables/apply  — Apply discovered variables to new data
# ---------------------------------------------------------------------------


@router.post(
    "/apply",
    summary="Apply discovered latent variables to a new dataset",
)
async def apply_latent_variables(
    file: UploadFile,
    experiment_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Apply discovered latent variables to a new dataset.

    Accepts a CSV file and an experiment_id. Retrieves all validated
    latent variables from that experiment, applies their construction
    rules to the new data, and returns the results.
    """
    import io

    import numpy as np
    import pandas as pd

    from ive.construction.variable_synthesizer import apply_construction_rule

    # Parse experiment_id to UUID
    try:
        exp_uuid = UUID(experiment_id)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_EXPERIMENT_ID",
                    "message": f"Invalid experiment_id: {experiment_id!r} is not a valid UUID.",
                }
            },
        )

    # Read the uploaded CSV
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "INVALID_CSV",
                    "message": f"Failed to parse CSV: {e}",
                }
            },
        )

    # Fetch validated latent variables for the experiment
    lv_repo = LatentVariableRepository(db, LatentVariable)
    lvs = await lv_repo.get_by_experiment(exp_uuid)
    validated_lvs = [lv for lv in lvs if lv.status == "validated"]

    if not validated_lvs:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code": "NO_VARIABLES",
                    "message": "No validated latent variables found for this experiment.",
                }
            },
        )

    # Apply each construction rule
    columns_added: list[dict[str, Any]] = []
    for lv in validated_lvs:
        rule = lv.construction_rule or {}
        pattern_type = rule.get("type", "subgroup")

        try:
            scores = apply_construction_rule(rule, pattern_type, df)
            col_name = lv.name or f"lv_{lv.id}"

            columns_added.append(
                {
                    "name": col_name,
                    "variable_id": str(lv.id),
                    "construction_rule": rule,
                    "scores": [round(float(s), 6) for s in scores],
                    "non_zero_count": int(np.sum(scores > 0)),
                    "mean_score": round(float(np.mean(scores)), 6),
                }
            )
        except Exception as e:
            log.warning(
                "ive.apply.variable_failed",
                variable_id=str(lv.id),
                error=str(e),
            )
            columns_added.append(
                {
                    "name": lv.name or f"lv_{lv.id}",
                    "variable_id": str(lv.id),
                    "error": str(e),
                    "scores": [],
                }
            )

    return JSONResponse(
        status_code=200,
        content={
            "experiment_id": experiment_id,
            "n_rows": len(df),
            "n_variables_applied": sum(1 for c in columns_added if "error" not in c),
            "n_variables_failed": sum(1 for c in columns_added if "error" in c),
            "columns_added": columns_added,
        },
    )
