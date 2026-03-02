"""
Dataset Endpoints — Invisible Variables Engine.

Handles the full dataset lifecycle:

    POST   /datasets/                    — Upload CSV, ingest, profile, return metadata
    GET    /datasets/                    — Paginated list (optional name search)
    GET    /datasets/{dataset_id}        — Full dataset detail
    DELETE /datasets/{dataset_id}        — Delete dataset + stored file (204)
    GET    /datasets/{dataset_id}/profile — Detailed statistical profile

All endpoints require a valid ``X-API-Key`` header (enforced by middleware).
"""

from __future__ import annotations

import io
from uuid import UUID

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db, get_pagination
from ive.api.v1.schemas.dataset_schemas import (
    DatasetListResponse,
    DatasetProfileResponse,
    DatasetResponse,
    DeleteResponse,
)
from ive.config import get_settings
from ive.data.ingestion import DataIngestionService, DatasetValidationError
from ive.data.profiler import DataProfiler
from ive.db.models import Dataset
from ive.db.repositories.dataset_repo import DatasetRepository
from ive.storage.artifact_store import get_artifact_store
from ive.utils.logging import get_logger

log = get_logger(__name__)
router = APIRouter()

_ALLOWED_EXTENSIONS = {".csv"}
_MAX_BYTES = 500 * 1024 * 1024  # 500 MB


# ---------------------------------------------------------------------------
# POST /datasets/ — Upload dataset
# ---------------------------------------------------------------------------

@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=DatasetResponse,
    summary="Upload a CSV dataset",
    description=(
        "Upload a CSV file for analysis.  The file is validated, ingested, "
        "auto-profiled, and stored.  Returns full dataset metadata including "
        "detected column types and quality score."
    ),
)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file to upload"),
    target_column: str = Form(..., description="Name of the target / label column"),
    time_column: str | None = Form(None, description="Optional datetime column for temporal analysis"),
    name: str | None = Form(None, description="Optional display name (defaults to filename)"),
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Ingest a CSV dataset through the full ingestion + profiling pipeline.

    Steps:
        1. Validate file extension and size.
        2. ``DataIngestionService.ingest()`` — parse, detect types, validate,
           store file, create DB record.
        3. ``DataProfiler.profile()`` — compute statistical summary.
        4. Persist profile (quality_score, correlation matrix) into
           ``schema_json`` on the Dataset record.
        5. Return ``DatasetResponse``.
    """
    log.info("datasets.upload.start", filename=file.filename, target=target_column)

    # ── Extension check ───────────────────────────────────────────────
    filename: str = file.filename or "upload.csv"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only CSV files are accepted; received '{suffix}'.",
        )

    # ── Read bytes ────────────────────────────────────────────────────
    file_content = await file.read()
    if len(file_content) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )
    if len(file_content) > _MAX_BYTES:
        mb = len(file_content) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({mb:.0f} MB) exceeds the 500 MB limit.",
        )

    # ── Ingestion ─────────────────────────────────────────────────────
    try:
        svc = DataIngestionService()
        result = await svc.ingest(
            file_content=file_content,
            filename=name or filename,
            target_column=target_column,
            time_column=time_column,
            session=db,
        )
    except DatasetValidationError as exc:
        log.warning("datasets.upload.validation_error", errors=exc.errors)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": str(exc), "errors": exc.errors},
        )

    # ── Profiling (runs in-process on the already-parsed data) ────────
    try:
        # Re-parse for profiling (ingestion result holds column type info)
        import csv as _csv
        text_data = file_content.decode("utf-8", errors="replace")
        delimiter = ","
        try:
            dialect = _csv.Sniffer().sniff(text_data[:10240], delimiters=",;\t|")
            delimiter = dialect.delimiter
        except _csv.Error:
            pass
        df = pd.read_csv(io.StringIO(text_data), delimiter=delimiter, on_bad_lines="warn")

        profiler = DataProfiler()
        profile = profiler.profile(
            df=df,
            target_column=target_column,
            time_column=time_column,
            column_types=result.columns,
            dataset_id=result.dataset_id,
        )

        # Merge profile data into schema_json stored in the DB record
        repo = DatasetRepository(db, Dataset)
        existing_schema = result.schema_json.copy()
        existing_schema["quality_score"] = profile.quality_score
        existing_schema["quality_issues"] = [
            qi.model_dump() for qi in profile.quality_issues
        ]
        existing_schema["recommendations"] = profile.recommendations
        existing_schema["top_correlations"] = [
            cp.model_dump() for cp in profile.top_correlations
        ]
        await repo.update(
            id=_parse_uuid(result.dataset_id),
            schema_json=existing_schema,
        )
        result.schema_json.update(existing_schema)

    except Exception as exc:
        # Profiling failure is non-fatal — dataset is already saved
        log.warning("datasets.upload.profile_failed", error=str(exc))

    # ── Fetch updated ORM object and return ───────────────────────────
    repo = DatasetRepository(db, Dataset)
    dataset = await repo.get_by_id(_parse_uuid(result.dataset_id))
    if dataset is None:
        raise HTTPException(status_code=500, detail="Dataset record not found after creation.")

    log.info(
        "datasets.upload.done",
        dataset_id=result.dataset_id,
        rows=result.row_count,
        cols=result.col_count,
    )
    return DatasetResponse.from_dataset(dataset)


# ---------------------------------------------------------------------------
# GET /datasets/ — List datasets
# ---------------------------------------------------------------------------

@router.get(
    "/",
    response_model=DatasetListResponse,
    summary="List datasets",
    description="Return a paginated list of uploaded datasets. Optionally filter by name.",
)
async def list_datasets(
    search: str | None = Query(None, description="Case-insensitive name search"),
    pagination: dict = Depends(get_pagination),
    db: AsyncSession = Depends(get_db),
) -> DatasetListResponse:
    """Return all datasets with pagination and optional name search."""
    log.info("datasets.list", search=search, **pagination)
    repo = DatasetRepository(db, Dataset)
    skip, limit = pagination["skip"], pagination["limit"]

    if search:
        datasets = await repo.search_by_name(search, limit=limit)
        total = len(datasets)
    else:
        total = await repo.count()
        datasets = await repo.get_all(skip=skip, limit=limit)

    return DatasetListResponse(
        datasets=[DatasetResponse.from_dataset(ds) for ds in datasets],
        total=total,
        skip=skip,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id} — Dataset detail
# ---------------------------------------------------------------------------

@router.get(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Get dataset detail",
    description="Return full metadata for a single dataset including schema and column info.",
)
async def get_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DatasetResponse:
    """Retrieve a single dataset by ID."""
    log.info("datasets.get", dataset_id=str(dataset_id))
    repo = DatasetRepository(db, Dataset)
    dataset = await repo.get_by_id(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )
    return DatasetResponse.from_dataset(dataset)


# ---------------------------------------------------------------------------
# DELETE /datasets/{dataset_id} — Delete dataset
# ---------------------------------------------------------------------------

from fastapi import Response, status

@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a dataset",
    description=(
        "Delete a dataset record and its stored file from the artifact store. "
        "All associated experiments are also deleted (cascade)."
    ),
)
async def delete_dataset(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete dataset and its artifact file. Returns 204 on success."""
    
    log.info("datasets.delete", dataset_id=str(dataset_id))

    repo = DatasetRepository(db, Dataset)
    dataset = await repo.get_by_id(dataset_id)

    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    # Remove artifact file (non-fatal if already missing)
    try:
        store = get_artifact_store()
        await store.delete_file(dataset.file_path)
    except Exception as exc:
        log.warning("datasets.delete.artifact_error", error=str(exc))

    deleted = await repo.delete(dataset_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete dataset record.",
        )

    log.info("datasets.deleted", dataset_id=str(dataset_id))

    return Response(status_code=status.HTTP_204_NO_CONTENT)

# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}/profile — Statistical profile
# ---------------------------------------------------------------------------

@router.get(
    "/{dataset_id}/profile",
    response_model=DatasetProfileResponse,
    summary="Get dataset profile",
    description=(
        "Return the full statistical profile including column statistics, "
        "correlation matrix, quality issues, and recommendations."
    ),
)
async def get_dataset_profile(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DatasetProfileResponse:
    """Return the profile stored in ``schema_json`` for this dataset."""
    log.info("datasets.profile", dataset_id=str(dataset_id))
    repo = DatasetRepository(db, Dataset)
    dataset = await repo.get_by_id(dataset_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found.",
        )

    schema: dict = dataset.schema_json or {}

    if not schema.get("quality_score"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Profile for dataset '{dataset_id}' is not yet available. "
                "It may still be processing."
            ),
        )

    return DatasetProfileResponse(
        dataset_id=dataset_id,
        row_count=dataset.row_count,
        col_count=dataset.col_count,
        memory_usage_mb=schema.get("memory_usage_mb", 0.0),
        target_stats=schema.get("target", {}),
        column_profiles=schema.get("columns", []),
        quality_score=schema.get("quality_score", 0.0),
        quality_issues=schema.get("quality_issues", []),
        recommendations=schema.get("recommendations", []),
        top_correlations=schema.get("top_correlations", []),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_uuid(value: str | UUID) -> UUID:
    """Coerce a string or UUID to ``uuid.UUID``."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))
