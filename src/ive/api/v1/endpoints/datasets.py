"""
Datasets API Endpoints.

CRUD operations for dataset management. Supports CSV and Parquet file uploads.
After upload, a background profiling task is automatically enqueued.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from ive.api.v1.dependencies import get_db_session
from ive.api.v1.schemas.dataset_schemas import (
    DatasetCreateResponse,
    DatasetDetailResponse,
    DatasetListResponse,
)

log = structlog.get_logger(__name__)

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/octet-stream",
    "application/vnd.apache.parquet",
}
_MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB


@router.post(
    "",
    response_model=DatasetCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a dataset",
)
async def upload_dataset(
    file: Annotated[UploadFile, File(description="CSV or Parquet file to upload")],
    name: Annotated[str, Form(description="Human-readable dataset name")],
    target_column: Annotated[str, Form(description="Name of the target/label column")],
    description: Annotated[str | None, Form(description="Optional description")] = None,
    # db: Annotated[AsyncSession, Depends(get_db_session)] = ...,  # TODO: enable
) -> DatasetCreateResponse:
    """
    Upload a new dataset.

    Steps:
        1. Validate file type and size
        2. Save file to the artifact store
        3. Create dataset record in the database
        4. Enqueue a profiling Celery task
        5. Return 201 with dataset metadata

    TODO:
        - Validate file content type
        - Stream file to ArtifactStore.save()
        - Call DatasetRepo.create() with DB session
        - Enqueue workers.tasks.profile_dataset.delay(dataset_id)
    """
    log.info("ive.dataset.uploading", name=name, filename=file.filename)

    # TODO: Validate content type
    # if file.content_type not in _ALLOWED_CONTENT_TYPES:
    #     raise HTTPException(status_code=415, detail="Unsupported file type")

    # TODO: Validate file size
    # content = await file.read()
    # if len(content) > _MAX_FILE_SIZE_BYTES:
    #     raise HTTPException(status_code=413, detail="File too large (max 500MB)")

    dataset_id = uuid.uuid4()

    # TODO: Save to artifact store and DB
    # file_path = await artifact_store.save(dataset_id, content, file.filename)
    # dataset = await dataset_repo.create(db, {...})

    return DatasetCreateResponse(
        id=dataset_id,
        name=name,
        target_column=target_column,
        description=description,
        row_count=0,
        column_count=0,
        status="uploaded",
    )


@router.get("", response_model=DatasetListResponse, summary="List datasets")
async def list_datasets(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    search: Annotated[str | None, Query()] = None,
) -> DatasetListResponse:
    """
    List all datasets with pagination.

    TODO:
        - Call DatasetRepo.list(db, offset=(page-1)*page_size, limit=page_size, search=search)
        - Return paginated response
    """
    # TODO: Replace with real DB query
    return DatasetListResponse(items=[], total=0, page=page, page_size=page_size)


@router.get("/{dataset_id}", response_model=DatasetDetailResponse, summary="Get dataset details")
async def get_dataset(dataset_id: uuid.UUID) -> DatasetDetailResponse:
    """
    Retrieve a single dataset by ID.

    TODO:
        - Call DatasetRepo.get_by_id(db, dataset_id)
        - Raise 404 if not found
        - Include profile summary if available
    """
    # TODO: Replace with real DB query
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Dataset '{dataset_id}' not found",
    )


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete dataset")
async def delete_dataset(dataset_id: uuid.UUID) -> None:
    """
    Delete a dataset and all associated experiments.

    TODO:
        - Call DatasetRepo.delete(db, dataset_id)
        - Delete artifact files from storage
        - Return 204 on success, 404 if not found
    """
    # TODO: Implement deletion
    log.info("ive.dataset.delete", dataset_id=str(dataset_id))
