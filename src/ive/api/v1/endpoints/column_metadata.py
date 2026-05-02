"""Per-dataset column sensitivity endpoints.

These power the upload-page sensitivity editor (per plan §174 / §202):
    - View current sensitivity per column (read scope)
    - Bulk-edit multiple columns in one request (write scope)

Routes (mounted at ``/api/v1/datasets/{dataset_id}/columns``):

    GET  /                    list metadata for one dataset's columns
    PUT  /                    bulk update sensitivity for N columns
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db
from ive.api.v1.schemas.column_metadata_schemas import (
    BulkSensitivityUpdate,
    ColumnMetadataListResponse,
    ColumnMetadataResponse,
)
from ive.auth.scopes import AuthContext, Scope, require_scope
from ive.db.repositories.dataset_column_metadata_repo import (
    DatasetColumnMetadataRepo,
)

router = APIRouter()


@router.get(
    "/",
    response_model=ColumnMetadataListResponse,
    summary="List column sensitivity metadata for a dataset.",
)
async def list_columns(
    dataset_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.READ)),
) -> ColumnMetadataListResponse:
    repo = DatasetColumnMetadataRepo(session)
    rows = await repo.list_for_dataset(dataset_id)
    items = [ColumnMetadataResponse.model_validate(r) for r in rows]
    public_count = sum(1 for r in rows if r.sensitivity == "public")
    return ColumnMetadataListResponse(
        items=items,
        total=len(items),
        public_count=public_count,
    )


@router.put(
    "/",
    response_model=ColumnMetadataListResponse,
    summary="Bulk-update column sensitivity for a dataset.",
)
async def update_columns(
    dataset_id: uuid.UUID,
    payload: BulkSensitivityUpdate,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.WRITE)),
) -> ColumnMetadataListResponse:
    repo = DatasetColumnMetadataRepo(session)
    update_map = {u.column_name: u.sensitivity for u in payload.updates}
    changed = await repo.bulk_set(dataset_id, update_map)
    if not changed and not await repo.list_for_dataset(dataset_id):
        # Dataset has no metadata at all → likely a missing dataset_id.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No column metadata found for dataset {dataset_id}.",
        )
    rows = await repo.list_for_dataset(dataset_id)
    return ColumnMetadataListResponse(
        items=[ColumnMetadataResponse.model_validate(r) for r in rows],
        total=len(rows),
        public_count=sum(1 for r in rows if r.sensitivity == "public"),
    )
