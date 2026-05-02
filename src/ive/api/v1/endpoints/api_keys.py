"""Admin endpoints for API key management.

All routes require the ``admin`` scope (per plan §155 / §121). Raw key
values are returned **only** in the create / rotate responses; the DB
stores SHA-256 hashes.

Routes
------
    POST    /api-keys              Create a new key. Returns the raw value once.
    GET     /api-keys              List keys (metadata only, no raw values).
    GET     /api-keys/{id}         Detail for one key.
    POST    /api-keys/{id}/rotate  Rotate the key (new raw value, same metadata).
    DELETE  /api-keys/{id}         Revoke (mark inactive). Idempotent.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db
from ive.api.v1.schemas.api_key_schemas import (
    APIKeyCreatedResponse,
    APIKeyCreateRequest,
    APIKeyListResponse,
    APIKeyResponse,
)
from ive.auth.scopes import AuthContext, Scope, require_scope
from ive.db.repositories.api_key_repo import APIKeyRepo

router = APIRouter()


@router.post(
    "/",
    response_model=APIKeyCreatedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new API key (admin only). Raw key shown once.",
)
async def create_api_key(
    payload: APIKeyCreateRequest,
    session: AsyncSession = Depends(get_db),
    actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> APIKeyCreatedResponse:
    repo = APIKeyRepo(session)
    if await repo.get_by_name(payload.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"API key named {payload.name!r} already exists.",
        )
    row, raw = await repo.create(
        name=payload.name,
        scopes=list(payload.scopes),
        rate_limit=payload.rate_limit,
        expires_at=payload.expires_at,
        created_by=actor.api_key_name,
    )
    return APIKeyCreatedResponse(
        id=row.id,
        name=row.name,
        scopes=list(row.scopes),
        rate_limit=row.rate_limit,
        is_active=row.is_active,
        created_at=row.created_at,
        created_by=row.created_by,
        expires_at=row.expires_at,
        last_used_at=row.last_used_at,
        last_rotated_at=row.last_rotated_at,
        raw_key=raw,
    )


@router.get(
    "/",
    response_model=APIKeyListResponse,
    summary="List API keys (admin only).",
)
async def list_api_keys(
    include_inactive: bool = False,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> APIKeyListResponse:
    repo = APIKeyRepo(session)
    rows = await repo.list_all(include_inactive=include_inactive)
    return APIKeyListResponse(
        items=[APIKeyResponse.model_validate(r) for r in rows],
        total=len(rows),
    )


@router.get(
    "/{key_id}",
    response_model=APIKeyResponse,
    summary="API key detail (admin only).",
)
async def get_api_key(
    key_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> APIKeyResponse:
    repo = APIKeyRepo(session)
    row = await repo.get_by_id(key_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found.",
        )
    return APIKeyResponse.model_validate(row)


@router.post(
    "/{key_id}/rotate",
    response_model=APIKeyCreatedResponse,
    summary="Rotate an API key (admin only). Raw value shown once.",
)
async def rotate_api_key(
    key_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> APIKeyCreatedResponse:
    repo = APIKeyRepo(session)
    out = await repo.rotate(key_id)
    if out is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found.",
        )
    row, raw = out
    return APIKeyCreatedResponse(
        id=row.id,
        name=row.name,
        scopes=list(row.scopes),
        rate_limit=row.rate_limit,
        is_active=row.is_active,
        created_at=row.created_at,
        created_by=row.created_by,
        expires_at=row.expires_at,
        last_used_at=row.last_used_at,
        last_rotated_at=row.last_rotated_at,
        raw_key=raw,
    )


@router.delete(
    "/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Revoke an API key (admin only). Idempotent.",
)
async def revoke_api_key(
    key_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.ADMIN)),
) -> Response:
    repo = APIKeyRepo(session)
    revoked = await repo.revoke(key_id)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key {key_id} not found.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
