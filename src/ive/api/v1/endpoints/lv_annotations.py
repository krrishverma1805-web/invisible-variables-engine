"""LV annotation endpoints (Phase C2.1).

Routes (mounted at ``/api/v1/latent-variables/{lv_id}/annotations``):

    GET    /                  List annotations for an LV (read scope).
    POST   /                  Create a new annotation (write scope).
    PUT    /{annotation_id}   Update body (write scope; author or admin).
    DELETE /{annotation_id}   Delete (write scope; author or admin).

Author identity comes from the AuthContext set by the auth middleware
(PR-2). Edit/delete is gated by author == requester OR admin scope so
power users can curate their own notes without an admin round-trip.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ive.api.v1.dependencies import get_db
from ive.api.v1.schemas.lv_annotation_schemas import (
    AnnotationCreate,
    AnnotationListResponse,
    AnnotationResponse,
    AnnotationUpdate,
)
from ive.auth.scopes import AuthContext, Scope, require_scope
from ive.db.repositories.lv_annotation_repo import LVAnnotationRepo

router = APIRouter()


@router.get(
    "/",
    response_model=AnnotationListResponse,
    summary="List annotations for a latent variable.",
)
async def list_annotations(
    lv_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    _actor: AuthContext = Depends(require_scope(Scope.READ)),
) -> AnnotationListResponse:
    repo = LVAnnotationRepo(session)
    rows = await repo.list_for_lv(lv_id)
    return AnnotationListResponse(
        items=[AnnotationResponse.model_validate(r) for r in rows],
        total=len(rows),
    )


@router.post(
    "/",
    response_model=AnnotationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an annotation on a latent variable.",
)
async def create_annotation(
    lv_id: uuid.UUID,
    payload: AnnotationCreate,
    session: AsyncSession = Depends(get_db),
    actor: AuthContext = Depends(require_scope(Scope.WRITE)),
) -> AnnotationResponse:
    repo = LVAnnotationRepo(session)
    # AuthContext.api_key_id is `str | None` (UUID as string when DB-
    # resolved); coerce to UUID for the typed repo signature.
    api_key_uuid = (
        uuid.UUID(actor.api_key_id) if actor.api_key_id else None
    )
    row = await repo.create(
        latent_variable_id=lv_id,
        body=payload.body,
        api_key_id=api_key_uuid,
        api_key_name=actor.api_key_name,
    )
    await session.commit()
    return AnnotationResponse.model_validate(row)


@router.put(
    "/{annotation_id}",
    response_model=AnnotationResponse,
    summary="Update an annotation body.",
)
async def update_annotation(
    lv_id: uuid.UUID,
    annotation_id: uuid.UUID,
    payload: AnnotationUpdate,
    session: AsyncSession = Depends(get_db),
    actor: AuthContext = Depends(require_scope(Scope.WRITE)),
) -> AnnotationResponse:
    repo = LVAnnotationRepo(session)
    existing = await repo.get(annotation_id)
    if existing is None or existing.latent_variable_id != lv_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation {annotation_id} not found on LV {lv_id}.",
        )
    actor_uuid = uuid.UUID(actor.api_key_id) if actor.api_key_id else None
    if (
        existing.api_key_id is not None
        and actor_uuid != existing.api_key_id
        and Scope.ADMIN not in actor.scopes
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the original author or an admin may edit this annotation.",
        )
    row = await repo.update_body(annotation_id, payload.body)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation {annotation_id} not found.",
        )
    await session.commit()
    return AnnotationResponse.model_validate(row)


@router.delete(
    "/{annotation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    summary="Delete an annotation.",
)
async def delete_annotation(
    lv_id: uuid.UUID,
    annotation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    actor: AuthContext = Depends(require_scope(Scope.WRITE)),
) -> None:
    repo = LVAnnotationRepo(session)
    existing = await repo.get(annotation_id)
    if existing is None or existing.latent_variable_id != lv_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Annotation {annotation_id} not found on LV {lv_id}.",
        )
    actor_uuid = uuid.UUID(actor.api_key_id) if actor.api_key_id else None
    if (
        existing.api_key_id is not None
        and actor_uuid != existing.api_key_id
        and Scope.ADMIN not in actor.scopes
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the original author or an admin may delete this annotation.",
        )
    await repo.delete(annotation_id)
    await session.commit()
