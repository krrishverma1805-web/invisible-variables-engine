"""Repository helpers for ``latent_variable_annotations`` (Phase C2.1)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import LatentVariableAnnotation


class LVAnnotationRepo:
    """CRUD for LV annotations.

    The annotation row carries the authoring API key id + name so the
    PR-2 audit log ties to LV history. ``api_key_id`` is set NULL on
    key revocation (FK ``ON DELETE SET NULL``) but the cached name
    stays attached for historical attribution.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_for_lv(
        self,
        latent_variable_id: uuid.UUID,
    ) -> Sequence[LatentVariableAnnotation]:
        stmt = (
            select(LatentVariableAnnotation)
            .where(
                LatentVariableAnnotation.latent_variable_id == latent_variable_id
            )
            .order_by(LatentVariableAnnotation.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get(
        self,
        annotation_id: uuid.UUID,
    ) -> LatentVariableAnnotation | None:
        return await self._session.get(LatentVariableAnnotation, annotation_id)

    async def create(
        self,
        *,
        latent_variable_id: uuid.UUID,
        body: str,
        api_key_id: uuid.UUID | None = None,
        api_key_name: str | None = None,
    ) -> LatentVariableAnnotation:
        # Populate defaults explicitly so the row is fully formed before
        # ``flush()`` (in-process tests + endpoint serialization don't
        # round-trip through the DB to pick up server-side defaults).
        now = datetime.now(UTC)
        row = LatentVariableAnnotation(
            id=uuid.uuid4(),
            latent_variable_id=latent_variable_id,
            body=body,
            api_key_id=api_key_id,
            api_key_name=api_key_name,
            created_at=now,
            updated_at=now,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def update_body(
        self,
        annotation_id: uuid.UUID,
        body: str,
    ) -> LatentVariableAnnotation | None:
        row = await self.get(annotation_id)
        if row is None:
            return None
        row.body = body
        await self._session.flush()
        return row

    async def delete(self, annotation_id: uuid.UUID) -> bool:
        row = await self.get(annotation_id)
        if row is None:
            return False
        await self._session.delete(row)
        await self._session.flush()
        return True
