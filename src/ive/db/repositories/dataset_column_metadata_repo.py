"""Repository helpers for ``dataset_column_metadata``.

Per plan §142 / §174 / §203: a binary public/non_public sensitivity model.
Default for every column is ``non_public`` (safe by default); the upload
flow auto-creates one row per column at sensitivity ``non_public`` so the
record exists from day one.  The user opts in by promoting columns to
``public`` later.

Plan reference: §142 (per-column metadata), §174 (sensitive UX),
§203 (binary model — no name_only middle tier), §186 (egress E2E test).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import DatasetColumnMetadata

Sensitivity = Literal["public", "non_public"]


class DatasetColumnMetadataRepo:
    """CRUD + bulk operations on ``dataset_column_metadata``."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_for_dataset(
        self,
        dataset_id: uuid.UUID,
    ) -> Sequence[DatasetColumnMetadata]:
        stmt = (
            select(DatasetColumnMetadata)
            .where(DatasetColumnMetadata.dataset_id == dataset_id)
            .order_by(DatasetColumnMetadata.column_name.asc())
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get(
        self,
        dataset_id: uuid.UUID,
        column_name: str,
    ) -> DatasetColumnMetadata | None:
        stmt = select(DatasetColumnMetadata).where(
            DatasetColumnMetadata.dataset_id == dataset_id,
            DatasetColumnMetadata.column_name == column_name,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def bulk_create_default(
        self,
        dataset_id: uuid.UUID,
        column_names: Iterable[str],
    ) -> list[DatasetColumnMetadata]:
        """Insert one ``non_public`` row per column. Idempotent for new rows.

        Skips columns that already have a row (used when a user re-uploads
        the same dataset; existing decisions are preserved).
        """
        existing = {row.column_name for row in await self.list_for_dataset(dataset_id)}
        created: list[DatasetColumnMetadata] = []
        for name in column_names:
            if name in existing:
                continue
            row = DatasetColumnMetadata(
                dataset_id=dataset_id,
                column_name=name,
                sensitivity="non_public",
            )
            self._session.add(row)
            created.append(row)
        if created:
            await self._session.flush()
        return created

    async def set_sensitivity(
        self,
        dataset_id: uuid.UUID,
        column_name: str,
        sensitivity: Sensitivity,
    ) -> DatasetColumnMetadata | None:
        row = await self.get(dataset_id, column_name)
        if row is None:
            return None
        row.sensitivity = sensitivity
        await self._session.flush()
        return row

    async def bulk_set(
        self,
        dataset_id: uuid.UUID,
        updates: dict[str, Sensitivity],
    ) -> list[DatasetColumnMetadata]:
        """Update sensitivity for many columns in one call.

        Returns the rows that were actually updated (silently skips any
        column names that don't exist for the dataset — caller decides
        whether to surface that as a 404 or a partial-success).
        """
        rows = await self.list_for_dataset(dataset_id)
        by_name = {row.column_name: row for row in rows}
        changed: list[DatasetColumnMetadata] = []
        for col, sensitivity in updates.items():
            row = by_name.get(col)
            if row is None:
                continue
            if row.sensitivity != sensitivity:
                row.sensitivity = sensitivity
                changed.append(row)
        if changed:
            await self._session.flush()
        return changed

    async def public_column_names(self, dataset_id: uuid.UUID) -> set[str]:
        """Set of column names safe to send to the LLM."""
        stmt = select(DatasetColumnMetadata.column_name).where(
            DatasetColumnMetadata.dataset_id == dataset_id,
            DatasetColumnMetadata.sensitivity == "public",
        )
        result = await self._session.execute(stmt)
        return {row[0] for row in result.all()}
