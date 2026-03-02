"""
Latent Variable Repository.

Data access layer for the LatentVariable ORM model.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import LatentVariable


class LatentVariableRepo:
    """CRUD operations for LatentVariable records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, lv_id: uuid.UUID) -> LatentVariable | None:
        """Fetch a single latent variable by primary key."""
        return await self.session.get(LatentVariable, lv_id)

    async def list_by_experiment(
        self,
        experiment_id: uuid.UUID,
    ) -> list[LatentVariable]:
        """Return all latent variables for an experiment, ordered by rank."""
        stmt = (
            select(LatentVariable)
            .where(LatentVariable.experiment_id == experiment_id)
            .order_by(LatentVariable.rank)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def bulk_create(
        self,
        experiment_id: uuid.UUID,
        candidates: list[dict[str, Any]],
    ) -> list[LatentVariable]:
        """
        Create multiple LatentVariable records in a single flush.

        Args:
            experiment_id: Parent experiment UUID.
            candidates: List of dicts with LatentVariable field values.

        Returns:
            List of created LatentVariable ORM objects.

        TODO:
            - Build LatentVariable objects from each candidate dict
            - Batch add to session and flush once
        """
        items: list[LatentVariable] = []
        for data in candidates:
            lv = LatentVariable(experiment_id=experiment_id, **data)
            self.session.add(lv)
            items.append(lv)
        await self.session.flush()
        return items

    async def delete_for_experiment(self, experiment_id: uuid.UUID) -> int:
        """
        Delete all latent variables for an experiment.

        Returns: Number of deleted rows.

        TODO:
            - Execute DELETE WHERE experiment_id = ... and return rowcount
        """
        items = await self.list_by_experiment(experiment_id)
        for item in items:
            await self.session.delete(item)
        await self.session.flush()
        return len(items)
