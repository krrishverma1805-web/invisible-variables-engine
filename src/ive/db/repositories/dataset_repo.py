"""
Dataset Repository — Invisible Variables Engine.

Provides ``DatasetRepository``, which extends
:class:`~ive.db.repositories.base_repo.BaseRepository` with dataset-specific
query methods: duplicate detection by checksum, eager-loading of experiments,
and recency ordering.
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from ive.db.models import Dataset
from ive.db.repositories.base_repo import BaseRepository

log = structlog.get_logger(__name__)


class DatasetRepository(BaseRepository[Dataset]):
    """Dataset-specific query methods on top of :class:`BaseRepository`."""

    async def get_by_checksum(self, checksum: str) -> Dataset | None:
        """Fetch a dataset by its SHA-256 file checksum.

        Used to detect duplicate uploads before persisting the file.

        Args:
            checksum: 64-character hex SHA-256 digest.

        Returns:
            Existing ``Dataset`` row if found, ``None`` otherwise.
        """
        result = await self.session.execute(select(Dataset).where(Dataset.checksum == checksum))
        dataset = result.scalar_one_or_none()
        if dataset:
            log.debug("dataset.duplicate_found", checksum=checksum[:8] + "...", id=str(dataset.id))
        return dataset

    async def get_with_experiments(self, dataset_id: uuid.UUID) -> Dataset | None:
        """Fetch a dataset and eagerly load its ``experiments`` relationship.

        Using ``selectinload`` avoids the N+1 problem by issuing a single
        IN-clause query for related experiments.

        Args:
            dataset_id: UUID of the dataset.

        Returns:
            ``Dataset`` with ``experiments`` populated, or ``None``.
        """
        result = await self.session.execute(
            select(Dataset)
            .options(selectinload(Dataset.experiments))
            .where(Dataset.id == dataset_id)
        )
        return result.scalar_one_or_none()

    async def get_recent(self, limit: int = 10) -> list[Dataset]:
        """Return the most recently created datasets.

        Args:
            limit: Maximum number of rows to return (default 10).

        Returns:
            List of ``Dataset`` rows ordered by ``created_at`` descending.
        """
        result = await self.session.execute(
            select(Dataset).order_by(desc(Dataset.created_at)).limit(limit)
        )
        return list(result.scalars().all())

    async def search_by_name(self, query: str, limit: int = 20) -> list[Dataset]:
        """Case-insensitive ``ILIKE`` search on the ``name`` column.

        Args:
            query: Free-text search string (SQL wildcards allowed).
            limit: Maximum rows to return.

        Returns:
            Matching datasets ordered by name.
        """
        pattern = f"%{query}%"
        result = await self.session.execute(
            select(Dataset).where(Dataset.name.ilike(pattern)).order_by(Dataset.name).limit(limit)
        )
        return list(result.scalars().all())
