"""
Dataset Repository.

Data access layer for the Dataset ORM model. Implements the repository
pattern to decouple business logic from database access details.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import Dataset


class DatasetRepo:
    """CRUD operations for Dataset records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, dataset_id: uuid.UUID) -> Dataset | None:
        """
        Fetch a single dataset by primary key.

        Returns None if not found.

        TODO:
            - return await self.session.get(Dataset, dataset_id)
        """
        return await self.session.get(Dataset, dataset_id)

    async def list(
        self,
        offset: int = 0,
        limit: int = 20,
        search: str | None = None,
    ) -> tuple[list[Dataset], int]:
        """
        Return paginated datasets with total count.

        TODO:
            - Build select query with optional ilike filter on name
            - Execute with offset/limit
            - Return (items, total)
        """
        stmt = select(Dataset).order_by(Dataset.created_at.desc())
        if search:
            stmt = stmt.where(Dataset.name.ilike(f"%{search}%"))

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()

        result = await self.session.execute(stmt.offset(offset).limit(limit))
        items = list(result.scalars().all())
        return items, total

    async def create(self, data: dict[str, Any]) -> Dataset:
        """
        Create and persist a new Dataset record.

        TODO:
            - dataset = Dataset(**data)
            - self.session.add(dataset)
            - await self.session.flush()  # get generated ID
            - return dataset
        """
        dataset = Dataset(**data)
        self.session.add(dataset)
        await self.session.flush()
        return dataset

    async def update(self, dataset_id: uuid.UUID, data: dict[str, Any]) -> Dataset | None:
        """Update a dataset record. Returns None if not found."""
        dataset = await self.get_by_id(dataset_id)
        if dataset is None:
            return None
        for key, value in data.items():
            setattr(dataset, key, value)
        await self.session.flush()
        return dataset

    async def delete(self, dataset_id: uuid.UUID) -> bool:
        """Delete a dataset. Returns True if deleted, False if not found."""
        dataset = await self.get_by_id(dataset_id)
        if dataset is None:
            return False
        await self.session.delete(dataset)
        await self.session.flush()
        return True
