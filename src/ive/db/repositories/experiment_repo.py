"""
Experiment Repository.

Data access layer for the Experiment ORM model.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import Experiment


class ExperimentRepo:
    """CRUD operations for Experiment records."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_id(self, experiment_id: uuid.UUID) -> Experiment | None:
        """Fetch a single experiment by primary key."""
        return await self.session.get(Experiment, experiment_id)

    async def list(
        self,
        dataset_id: uuid.UUID | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Experiment], int]:
        """
        Return paginated experiments with optional filters.

        TODO:
            - Apply dataset_id filter if provided
            - Apply status filter if provided
        """
        stmt = select(Experiment).order_by(Experiment.created_at.desc())
        if dataset_id is not None:
            stmt = stmt.where(Experiment.dataset_id == dataset_id)
        if status is not None:
            stmt = stmt.where(Experiment.status == status)

        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.session.execute(count_stmt)).scalar_one()
        result = await self.session.execute(stmt.offset(offset).limit(limit))
        return list(result.scalars().all()), total

    async def create(self, data: dict[str, Any]) -> Experiment:
        """Create a new Experiment record."""
        experiment = Experiment(**data)
        self.session.add(experiment)
        await self.session.flush()
        return experiment

    async def update_status(
        self,
        experiment_id: uuid.UUID,
        status: str,
        phase: str | None = None,
        error_msg: str | None = None,
    ) -> Experiment | None:
        """
        Update experiment status and optionally phase and error_msg.

        TODO:
            - Fetch experiment, update status/phase/error_msg, flush
        """
        experiment = await self.get_by_id(experiment_id)
        if experiment is None:
            return None
        experiment.status = status
        if phase is not None:
            experiment.current_phase = phase
        if error_msg is not None:
            experiment.error_msg = error_msg
        await self.session.flush()
        return experiment

    async def delete(self, experiment_id: uuid.UUID) -> bool:
        """Delete an experiment and all cascade relationships."""
        experiment = await self.get_by_id(experiment_id)
        if experiment is None:
            return False
        await self.session.delete(experiment)
        await self.session.flush()
        return True
