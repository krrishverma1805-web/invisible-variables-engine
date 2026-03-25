"""
Generic Base Repository — Invisible Variables Engine.

Provides a typed ``BaseRepository[ModelType]`` that wraps async SQLAlchemy
``select``, ``insert``, ``update``, and ``delete`` operations.  All concrete
repositories in this package inherit from this class and add domain-specific
query methods on top.

Usage::

    class DatasetRepository(BaseRepository[Dataset]):
        ...
"""

from __future__ import annotations

import uuid
from typing import Any, Generic, TypeVar

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.database import Base

log = structlog.get_logger(__name__)

# Generic model type variable — bounded to any SQLAlchemy ORM model class.
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Generic async CRUD repository.

    Wraps SQLAlchemy 2.0 ``select``-based API.  All methods are async and
    commit/rollback is the caller's responsibility (via the session context
    manager in ``ive.db.database.get_session``).

    Args:
        session:     An active ``AsyncSession`` from the session factory.
        model_class: The SQLAlchemy model class (e.g. ``Dataset``).
    """

    def __init__(self, session: AsyncSession, model_class: type[ModelType]) -> None:
        self.session = session
        self.model_class = model_class
        self._log = log.bind(repo=self.__class__.__name__, model=model_class.__name__)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(self, **kwargs: Any) -> ModelType:
        """Insert a new row and return the persisted instance.

        Args:
            **kwargs: Column values for the new row.

        Returns:
            The newly created and flushed model instance.

        Raises:
            IntegrityError: On unique constraint violations.
        """
        instance = self.model_class(**kwargs)
        self.session.add(instance)
        try:
            await self.session.flush()
        except IntegrityError as exc:
            self._log.error("create.integrity_error", error=str(exc))
            raise
        self._log.debug("created", id=str(instance.id))  # type: ignore[attr-defined]
        return instance

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_by_id(self, id: uuid.UUID) -> ModelType | None:
        """Fetch a single row by primary key.

        Args:
            id: The UUID primary key.

        Returns:
            The model instance, or ``None`` if not found.
        """
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == id)  # type: ignore[attr-defined]
        )
        return result.scalar_one_or_none()

    async def get_or_raise(self, id: uuid.UUID) -> ModelType:
        """Fetch a single row or raise ``NoResultFound``.

        Args:
            id: The UUID primary key.

        Raises:
            NoResultFound: If no row with that ID exists.
        """
        instance = await self.get_by_id(id)
        if instance is None:
            raise NoResultFound(f"{self.model_class.__name__} with id={id} not found")
        return instance

    async def get_all(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
    ) -> list[ModelType]:
        """Return a page of rows, optionally filtered by exact column equality.

        Args:
            skip:    Number of rows to skip (offset).
            limit:   Maximum rows to return.
            filters: Dict of ``{column_name: value}`` equality filters.

        Returns:
            A list of model instances.
        """
        stmt = select(self.model_class)
        if filters:
            for col_name, val in filters.items():
                col = getattr(self.model_class, col_name, None)
                if col is not None:
                    stmt = stmt.where(col == val)
        stmt = stmt.offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update(self, id: uuid.UUID, **kwargs: Any) -> ModelType | None:
        """Update columns on an existing row.

        Only the provided keyword arguments are updated; other columns
        are left unchanged.

        Args:
            id:      UUID of the row to update.
            **kwargs: Column name → new value pairs.

        Returns:
            The updated model instance, or ``None`` if not found.
        """
        instance = await self.get_by_id(id)
        if instance is None:
            return None
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        self.session.add(instance)
        await self.session.flush()
        self._log.debug("updated", id=str(id), fields=list(kwargs.keys()))
        return instance

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete(self, id: uuid.UUID) -> bool:
        """Delete a row by primary key.

        Args:
            id: UUID of the row to delete.

        Returns:
            ``True`` if the row was deleted, ``False`` if it was not found.
        """
        instance = await self.get_by_id(id)
        if instance is None:
            return False
        await self.session.delete(instance)
        await self.session.flush()
        self._log.debug("deleted", id=str(id))
        return True

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Return the total number of rows, optionally filtered.

        Args:
            filters: Dict of ``{column_name: value}`` equality filters.

        Returns:
            Row count as an integer.
        """
        stmt = select(func.count()).select_from(self.model_class)
        if filters:
            for col_name, val in filters.items():
                col = getattr(self.model_class, col_name, None)
                if col is not None:
                    stmt = stmt.where(col == val)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def exists(self, id: uuid.UUID) -> bool:
        """Check whether a row with the given primary key exists.

        Args:
            id: UUID primary key.

        Returns:
            ``True`` if the row exists.
        """
        stmt = select(func.count()).select_from(self.model_class).where(self.model_class.id == id)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        return result.scalar_one() > 0
