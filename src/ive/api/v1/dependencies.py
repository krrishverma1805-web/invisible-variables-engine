"""
Shared FastAPI Dependencies — Invisible Variables Engine.

Reusable dependency functions injected via ``Depends()``.

Provides:
    - ``get_db()``              — yields an ``AsyncSession`` scoped to the request
    - ``get_current_api_key()`` — returns the validated API key from the header
    - ``get_pagination()``      — standard skip/limit pagination params
    - Typed repository shortcuts for convenience
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.database import get_session
from ive.db.models import Dataset, Experiment, LatentVariable
from ive.db.repositories.dataset_repo import DatasetRepository
from ive.db.repositories.experiment_repo import ExperimentRepository
from ive.db.repositories.latent_variable_repo import LatentVariableRepository
from ive.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Database session
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` bound to the current request.

    The session auto-commits on clean exit and rolls back on any exception.
    Wraps :func:`ive.db.database.get_session`.

    Usage::

        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with get_session() as session:
        yield session


# Alias for backward compatibility with existing endpoints
get_db_session = get_db


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------


async def get_current_api_key(request: Request) -> str:
    """Extract the validated API key from the request.

    The key has already been validated by :class:`APIKeyMiddleware`.
    This dependency simply retrieves it from ``request.state`` for
    audit logging or fine-grained permission checks.

    Args:
        request: The incoming request.

    Returns:
        The API key string.

    Raises:
        ValueError: If the middleware hasn't run (should never happen in
            production).
    """
    api_key = getattr(request.state, "api_key", None)
    if not api_key:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key not found in request state.",
        )
    return api_key


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def get_pagination(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
) -> dict[str, int]:
    """Standard pagination parameters as a dependency.

    Usage::

        @router.get("/items")
        async def list_items(pagination: dict = Depends(get_pagination)):
            skip, limit = pagination["skip"], pagination["limit"]

    Returns:
        Dict with ``skip`` and ``limit`` keys.
    """
    return {"skip": skip, "limit": limit}


# ---------------------------------------------------------------------------
# Repository shortcuts
# ---------------------------------------------------------------------------


def get_dataset_repo(
    session: AsyncSession = Depends(get_db),
) -> DatasetRepository:
    """Return a :class:`DatasetRepository` bound to the request session."""
    return DatasetRepository(session, Dataset)


def get_experiment_repo(
    session: AsyncSession = Depends(get_db),
) -> ExperimentRepository:
    """Return an :class:`ExperimentRepository` bound to the request session."""
    return ExperimentRepository(session, Experiment)


def get_lv_repo(
    session: AsyncSession = Depends(get_db),
) -> LatentVariableRepository:
    """Return a :class:`LatentVariableRepository` bound to the request session."""
    return LatentVariableRepository(session, LatentVariable)
