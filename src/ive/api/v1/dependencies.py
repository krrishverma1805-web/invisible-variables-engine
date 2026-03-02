"""
FastAPI Dependencies.

Reusable dependency functions injected via FastAPI's Depends() mechanism.
Provides database sessions, repository instances, and current-user context.
"""

from __future__ import annotations

from typing import AsyncGenerator

import structlog
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

log = structlog.get_logger(__name__)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield an async SQLAlchemy session, closing it after the request.

    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db_session)):
            ...

    TODO:
        - Import async_session_factory from ive.db.database
        - Use async with async_session_factory() as session: yield session
    """
    # TODO: Implement real session factory
    # from ive.db.database import async_session_factory
    # async with async_session_factory() as session:
    #     yield session
    raise NotImplementedError("DB session dependency not yet implemented")


async def get_current_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> str:
    """
    Extract and validate the API key from the request header.

    Note: Primary validation is in APIKeyMiddleware. This dependency is
    available for endpoints that want the key value (e.g., for audit logs).

    TODO:
        - Return the key for audit/logging use
        - Or look up the associated identity in a DB keys table
    """
    # TODO: Validate and return the API key principal
    return x_api_key


# ---------------------------------------------------------------------------
# Repository dependencies (convenience shortcuts)
# ---------------------------------------------------------------------------

# TODO: Uncomment these as the repository classes are implemented
#
# from ive.db.repositories.dataset_repo import DatasetRepo
# from ive.db.repositories.experiment_repo import ExperimentRepo
# from ive.db.repositories.latent_variable_repo import LatentVariableRepo
#
# def get_dataset_repo(db: AsyncSession = Depends(get_db_session)) -> DatasetRepo:
#     return DatasetRepo(db)
#
# def get_experiment_repo(db: AsyncSession = Depends(get_db_session)) -> ExperimentRepo:
#     return ExperimentRepo(db)
#
# def get_lv_repo(db: AsyncSession = Depends(get_db_session)) -> LatentVariableRepo:
#     return LatentVariableRepo(db)
