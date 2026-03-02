"""
Database Module — Engine and Session Management.

Provides the async SQLAlchemy engine and session factory.
All database I/O in the application uses async/await through asyncpg.

Functions:
    init_db()       — Call on application startup
    close_db()      — Call on application shutdown
    get_session()   — Async context manager yielding a session
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from ive.config import get_settings

log = structlog.get_logger(__name__)

# Module-level engine and session factory (initialised in init_db())
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


class Base(DeclarativeBase):
    """
    Declarative base for all SQLAlchemy ORM models.

    All ORM model classes in ive.db.models inherit from this Base.
    Alembic uses Base.metadata for autogenerate.
    """


async def init_db() -> None:
    """
    Create the async engine and session factory.

    Called once at application startup (see ive.main.lifespan).

    TODO:
        - Configure pool_pre_ping=True for connection health checks
        - Set echo=settings.debug for SQL logging in development
    """
    global _engine, _async_session_factory
    settings = get_settings()

    log.info("ive.db.init", url=settings.database_url.split("@")[-1])

    _engine = create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_timeout=settings.database_pool_timeout,
        pool_pre_ping=True,
        echo=settings.debug,
    )

    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )

    log.info("ive.db.ready")


async def close_db() -> None:
    """
    Dispose the async engine, closing all pooled connections.

    Called at application shutdown (see ive.main.lifespan).
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        log.info("ive.db.closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that yields a database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(select(Dataset))

    Commits on success, rolls back on any exception.
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialised. Call init_db() first.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
