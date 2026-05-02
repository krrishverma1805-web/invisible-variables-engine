"""
Database Module — Engine and Session Management.

Provides the async SQLAlchemy engine, session factory, and the declarative
base class used by all ORM models.

Functions:
    init_db()       — Call once at application startup (FastAPI lifespan)
    close_db()      — Call once at application shutdown
    get_session()   — Async context manager yielding a bound session

The ``Base`` class lives here (not in ``models.py``) to avoid circular imports,
since ``models.py`` needs ``Base`` and ``database.py`` needs the config.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from ive.config import get_settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Naming convention — ensures Alembic can auto-name constraints portably.
# See https://alembic.sqlalchemy.org/en/latest/naming.html
# ---------------------------------------------------------------------------
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):  # type: ignore[misc]
    """Declarative base for all IVE ORM models.

    All model classes in ``ive.db.models`` inherit from this base.
    Alembic uses ``Base.metadata`` for autogenerate support.
    """

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ---------------------------------------------------------------------------
# Module-level singletons (initialised in init_db)
# ---------------------------------------------------------------------------
_engine: AsyncEngine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_db() -> None:
    """Create the async engine and session factory.

    Must be called **once** at application startup — typically in the FastAPI
    lifespan handler or the Celery ``worker_init`` signal::

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await init_db()
            yield
            await close_db()

    Pool settings are read from :class:`ive.config.Settings`.
    """
    global _engine, _async_session_factory
    settings = get_settings()

    # Log the host portion only — never the password
    safe_url = settings.database_url.split("@")[-1] if "@" in settings.database_url else "<local>"
    log.info("db.initialising", host=safe_url, pool_size=settings.database_pool_size)

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

    log.info("db.ready")


async def close_db() -> None:
    """Dispose the async engine, returning all pooled connections.

    Called at application shutdown.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        log.info("db.closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an ``AsyncSession`` that auto-commits on success, rolls back on error.

    Usage::

        async with get_session() as session:
            result = await session.execute(select(Dataset))
            datasets = result.scalars().all()

    Raises:
        RuntimeError: If :func:`init_db` has not been called yet.
    """
    if _async_session_factory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")

    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_engine() -> AsyncEngine:
    """Return the current async engine (for raw operations / health checks).

    Raises:
        RuntimeError: If :func:`init_db` has not been called.
    """
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the configured async session factory, or ``None`` if not initialised.

    Unlike :func:`get_session` this does **not** raise — the auth
    middleware and other early-path callers need to check whether the DB
    is wired up and degrade gracefully when it is not (e.g. tests).
    """
    return _async_session_factory
