"""
Alembic migration environment for Invisible Variables Engine.

This module wires Alembic to the IVE SQLAlchemy metadata so that
``alembic revision --autogenerate`` can detect schema changes.

The Alembic CLI requires a **synchronous** database driver.  Since the
application uses ``asyncpg``, this env.py rewrites ``DATABASE_URL`` to
use ``psycopg2`` (the sync driver) before creating the engine.

Supports both offline (SQL script generation) and online (direct apply) modes.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# ---------------------------------------------------------------------------
# Import the ORM metadata so Alembic can detect model changes.
# We must import the models module to ensure every model class is registered
# on Base.metadata before Alembic inspects it.
# ---------------------------------------------------------------------------
from ive.db.database import Base
import ive.db.models  # noqa: F401 — registers all model classes on Base.metadata

# ---------------------------------------------------------------------------
# Alembic Config object (provides access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Configure Python logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ---------------------------------------------------------------------------
# Resolve the sync database URL
# ---------------------------------------------------------------------------
# Priority: DATABASE_URL env var (rewritten to sync) > alembic.ini value
_env_url: str | None = os.getenv("DATABASE_URL")
if _env_url:
    # Replace async driver with sync driver for Alembic CLI
    sync_url = (
        _env_url
        .replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        .replace("postgresql://", "postgresql+psycopg2://", 1)
    )
    config.set_main_option("sqlalchemy.url", sync_url)

# Target metadata for autogenerate
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline mode — emit SQL text without a live database connection
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Emits SQL to stdout (or ``--sql`` file) without connecting to a database.
    Useful for generating migration scripts in CI where the DB is unavailable.
    """
    url = config.get_main_option("sqlalchemy.url")

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
        render_as_batch=False,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode — apply migrations against a live database
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    """Run migrations in 'online' mode with a sync engine.

    Creates a standard (non-async) ``Engine``, opens a connection, and
    executes migrations within a transaction.
    """
    connectable_cfg = config.get_section(config.config_ini_section, {})
    connectable_cfg["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url", "")

    connectable = create_engine(
        connectable_cfg["sqlalchemy.url"],
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=False,
        )
        with context.begin_transaction():
            context.run_migrations()

    connectable.dispose()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
