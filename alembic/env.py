"""
Alembic environment configuration for Invisible Variables Engine.

This module configures the Alembic migration environment, connecting it to the
IVE SQLAlchemy metadata so that autogenerate can detect schema changes.

Supports both synchronous (used by Alembic CLI) and offline modes.
"""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

# ---------------------------------------------------------------------------
# Import the IVE metadata so Alembic can detect model changes
# ---------------------------------------------------------------------------
# TODO: Once the DB models module is implemented, this import provides the
#       declarative Base.metadata that Alembic uses for autogenerate.
from ive.db.models import Base  # noqa: E402

# ---------------------------------------------------------------------------
# Alembic Config object (gives access to alembic.ini values)
# ---------------------------------------------------------------------------
config = context.config

# Interpret the config file for Python logging if present
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override the SQLAlchemy URL from the environment if set
# This allows Docker/CI environments to inject the connection string at runtime
_db_url = os.getenv("DATABASE_URL", "").replace(
    "postgresql+asyncpg", "postgresql"  # Alembic CLI needs sync driver
)
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)

# Provide the target metadata for autogenerate support
target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode (synchronous wrapper around async engine)
# ---------------------------------------------------------------------------

def do_run_migrations(connection: object) -> None:
    """Execute migrations within a synchronous connection context."""
    context.configure(
        connection=connection,  # type: ignore[arg-type]
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations using an async engine (required for asyncpg driver)."""
    # Build a synchronous DSN for Alembic (asyncpg can't be used by Alembic CLI directly)
    connectable_cfg = config.get_section(config.config_ini_section, {})
    connectable_cfg["sqlalchemy.url"] = config.get_main_option("sqlalchemy.url", "")

    connectable = async_engine_from_config(
        connectable_cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
