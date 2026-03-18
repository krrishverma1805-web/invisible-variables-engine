"""
Script — Seed Database.

Populates the database with a default API key, a sample dataset record,
and a sample experiment record for immediate development use.

Usage:
    python scripts/seed_db.py
    python scripts/seed_db.py --reset  (drops and re-creates all tables first)

Requires:
    - DATABASE_URL environment variable set (or .env file)
    - Database server running and accessible
    - Alembic migrations applied: make migrate
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Add src/ to path so ive package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("ENV", "development")


async def seed_data(engine: Any) -> None:
    """Main seeding logic."""
    from ive.config import get_settings
    from ive.db.database import close_db, get_session, init_db
    from ive.db.models import Dataset, Experiment

    settings = get_settings()
    print(f"Connecting to: {settings.database_url.split('@')[-1]}")

    await init_db()

    async with get_session() as session:
        # -----------------------------------------------------------------------
        # Seed Dataset
        # -----------------------------------------------------------------------
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Synthetic Housing Dataset",
            description="Auto-generated seed dataset with a planted binary latent variable.",
            target_column="target",
            row_count=1000,
            column_count=6,
            file_path="data/synthetic/binary_latent_1000rows.csv",
            status="profiled",
        )
        session.add(dataset)
        await session.flush()
        print(f"  ✅ Seeded dataset:    {dataset_id}")

        # -----------------------------------------------------------------------
        # Seed Experiment
        # -----------------------------------------------------------------------
        experiment_id = uuid.uuid4()
        experiment = Experiment(
            id=experiment_id,
            dataset_id=dataset_id,
            name="Housing Experiment — Default",
            config={
                "target_column": "target",
                "model_types": ["linear", "xgboost"],
                "cv_folds": 5,
                "max_latent_variables": 5,
                "random_seed": 42,
                "min_cluster_size": 10,
            },
            status="queued",
        )
        session.add(experiment)
        await session.flush()
        print(f"  ✅ Seeded experiment: {experiment_id}")

    await close_db()
    print("\nSeed complete! Use the API key from .env.example to authenticate.")


async def reset_and_seed(engine: Any) -> None:
    """Drop all tables and recreate before seeding."""
    from ive.db.database import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    print("  ✅ Tables reset.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the IVE database with development data")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables and recreate before seeding",
    )
    args = parser.parse_args()

    if args.reset:
        print("⚠️  --reset flag set: dropping and recreating all tables...")

        async def _run():
            from ive.db.database import _engine, close_db, init_db

            await init_db()
            await reset_and_seed(_engine)
            await seed_data(_engine)
            await close_db()

        asyncio.run(_run())
    else:
        asyncio.run(seed_data(None))


if __name__ == "__main__":
    main()
