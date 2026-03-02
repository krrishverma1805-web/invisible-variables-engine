"""
IVE Database Repositories Package.

Re-exports all repository classes and provides a convenience factory
function :func:`get_repositories` that instantiates every repository
bound to the same ``AsyncSession``::

    async with get_session() as session:
        repos = get_repositories(session)
        dataset = await repos["datasets"].get_by_id(dataset_id)
        experiments = await repos["experiments"].get_by_dataset(dataset_id)
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from ive.db.models import Dataset, Experiment, LatentVariable
from ive.db.repositories.base_repo import BaseRepository
from ive.db.repositories.dataset_repo import DatasetRepository
from ive.db.repositories.experiment_repo import ExperimentRepository
from ive.db.repositories.latent_variable_repo import LatentVariableRepository


def get_repositories(session: AsyncSession) -> dict[str, BaseRepository]:
    """Instantiate all repositories bound to the given session.

    All repositories share the same ``AsyncSession``, so operations can
    span multiple repos within one transaction.

    Args:
        session: An active :class:`sqlalchemy.ext.asyncio.AsyncSession`.

    Returns:
        Dict with keys ``"datasets"``, ``"experiments"``,
        ``"latent_variables"``.

    Example::

        async with get_session() as session:
            repos = get_repositories(session)
            lv = await repos["latent_variables"].get_validated(experiment_id)
    """
    return {
        "datasets": DatasetRepository(session, Dataset),
        "experiments": ExperimentRepository(session, Experiment),
        "latent_variables": LatentVariableRepository(session, LatentVariable),
    }


__all__ = [
    "BaseRepository",
    "DatasetRepository",
    "ExperimentRepository",
    "LatentVariableRepository",
    "get_repositories",
]
