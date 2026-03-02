"""
Latent Variable Repository — Invisible Variables Engine.

Provides ``LatentVariableRepository`` with experiment-scoped listing,
status-filtered queries, post-bootstrap validation updates, and a
bulk create helper for the construction phase.
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from ive.db.models import LatentVariable
from ive.db.repositories.base_repo import BaseRepository

log = structlog.get_logger(__name__)


class LatentVariableRepository(BaseRepository[LatentVariable]):
    """Query and mutation methods for discovered latent variables."""

    async def get_by_experiment(
        self,
        experiment_id: uuid.UUID,
        status: str | None = None,
    ) -> list[LatentVariable]:
        """Return all latent variables for an experiment.

        Args:
            experiment_id: UUID of the parent experiment.
            status:        Optional filter: ``"candidate"``, ``"validated"``,
                           or ``"rejected"``.  If ``None``, all statuses returned.

        Returns:
            List of ``LatentVariable`` rows ordered by ``importance_score`` descending.
        """
        stmt = select(LatentVariable).where(LatentVariable.experiment_id == experiment_id)
        if status is not None:
            stmt = stmt.where(LatentVariable.status == status)
        stmt = stmt.order_by(LatentVariable.importance_score.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_validated(self, experiment_id: uuid.UUID) -> list[LatentVariable]:
        """Return only ``status="validated"`` latent variables for an experiment.

        Convenience wrapper around :meth:`get_by_experiment` with
        ``status="validated"``.

        Args:
            experiment_id: UUID of the parent experiment.

        Returns:
            Validated ``LatentVariable`` rows, ordered by ``importance_score`` desc.
        """
        return await self.get_by_experiment(experiment_id, status="validated")

    async def update_validation(
        self,
        variable_id: uuid.UUID,
        stability_score: float,
        bootstrap_presence_rate: float,
        status: str,
        confidence_interval_lower: float | None = None,
        confidence_interval_upper: float | None = None,
    ) -> LatentVariable:
        """Apply bootstrap validation results to a latent variable row.

        Updates ``stability_score``, ``bootstrap_presence_rate``, ``status``,
        and optionally the 95% confidence interval bounds.

        Args:
            variable_id:               UUID of the ``LatentVariable`` row.
            stability_score:           Fraction of bootstrap iterations where
                                       the pattern was re-detected (0.0–1.0).
            bootstrap_presence_rate:   Alias for stability_score stored separately.
            status:                    ``"validated"`` or ``"rejected"``.
            confidence_interval_lower: Lower bound of the bootstrapped CI.
            confidence_interval_upper: Upper bound of the bootstrapped CI.

        Returns:
            Updated ``LatentVariable`` instance.

        Raises:
            NoResultFound: If no variable with that ID exists.
        """
        kwargs: dict[str, Any] = {
            "stability_score": stability_score,
            "bootstrap_presence_rate": bootstrap_presence_rate,
            "status": status,
        }
        if confidence_interval_lower is not None:
            kwargs["confidence_interval_lower"] = confidence_interval_lower
        if confidence_interval_upper is not None:
            kwargs["confidence_interval_upper"] = confidence_interval_upper

        instance = await self.update(variable_id, **kwargs)
        if instance is None:
            from sqlalchemy.exc import NoResultFound

            raise NoResultFound(f"LatentVariable {variable_id} not found")

        log.info(
            "latent_variable.validated",
            id=str(variable_id),
            status=status,
            stability=round(stability_score, 3),
        )
        return instance

    async def bulk_create(
        self,
        experiment_id: uuid.UUID,
        variables: list[dict[str, Any]],
    ) -> list[LatentVariable]:
        """Bulk-insert multiple latent variable rows and return them as ORM objects.

        Uses Core ``insert`` for efficiency, then re-fetches via SELECT to
        populate the ORM instances (needed for relationships and defaults).

        Args:
            experiment_id: UUID of the parent experiment.
            variables:     List of column-value dicts (without ``id`` or
                           ``experiment_id`` — these are injected automatically).

        Returns:
            List of created ``LatentVariable`` instances, in insertion order.

        Raises:
            IntegrityError: On constraint violations.
        """
        if not variables:
            return []

        ids = [uuid.uuid4() for _ in variables]
        rows = [
            {"id": ids[i], "experiment_id": experiment_id, **var} for i, var in enumerate(variables)
        ]

        try:
            await self.session.execute(insert(LatentVariable), rows)
            await self.session.flush()
        except IntegrityError as exc:
            log.error(
                "latent_variable.bulk_create_error",
                experiment_id=str(experiment_id),
                error=str(exc),
            )
            raise

        # Re-fetch in the same session to get fully populated ORM instances
        result = await self.session.execute(
            select(LatentVariable)
            .where(LatentVariable.id.in_(ids))
            .order_by(LatentVariable.importance_score.desc())
        )
        instances = list(result.scalars().all())

        log.info(
            "latent_variables.bulk_created",
            experiment_id=str(experiment_id),
            count=len(instances),
        )
        return instances
