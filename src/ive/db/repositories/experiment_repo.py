"""
Experiment Repository — Invisible Variables Engine.

Provides ``ExperimentRepository`` with lifecycle helpers (start, fail,
complete, progress updates), eager-loading queries, and efficient bulk
insert methods for child entities (``TrainedModel``, ``Residual``,
``ErrorPattern``).

The bulk residual insert uses ``insert().values()`` with chunking so that
experiments with 100 K+ residual rows don't generate a single enormous SQL
statement that could exceed server-side limits.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import desc, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from ive.db.models import ErrorPattern, Experiment, ExperimentEvent, Residual, TrainedModel
from ive.db.repositories.base_repo import BaseRepository

log = structlog.get_logger(__name__)

# Residuals are bulk-inserted in chunks to stay within PG parameter limits.
_RESIDUAL_CHUNK_SIZE = 500


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ExperimentRepository(BaseRepository[Experiment]):
    """Experiment-specific query methods and child-entity writers."""

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def update_status(
        self,
        experiment_id: uuid.UUID,
        status: str,
        error_message: str | None = None,
    ) -> Experiment:
        """Set the ``status`` (and optionally ``error_message``) of an experiment.

        Args:
            experiment_id: UUID of the experiment.
            status:        New status string (``queued``, ``running``, etc.).
            error_message: Error description, set when ``status="failed"``.

        Returns:
            Updated ``Experiment`` instance.

        Raises:
            NoResultFound: If no experiment with that ID exists.
        """
        kwargs: dict[str, Any] = {"status": status}
        if error_message is not None:
            kwargs["error_message"] = error_message
        instance = await self.update(experiment_id, **kwargs)
        if instance is None:
            from sqlalchemy.exc import NoResultFound

            raise NoResultFound(f"Experiment {experiment_id} not found")
        log.info("experiment.status_updated", id=str(experiment_id), status=status)
        return instance

    async def update_progress(
        self,
        experiment_id: uuid.UUID,
        progress_pct: int,
        current_stage: str,
    ) -> Experiment:
        """Update ``progress_pct`` and ``current_stage`` without touching status.

        Args:
            experiment_id: UUID of the experiment.
            progress_pct:  Integer 0–100.
            current_stage: Current pipeline stage name.

        Returns:
            Updated ``Experiment`` instance.
        """
        instance = await self.update(
            experiment_id,
            progress_pct=progress_pct,
            current_stage=current_stage,
        )
        if instance is None:
            from sqlalchemy.exc import NoResultFound

            raise NoResultFound(f"Experiment {experiment_id} not found")
        return instance

    async def mark_started(self, experiment_id: uuid.UUID) -> Experiment:
        """Set ``status="running"`` and record ``started_at``.

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            Updated ``Experiment`` instance.
        """
        instance = await self.update(
            experiment_id,
            status="running",
            started_at=_utcnow(),
            progress_pct=0,
            current_stage="understand",
        )
        if instance is None:
            from sqlalchemy.exc import NoResultFound

            raise NoResultFound(f"Experiment {experiment_id} not found")
        log.info("experiment.started", id=str(experiment_id))
        return instance

    async def mark_completed(self, experiment_id: uuid.UUID) -> Experiment:
        """Set ``status="completed"`` and record ``completed_at``.

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            Updated ``Experiment`` instance.
        """
        instance = await self.update(
            experiment_id,
            status="completed",
            progress_pct=100,
            current_stage=None,
            completed_at=_utcnow(),
        )
        if instance is None:
            from sqlalchemy.exc import NoResultFound

            raise NoResultFound(f"Experiment {experiment_id} not found")
        log.info("experiment.completed", id=str(experiment_id))
        return instance

    async def mark_failed(self, experiment_id: uuid.UUID, error_message: str) -> Experiment:
        """Set ``status="failed"`` with an error message and record ``completed_at``.

        Args:
            experiment_id: UUID of the experiment.
            error_message: Human-readable failure description.

        Returns:
            Updated ``Experiment`` instance.
        """
        instance = await self.update(
            experiment_id,
            status="failed",
            error_message=error_message,
            completed_at=_utcnow(),
        )
        if instance is None:
            from sqlalchemy.exc import NoResultFound

            raise NoResultFound(f"Experiment {experiment_id} not found")
        log.error(
            "experiment.failed",
            id=str(experiment_id),
            error=error_message[:200],
        )
        return instance

    async def mark_cancelled(self, experiment_id: uuid.UUID) -> Experiment:
        """Set ``status="cancelled"`` and record ``completed_at``.

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            Updated ``Experiment`` instance.
        """
        instance = await self.update(
            experiment_id,
            status="cancelled",
            completed_at=_utcnow(),
        )
        if instance is None:
            from sqlalchemy.exc import NoResultFound

            raise NoResultFound(f"Experiment {experiment_id} not found")
        return instance

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get_by_dataset(self, dataset_id: uuid.UUID) -> list[Experiment]:
        """Return all experiments for a dataset, newest first.

        Args:
            dataset_id: UUID of the parent dataset.

        Returns:
            List of ``Experiment`` rows ordered by ``created_at`` descending.
        """
        result = await self.session.execute(
            select(Experiment)
            .where(Experiment.dataset_id == dataset_id)
            .order_by(desc(Experiment.created_at))
        )
        return list(result.scalars().all())

    async def get_running(self) -> list[Experiment]:
        """Return all experiments currently in ``status="running"``.

        Used by the health check and scheduler to detect stale jobs.

        Returns:
            List of running ``Experiment`` rows.
        """
        result = await self.session.execute(
            select(Experiment).where(Experiment.status == "running").order_by(Experiment.started_at)
        )
        return list(result.scalars().all())

    async def get_with_results(self, experiment_id: uuid.UUID) -> Experiment | None:
        """Eagerly load an experiment with its full result relationships.

        Loads ``trained_models``, ``residuals`` (summary rows), ``error_patterns``,
        and ``latent_variables`` in separate IN-clause queries (``selectinload``).

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            ``Experiment`` with all result collections populated, or ``None``.
        """
        result = await self.session.execute(
            select(Experiment)
            .options(
                selectinload(Experiment.trained_models),
                selectinload(Experiment.error_patterns),
                selectinload(Experiment.latent_variables),
            )
            .where(Experiment.id == experiment_id)
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # Child entity writers
    # ------------------------------------------------------------------

    async def add_trained_model(self, experiment_id: uuid.UUID, **kwargs: Any) -> TrainedModel:
        """Insert a ``TrainedModel`` row for the given experiment.

        Args:
            experiment_id: UUID of the parent experiment.
            **kwargs:      Column values (``model_type``, ``fold_number``,
                           ``train_metric``, ``val_metric``, ``metric_name``, …).

        Returns:
            The newly created ``TrainedModel`` instance.

        Raises:
            IntegrityError: On duplicate ``(experiment_id, model_type, fold_number)``.
        """
        instance = TrainedModel(experiment_id=experiment_id, **kwargs)
        self.session.add(instance)
        try:
            await self.session.flush()
        except IntegrityError as exc:
            log.error(
                "trained_model.integrity_error", experiment_id=str(experiment_id), error=str(exc)
            )
            raise
        log.debug(
            "trained_model.created",
            id=str(instance.id),
            type=kwargs.get("model_type"),
            fold=kwargs.get("fold_number"),
        )
        return instance

    async def add_residuals_batch(
        self,
        experiment_id: uuid.UUID,
        residuals: list[dict[str, Any]],
    ) -> int:
        """Bulk-insert residuals efficiently using Core INSERT.

        Since residuals can run to hundreds of thousands of rows, this method
        uses ``sqlalchemy.insert`` with chunked ``executemany`` semantics instead
        of individual ORM ``add()`` calls.  Each chunk is capped at
        ``_RESIDUAL_CHUNK_SIZE`` rows to stay within PostgreSQL's parameter limit.

        Args:
            experiment_id: UUID of the parent experiment (injected into every row).
            residuals:     List of dicts with keys matching ``Residual`` columns
                           (``model_type``, ``sample_index``, ``fold_number``,
                           ``actual_value``, ``predicted_value``,
                           ``residual_value``, ``abs_residual``).

        Returns:
            Total number of rows inserted.
        """
        if not residuals:
            return 0

        total = 0
        # Inject experiment_id and a fresh UUID into every row dict
        rows = [
            {
                "id": uuid.uuid4(),
                "experiment_id": experiment_id,
                **row,
            }
            for row in residuals
        ]

        # Chunk to stay within pg_prepare parameter limit (65535 / columns)
        for i in range(0, len(rows), _RESIDUAL_CHUNK_SIZE):
            chunk = rows[i : i + _RESIDUAL_CHUNK_SIZE]
            await self.session.execute(insert(Residual), chunk)
            total += len(chunk)

        log.info(
            "residuals.bulk_inserted",
            experiment_id=str(experiment_id),
            count=total,
        )
        return total

    async def add_error_pattern(self, experiment_id: uuid.UUID, **kwargs: Any) -> ErrorPattern:
        """Insert an ``ErrorPattern`` row for the given experiment.

        Args:
            experiment_id: UUID of the parent experiment.
            **kwargs:      Column values (``pattern_type``, ``subgroup_definition``,
                           ``effect_size``, ``p_value``, etc.).

        Returns:
            The newly created ``ErrorPattern`` instance.
        """
        instance = ErrorPattern(experiment_id=experiment_id, **kwargs)
        self.session.add(instance)
        await self.session.flush()
        log.debug(
            "error_pattern.created",
            id=str(instance.id),
            type=kwargs.get("pattern_type"),
        )
        return instance

    async def add_error_patterns_batch(
        self,
        experiment_id: uuid.UUID,
        patterns: list[dict[str, Any]],
    ) -> int:
        """Bulk-insert multiple ``ErrorPattern`` rows efficiently.

        Args:
            experiment_id: UUID of the parent experiment.
            patterns:      List of column-value dicts.

        Returns:
            Number of patterns inserted.
        """
        if not patterns:
            return 0
        rows = [{"id": uuid.uuid4(), "experiment_id": experiment_id, **p} for p in patterns]
        await self.session.execute(insert(ErrorPattern), rows)
        log.info("error_patterns.bulk_inserted", experiment_id=str(experiment_id), count=len(rows))
        return len(rows)

    # ------------------------------------------------------------------
    # Child entity queries
    # ------------------------------------------------------------------

    async def get_residuals(
        self,
        experiment_id: uuid.UUID,
        model_type: str | None = None,
        *,
        skip: int = 0,
        limit: int = 1000,
    ) -> list[Residual]:
        """Return residuals for an experiment, optionally filtered by model type.

        Args:
            experiment_id: UUID of the parent experiment.
            model_type:    Optional model type filter (``"linear"`` or ``"xgboost"``).
            skip:          Offset for pagination.
            limit:         Max rows (default 1000 — use pagination for large datasets).

        Returns:
            List of ``Residual`` rows.
        """
        stmt = select(Residual).where(Residual.experiment_id == experiment_id)
        if model_type:
            stmt = stmt.where(Residual.model_type == model_type)
        stmt = stmt.order_by(Residual.sample_index).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_error_patterns(self, experiment_id: uuid.UUID) -> list[ErrorPattern]:
        """Return all error patterns for an experiment, ordered by effect size.

        Args:
            experiment_id: UUID of the parent experiment.

        Returns:
            List of ``ErrorPattern`` rows, largest effect first.
        """
        result = await self.session.execute(
            select(ErrorPattern)
            .where(ErrorPattern.experiment_id == experiment_id)
            .order_by(desc(ErrorPattern.effect_size))
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Experiment event log
    # ------------------------------------------------------------------

    async def add_event(
        self,
        experiment_id: uuid.UUID,
        event_type: str,
        message: str,
        phase: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentEvent:
        """Append a single event to the experiment audit log.

        Events are append-only; they are never updated or deleted.  Each
        event captures a discrete lifecycle milestone (e.g. ``dataset_loaded``,
        ``modeling_completed``) along with a human-readable ``message`` and
        an optional ``metadata`` payload stored as JSONB.

        Args:
            experiment_id: UUID of the parent experiment.
            event_type:    Machine-readable event identifier.
            message:       Human-readable description of the event.
            phase:         Optional pipeline phase label
                           (``understand`` / ``model`` / ``detect`` / ``construct``).
            metadata:      Optional supplementary data to persist as JSONB.

        Returns:
            The newly created ``ExperimentEvent`` ORM instance.
        """
        event = ExperimentEvent(
            experiment_id=experiment_id,
            event_type=event_type,
            phase=phase,
            payload={"message": message, **(metadata or {})},
        )
        self.session.add(event)
        await self.session.flush()
        log.debug(
            "experiment_event.created",
            experiment_id=str(experiment_id),
            event_type=event_type,
            phase=phase,
        )
        return event

    async def get_events(
        self,
        experiment_id: uuid.UUID,
        *,
        limit: int = 200,
    ) -> list[ExperimentEvent]:
        """Return the audit-log events for an experiment in chronological order.

        Args:
            experiment_id: UUID of the parent experiment.
            limit:         Maximum events to return (default 200).

        Returns:
            List of ``ExperimentEvent`` rows ordered by ``created_at`` ascending.
        """
        result = await self.session.execute(
            select(ExperimentEvent)
            .where(ExperimentEvent.experiment_id == experiment_id)
            .order_by(ExperimentEvent.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())
