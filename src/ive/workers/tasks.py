"""
Celery Task Definitions — Invisible Variables Engine.

All tasks are **synchronous** (Celery runs them in a regular thread/process).
Database operations use psycopg2 directly (sync driver) to avoid the overhead
of spinning up an asyncio event loop per task.

Tasks
-----
run_experiment      — Full 4-phase IVE pipeline skeleton
profile_dataset     — Post-upload dataset profiling (triggered after upload)
cancel_experiment   — Revoke a running task + mark DB as cancelled
health_check_task   — Worker liveness probe (returns immediately)

Progress reporting
------------------
run_experiment calls ``self.update_state(state='PROGRESS', meta={...})``
so the WebSocket endpoint can poll ``celery_app.AsyncResult(task_id).info``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from celery import Task

from ive.workers.celery_app import celery_app

log = structlog.get_logger("ive.workers.tasks")


# ---------------------------------------------------------------------------
# Sync DB helper  (uses psycopg2 — safe for Celery sync workers)
# ---------------------------------------------------------------------------

def _get_sync_conn():
    """Return a psycopg2 connection using the sync database URL."""
    import psycopg2

    from ive.config import get_settings
    settings = get_settings()
    # sync_database_url is "postgresql://..." (not asyncpg)
    dsn = settings.sync_database_url.replace("postgresql+psycopg2://", "postgresql://")
    return psycopg2.connect(dsn)


def _update_experiment(
    experiment_id: str,
    status: str,
    *,
    progress_pct: int | None = None,
    current_stage: str | None = None,
    error_message: str | None = None,
) -> None:
    """Update an experiment row via a direct psycopg2 connection.

    Args:
        experiment_id: String UUID of the experiment.
        status:        New status string (queued / running / completed / failed / cancelled).
        progress_pct:  0–100 progress percentage.  ``None`` means no change.
        current_stage: Pipeline stage name or ``None``.
        error_message: Error string for failed experiments.
    """
    now = datetime.now(timezone.utc)
    conn = _get_sync_conn()
    try:
        cur = conn.cursor()
        if status == "running":
            cur.execute(
                """
                UPDATE experiments
                SET status = %s, started_at = %s,
                    progress_pct = COALESCE(%s, progress_pct),
                    current_stage = COALESCE(%s, current_stage)
                WHERE id = %s::uuid
                """,
                (status, now, progress_pct, current_stage, experiment_id),
            )
        elif status == "completed":
            cur.execute(
                """
                UPDATE experiments
                SET status = %s, completed_at = %s,
                    progress_pct = 100, current_stage = 'done'
                WHERE id = %s::uuid
                """,
                (status, now, experiment_id),
            )
        elif status == "failed":
            cur.execute(
                """
                UPDATE experiments
                SET status = %s, completed_at = %s, error_message = %s
                WHERE id = %s::uuid
                """,
                (status, now, error_message, experiment_id),
            )
        elif status == "cancelled":
            cur.execute(
                """
                UPDATE experiments
                SET status = %s, completed_at = %s, current_stage = 'cancelled'
                WHERE id = %s::uuid
                """,
                (status, now, experiment_id),
            )
        else:
            # Generic update — at least update progress/stage if supplied
            sets = ["status = %s"]
            params: list[Any] = [status]
            if progress_pct is not None:
                sets.append("progress_pct = %s")
                params.append(progress_pct)
            if current_stage is not None:
                sets.append("current_stage = %s")
                params.append(current_stage)
            params.append(experiment_id)
            cur.execute(
                f"UPDATE experiments SET {', '.join(sets)} WHERE id = %s::uuid",
                params,
            )
        conn.commit()
        log.debug("task.db_updated", experiment_id=experiment_id, status=status)
    except Exception as exc:
        conn.rollback()
        log.error("task.db_update_failed", experiment_id=experiment_id, error=str(exc))
    finally:
        conn.close()


def _get_experiment(experiment_id: str) -> dict[str, Any] | None:
    """Fetch a minimal experiment row (status, celery_task_id, file_path)."""
    conn = _get_sync_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.id, e.status, e.celery_task_id, e.config_json, d.file_path
            FROM experiments e
            JOIN datasets d ON e.dataset_id = d.id
            WHERE e.id = %s::uuid
            """,
            (experiment_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "id": str(row[0]),
            "status": row[1],
            "celery_task_id": row[2],
            "config_json": row[3],
            "file_path": row[4],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Base task class with on_failure hook
# ---------------------------------------------------------------------------

class BaseIVETask(Task):
    """Celery base task that marks experiments as ``failed`` on unhandled errors."""

    abstract = True

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        experiment_id = kwargs.get("experiment_id") or (args[0] if args else None)
        if experiment_id:
            _update_experiment(str(experiment_id), "failed", error_message=str(exc))
        log.error(
            "task.failed",
            task_id=task_id,
            task_name=self.name,
            experiment_id=experiment_id,
            error=str(exc),
        )
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        log.warning(
            "task.retry",
            task_id=task_id,
            task_name=self.name,
            attempt=self.request.retries + 1,
            error=str(exc),
        )
        super().on_retry(exc, task_id, args, kwargs, einfo)


# ---------------------------------------------------------------------------
# Task 1: run_experiment
# ---------------------------------------------------------------------------

@celery_app.task(
    base=BaseIVETask,
    bind=True,
    name="ive.workers.tasks.run_experiment",
    max_retries=1,
    default_retry_delay=60,
    queue="analysis",
)
def run_experiment(
    self: Task,
    experiment_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute the full 4-phase IVE pipeline for one experiment.

    Phases
    ------
    1. understand  — data profiling + validation + preprocessing
    2. model       — train Linear + XGBoost via k-fold CV, collect residuals
    3. detect      — subgroup / cluster / SHAP / temporal analysis
    4. construct   — synthesise, bootstrap-validate, and explain latent variables

    Each phase updates the DB row so the polling WebSocket reflects live state.

    Args:
        experiment_id: String UUID of the experiment row.
        config:        Serialised :class:`ExperimentConfig` dict.

    Returns:
        Summary dict: experiment_id, status, elapsed_seconds.
    """
    import time

    t_start = time.perf_counter()
    log.info("task.run_experiment.start", experiment_id=experiment_id)

    def _progress(pct: int, stage: str) -> None:
        self.update_state(state="PROGRESS", meta={"progress": pct, "stage": stage})
        _update_experiment(
            experiment_id,
            "running",
            progress_pct=pct,
            current_stage=stage,
        )

    try:
        # ── Fetch experiment row from DB ─────────────────────────────
        exp = _get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found in DB.")

        _update_experiment(experiment_id, "running", progress_pct=0, current_stage="starting")

        # ── Phase 1: Understand ──────────────────────────────────────
        _progress(5, "understand")
        log.info("task.phase.understand", experiment_id=experiment_id)
        # TODO: from ive.core.phase_understand import run_understand_phase
        # profile, validated_df = run_understand_phase(exp["file_path"], config)
        _progress(20, "understand")

        # ── Phase 2: Model ───────────────────────────────────────────
        _progress(25, "model")
        log.info("task.phase.model", experiment_id=experiment_id)
        # TODO: from ive.core.phase_model import run_model_phase
        # residuals, trained_models = run_model_phase(validated_df, config)
        _progress(55, "model")

        # ── Phase 3: Detect ──────────────────────────────────────────
        _progress(58, "detect")
        log.info("task.phase.detect", experiment_id=experiment_id)
        # TODO: from ive.core.phase_detect import run_detect_phase
        # patterns = run_detect_phase(residuals, validated_df, config)
        _progress(80, "detect")

        # ── Phase 4: Construct ───────────────────────────────────────
        _progress(83, "construct")
        log.info("task.phase.construct", experiment_id=experiment_id)
        # TODO: from ive.core.phase_construct import run_construct_phase
        # latent_vars = run_construct_phase(patterns, config)
        # _store_latent_variables(experiment_id, latent_vars)
        n_latent_variables = 0  # Will be len(latent_vars) once implemented
        _progress(98, "construct")

        # ── Done ─────────────────────────────────────────────────────
        _update_experiment(experiment_id, "completed")
        elapsed = round(time.perf_counter() - t_start, 2)
        log.info(
            "task.run_experiment.done",
            experiment_id=experiment_id,
            elapsed_seconds=elapsed,
            n_latent_variables=n_latent_variables,
        )
        return {
            "experiment_id": experiment_id,
            "status": "completed",
            "elapsed_seconds": elapsed,
            "n_latent_variables": n_latent_variables,
        }

    except Exception as exc:
        log.error(
            "task.run_experiment.error",
            experiment_id=experiment_id,
            error=str(exc),
            exc_info=True,
        )
        _update_experiment(experiment_id, "failed", error_message=str(exc))
        raise self.retry(exc=exc) if self.request.retries < self.max_retries else exc


# ---------------------------------------------------------------------------
# Task 2: profile_dataset
# ---------------------------------------------------------------------------

@celery_app.task(
    name="ive.workers.tasks.profile_dataset",
    max_retries=1,
    default_retry_delay=30,
    queue="analysis",
)
def profile_dataset(dataset_id: str, file_path: str) -> dict[str, Any]:
    """Profile an uploaded dataset and update its DB record.

    Called automatically after a successful upload.

    Steps:
        1. Load file (CSV or Parquet) into a Pandas DataFrame.
        2. Run :class:`~ive.data.profiler.DataProfiler` → quality metrics.
        3. Save profile JSON to the artifact store.
        4. Update ``schema_json`` on the ``Dataset`` DB row.

    Args:
        dataset_id: String UUID of the dataset row.
        file_path:  Absolute filesystem path to the raw file.

    Returns:
        Dict: dataset_id, row_count, col_count, quality_score, status.
    """
    import json

    import pandas as pd

    log.info("task.profile_dataset.start", dataset_id=dataset_id, file_path=file_path)

    try:
        # ── Load data ────────────────────────────────────────────────
        if file_path.endswith((".parquet", ".pq")):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        # ── Fetch target_column from DB ───────────────────────────────
        target_col = "target"  # default fallback
        conn = _get_sync_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT target_column, schema_json FROM datasets WHERE id = %s::uuid",
                (dataset_id,),
            )
            row = cur.fetchone()
            if row:
                target_col = row[0]
                existing_schema: dict = row[1] or {}
        finally:
            conn.close()

        # ── Profile ───────────────────────────────────────────────────
        from ive.data.profiler import DataProfiler
        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=target_col, dataset_id=dataset_id)

        # ── Merge into schema_json ────────────────────────────────────
        existing_schema.update(
            {
                "quality_score": profile.quality_score,
                "quality_issues": [qi.model_dump() for qi in profile.quality_issues],
                "recommendations": profile.recommendations,
                "top_correlations": [cp.model_dump() for cp in profile.top_correlations],
                "memory_usage_mb": profile.memory_usage_mb,
            }
        )

        # ── Update DB ─────────────────────────────────────────────────
        conn = _get_sync_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE datasets
                SET schema_json = %s, row_count = %s, col_count = %s
                WHERE id = %s::uuid
                """,
                (json.dumps(existing_schema), profile.row_count, profile.col_count, dataset_id),
            )
            conn.commit()
        finally:
            conn.close()

        log.info(
            "task.profile_dataset.done",
            dataset_id=dataset_id,
            rows=profile.row_count,
            quality_score=profile.quality_score,
        )
        return {
            "dataset_id": dataset_id,
            "row_count": profile.row_count,
            "col_count": profile.col_count,
            "quality_score": profile.quality_score,
            "status": "profiled",
        }

    except Exception as exc:
        log.error("task.profile_dataset.error", dataset_id=dataset_id, error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Task 3: cancel_experiment
# ---------------------------------------------------------------------------

@celery_app.task(
    name="ive.workers.tasks.cancel_experiment",
    queue="high_priority",
)
def cancel_experiment(task_id: str, experiment_id: str) -> dict[str, Any]:
    """Revoke a running Celery task and mark the experiment as cancelled.

    Args:
        task_id:       Celery task ID to revoke (sends SIGTERM).
        experiment_id: String UUID of the experiment to update.

    Returns:
        Dict: experiment_id, status.
    """
    log.info("task.cancel", task_id=task_id, experiment_id=experiment_id)

    # Revoke the target task
    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")

    # Update DB
    try:
        _update_experiment(experiment_id, "cancelled")
    except Exception as exc:
        log.error("task.cancel.db_error", experiment_id=experiment_id, error=str(exc))

    return {"experiment_id": experiment_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# Task 4: health_check_task
# ---------------------------------------------------------------------------

@celery_app.task(
    name="ive.workers.tasks.health_check_task",
    queue="default",
)
def health_check_task() -> dict[str, Any]:
    """Verify that at least one Celery worker is running.

    Used by the ``/health/ready`` endpoint and monitoring scripts.

    Returns:
        Dict with ``status`` and ``timestamp``.
    """
    return {
        "status": "worker_healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
