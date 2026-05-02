"""
Celery Task Definitions — Invisible Variables Engine.

All tasks are **synchronous** (Celery runs them in a regular thread/process).
The main experiment task bridges into async land via :func:`asyncio.run` to
use the async SQLAlchemy session required by ``IVEPipeline``.  The remaining
tasks (``profile_dataset``, ``cancel_experiment``, ``health_check_task``)
use psycopg2 directly for lightweight DB updates that don't need ORM features.

Tasks
-----
run_experiment      — Full 4-phase IVE pipeline
profile_dataset     — Post-upload dataset profiling (triggered after upload)
cancel_experiment   — Revoke a running task + mark DB as cancelled
health_check_task   — Worker liveness probe (returns immediately)

Progress reporting
------------------
``run_experiment`` calls ``self.update_state(state='PROGRESS', meta={...})``
so the WebSocket endpoint can poll ``celery_app.AsyncResult(task_id).info``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from celery import Task

from ive.workers.celery_app import celery_app

log = structlog.get_logger("ive.workers.tasks")


# ---------------------------------------------------------------------------
# Sync DB helper  (uses psycopg2 — safe for Celery sync workers)
# ---------------------------------------------------------------------------


def _get_sync_conn() -> Any:
    """Return a psycopg2 connection using the sync database URL."""
    import psycopg2

    from ive.config import get_settings

    settings = get_settings()
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
    now = datetime.now(UTC)
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


def _record_llm_task_id(experiment_id: str, llm_task_id: str) -> None:
    """Persist the chained LLM task's Celery ID into ``experiments.llm_task_id``.

    Called from ``run_experiment`` after :func:`generate_llm_explanations`
    is queued, so :func:`cancel_experiment` can revoke the chained task
    on cooperative shutdown (per plan §171).  Best-effort — failures are
    logged but never propagate.
    """
    conn = _get_sync_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "UPDATE experiments SET llm_task_id = %s WHERE id = %s::uuid",
            (llm_task_id, experiment_id),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        log.warning(
            "task.llm_task_id.persist_failed",
            experiment_id=experiment_id,
            error=str(exc),
        )
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


class BaseIVETask(Task):  # type: ignore[misc]
    """Celery base task that marks experiments as ``failed`` on unhandled errors."""

    abstract = True

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
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
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
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
# Async pipeline bridge
# ---------------------------------------------------------------------------


async def _run_pipeline_async(experiment_id: str) -> dict[str, Any]:
    """Async helper that sets up a DB session and runs ``IVEPipeline``.

    This function is the bridge between the synchronous Celery task world
    and the async SQLAlchemy ORM used by ``IVEPipeline``.

    Args:
        experiment_id: String UUID of the experiment.

    Returns:
        Summary dict returned by :meth:`IVEPipeline.run_experiment`.
    """
    from ive.core.pipeline import IVEPipeline
    from ive.db.database import get_session, init_db
    from ive.storage.artifact_store import get_artifact_store

    store = get_artifact_store()
    await init_db()
    async with get_session() as session:
        pipeline = IVEPipeline(session, store)
        return await pipeline.run_experiment(UUID(experiment_id))
    return {}  # Fallback for MyPy


# ---------------------------------------------------------------------------
# Task 1: run_experiment
# ---------------------------------------------------------------------------


@celery_app.task(  # type: ignore[untyped-decorator]
    base=BaseIVETask,
    bind=True,
    name="ive.workers.tasks.run_experiment",
    max_retries=1,
    default_retry_delay=60,
    queue="analysis",
    acks_late=True,
)
def run_experiment(
    self: Task,
    experiment_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute the full 4-phase IVE pipeline for one experiment.

    Phases
    ------
    1. understand  — data loading, schema filtering, X/y split
    2. model       — train Linear + XGBoost via k-fold CV, collect OOF residuals
    3. detect      — subgroup KS-scan + HDBSCAN clustering
    4. construct   — synthesise, bootstrap-validate, and persist latent variables

    The Celery task is synchronous; it bridges to async code via
    :func:`asyncio.run`.  Progress is written to both the DB row and the
    Celery task state so the WebSocket polling endpoint stays up to date.

    Args:
        experiment_id: String UUID of the experiment row.
        config:        Serialised experiment config dict (informational — the
                       pipeline reads authoritative config from the DB row).

    Returns:
        Summary dict: experiment_id, status, elapsed_seconds, n_latent_variables.

    Raises:
        Exception: Any unhandled pipeline error.  The ``BaseIVETask.on_failure``
                   hook marks the experiment as ``"failed"`` automatically.
    """
    import time

    t_start = time.perf_counter()
    log.info("task.run_experiment.start", experiment_id=experiment_id)

    def _progress(pct: int, stage: str) -> None:
        self.update_state(state="PROGRESS", meta={"progress": pct, "stage": stage})
        _update_experiment(experiment_id, "running", progress_pct=pct, current_stage=stage)

    try:
        # Verify experiment exists before handing off to the async pipeline.
        exp = _get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found in DB.")

        _progress(0, "starting")
        log.info("task.pipeline.starting", experiment_id=experiment_id)

        # Delegate all four phases to the async IVEPipeline.
        # asyncio.run() creates a fresh event loop for this sync Celery context.
        result = asyncio.run(_run_pipeline_async(experiment_id))

        elapsed = round(float(time.perf_counter() - t_start), 2)
        n_latent = result.get("n_validated", 0)

        log.info(
            "task.run_experiment.done",
            experiment_id=experiment_id,
            elapsed_seconds=elapsed,
            n_latent_variables=n_latent,
        )

        # ── Chain LLM enrichment (per plan §A1) ──────────────────────────
        # Always queued — the task short-circuits to ``disabled`` when
        # LLM_EXPLANATIONS_ENABLED=false so explanation_status is
        # deterministic from the API surface.
        try:
            llm_async = generate_llm_explanations.apply_async(
                args=[experiment_id],
                countdown=2,
                headers={"request_id": str(self.request.id) if self.request.id else None},
            )
            _record_llm_task_id(experiment_id, str(llm_async.id))
        except Exception as exc:
            # Non-fatal: chaining failure should never fail the
            # experiment itself. Endpoints will fall back to rule-based.
            log.warning(
                "task.run_experiment.llm_chain_failed",
                experiment_id=experiment_id,
                error=str(exc),
            )

        return {
            "experiment_id": experiment_id,
            "status": "completed",
            "elapsed_seconds": elapsed,
            "n_latent_variables": n_latent,
            "n_patterns": result.get("n_patterns", 0),
        }

    except Exception as exc:
        log.error(
            "task.run_experiment.error",
            experiment_id=experiment_id,
            error=str(exc),
            exc_info=True,
        )
        _update_experiment(experiment_id, "failed", error_message=str(exc))
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        raise


# ---------------------------------------------------------------------------
# Task 2: profile_dataset
# ---------------------------------------------------------------------------


@celery_app.task(  # type: ignore[untyped-decorator]
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
        3. Merge results into ``schema_json`` on the ``Dataset`` DB row.

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
        if file_path.endswith((".parquet", ".pq")):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        target_col = "target"
        existing_schema: dict[str, Any] = {}
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
                existing_schema = row[1] or {}
        finally:
            conn.close()

        from ive.data.profiler import DataProfiler

        profiler = DataProfiler()
        profile = profiler.profile(df, target_column=target_col, dataset_id=dataset_id)

        existing_schema.update(
            {
                "quality_score": profile.quality_score,
                "quality_issues": [qi.model_dump() for qi in profile.quality_issues],
                "recommendations": profile.recommendations,
                "top_correlations": [cp.model_dump() for cp in profile.top_correlations],
                "memory_usage_mb": profile.memory_usage_mb,
            }
        )

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


@celery_app.task(  # type: ignore[untyped-decorator]
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

    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")

    try:
        _update_experiment(experiment_id, "cancelled")
    except Exception as exc:
        log.error("task.cancel.db_error", experiment_id=experiment_id, error=str(exc))

    return {"experiment_id": experiment_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# Task 4: health_check_task
# ---------------------------------------------------------------------------


@celery_app.task(  # type: ignore[untyped-decorator]
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
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Task 5: generate_llm_explanations  (per plan §A1, §103, §171)
# ---------------------------------------------------------------------------

try:
    from celery.contrib.abortable import AbortableTask

    _LLM_BASE_TASK = AbortableTask
except ImportError:  # pragma: no cover - older celery without abortable
    _LLM_BASE_TASK = BaseIVETask


@celery_app.task(  # type: ignore[untyped-decorator]
    base=_LLM_BASE_TASK,
    bind=True,
    name="ive.workers.tasks.generate_llm_explanations",
    max_retries=2,
    default_retry_delay=30,
    queue="analysis",
    acks_late=True,
)
def generate_llm_explanations(self: Task, experiment_id: str) -> dict[str, Any]:
    """Post-pipeline task that fills the ``llm_*`` columns on LVs and Experiment.

    Always tail-of-pipeline:  ``run_experiment`` chains to this task only
    when ``LLM_EXPLANATIONS_ENABLED=true``. With the flag off we still
    queue the task — it short-circuits to mark every row ``disabled``
    so endpoints can deterministically read explanation status.

    Cancellation: uses ``AbortableTask.is_aborted()`` (per plan §171). The
    async core polls this between LVs; in-flight HTTP completes within
    ``groq_timeout_seconds``.

    Args:
        experiment_id: String UUID of the experiment.

    Returns:
        Summary dict from :class:`EnrichmentResult`.
    """
    from ive.workers.llm_enrichment import run_llm_enrichment_async

    log.info("task.llm_enrichment.start", experiment_id=experiment_id)

    def _check_aborted() -> bool:
        return bool(getattr(self, "is_aborted", lambda: False)())

    try:
        result = asyncio.run(
            run_llm_enrichment_async(experiment_id, is_aborted=_check_aborted)
        )
    except Exception as exc:
        log.error(
            "task.llm_enrichment.error",
            experiment_id=experiment_id,
            error=str(exc),
            exc_info=True,
        )
        raise

    log.info(
        "task.llm_enrichment.done",
        experiment_id=experiment_id,
        status=result.status,
        n_total=result.n_lv_total,
        n_ready=result.n_lv_ready,
        n_disabled=result.n_lv_disabled,
        n_failed=result.n_lv_failed,
    )
    return {
        "experiment_id": result.experiment_id,
        "status": result.status,
        "n_lv_total": result.n_lv_total,
        "n_lv_ready": result.n_lv_ready,
        "n_lv_disabled": result.n_lv_disabled,
        "n_lv_failed": result.n_lv_failed,
    }


# ---------------------------------------------------------------------------
# FPR sentinel (Phase C4 — beat-scheduled nightly @ 02:30 UTC)
# ---------------------------------------------------------------------------


@celery_app.task(  # type: ignore[untyped-decorator]
    name="ive.workers.tasks.fpr_sentinel_run",
    queue="default",
    acks_late=True,
)
def fpr_sentinel_run() -> dict[str, Any]:
    """Run the synthetic-noise FPR sentinel and emit Prometheus + structlog.

    Per plan §C4 + §190: 20 seeds, alert if Clopper-Pearson upper-95% CI
    on observed FPR exceeds 7%. Pure-function core lives in
    :func:`ive.observability.fpr_sentinel.run_sentinel` for unit testability.
    """
    from ive.observability.fpr_sentinel import run_sentinel
    from ive.observability.metrics import record_fpr_sentinel

    log.info("task.fpr_sentinel.start")
    result = run_sentinel()

    record_fpr_sentinel(fpr=result.empirical_fpr, status=result.status)

    log.info(
        "task.fpr_sentinel.done",
        n_runs=result.n_runs,
        empirical_fpr=result.empirical_fpr,
        upper_95_ci=result.upper_95_ci,
        threshold=result.threshold,
        status=result.status,
    )
    return {
        "n_runs": result.n_runs,
        "n_false_positive_runs": result.n_false_positive_runs,
        "empirical_fpr": result.empirical_fpr,
        "upper_95_ci": result.upper_95_ci,
        "threshold": result.threshold,
        "status": result.status,
    }
