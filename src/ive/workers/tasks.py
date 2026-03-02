"""
Celery Task Definitions.

Defines all background tasks executed by the IVE Celery worker pool.

Tasks:
    run_experiment    — Executes the full 4-phase IVE pipeline
    profile_dataset   — Profiles an uploaded dataset asynchronously
    cancel_experiment — Revokes a running experiment task
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from celery import Task
from tenacity import retry, stop_after_attempt, wait_exponential

from ive.workers.celery_app import celery_app

log = structlog.get_logger(__name__)


class BaseIVETask(Task):
    """
    Base class for all IVE Celery tasks.

    Provides:
        - Structured error logging on failure
        - Automatic experiment status update to 'failed' on error

    TODO:
        - Override on_failure() to update DB experiment status
        - Override on_retry() to log retry attempts
    """

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: Any) -> None:
        """Handle task failure — update experiment status in DB."""
        log.error(
            "ive.task.failed",
            task_id=task_id,
            error=str(exc),
            task_name=self.name,
        )
        # TODO: Update experiment status to 'failed' in DB
        # experiment_id = kwargs.get('experiment_id')
        # if experiment_id:
        #     asyncio.run(update_experiment_failed(experiment_id, str(exc)))
        super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(
    base=BaseIVETask,
    bind=True,
    name="ive.workers.tasks.run_experiment",
    max_retries=2,
    default_retry_delay=60,
    queue="analysis",
)
def run_experiment(
    self: Task,
    experiment_id: str,
    config: dict[str, Any],
    data_path: str,
) -> dict[str, Any]:
    """
    Execute the full IVE pipeline for an experiment.

    This is the main long-running task. It:
        1. Initialises the IVEEngine
        2. Runs all four phases
        3. Persists results to the database
        4. Publishes final status to Redis pub/sub

    Args:
        experiment_id: String UUID of the experiment.
        config: Serialised ExperimentConfig dict.
        data_path: Path to the dataset artifact.

    Returns:
        Dict summary of results (number of latent variables, elapsed time).

    TODO:
        - from ive.core.engine import IVEEngine
        - from ive.api.v1.schemas.experiment_schemas import ExperimentConfig
        - engine = IVEEngine()
        - exp_config = ExperimentConfig(**config)
        - result = asyncio.run(engine.run(uuid.UUID(experiment_id), exp_config, data_path))
        - Persist result.latent_variables via LatentVariableRepo.bulk_create()
        - Update experiment status to 'completed'
    """
    log.info("ive.task.run_experiment.start", experiment_id=experiment_id)

    # TODO: Implement pipeline execution (see docstring)
    # For now, return a placeholder result
    return {
        "experiment_id": experiment_id,
        "n_latent_variables": 0,
        "elapsed_seconds": 0.0,
        "status": "completed",
    }


@celery_app.task(
    base=BaseIVETask,
    bind=True,
    name="ive.workers.tasks.profile_dataset",
    max_retries=1,
    queue="analysis",
)
def profile_dataset(self: Task, dataset_id: str, file_path: str) -> dict[str, Any]:
    """
    Profile an uploaded dataset and update the dataset record.

    Steps:
        1. Load dataset from file_path
        2. Run DataProfiler.profile()
        3. Save profile JSON to artifact store
        4. Update dataset record (row_count, column_count, status='profiled')

    TODO:
        - from ive.data.ingestion import DataIngestion
        - from ive.data.profiler import DataProfiler
        - df = DataIngestion().load(file_path)
        - profile = DataProfiler().profile(df)
        - Save profile artifact and update DB
    """
    log.info("ive.task.profile_dataset.start", dataset_id=dataset_id)
    # TODO: Implement dataset profiling
    return {"dataset_id": dataset_id, "status": "profiled"}


@celery_app.task(
    name="ive.workers.tasks.cancel_experiment",
    queue="high_priority",
)
def cancel_experiment(task_id: str, experiment_id: str) -> dict[str, Any]:
    """
    Revoke a running experiment task.

    TODO:
        - celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')
        - Update experiment status to 'cancelled' in DB
    """
    log.info("ive.task.cancel", task_id=task_id, experiment_id=experiment_id)
    celery_app.control.revoke(task_id, terminate=True)
    return {"experiment_id": experiment_id, "status": "cancelled"}
