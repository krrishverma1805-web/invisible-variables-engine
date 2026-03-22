"""
Integration tests — Background Job Processing.

Tests that Celery tasks behave correctly.  The ``cancel_experiment`` task is
tested without touching a real DB (it uses psycopg2).  The
``run_experiment`` task requires a live DB + broker and is therefore skipped
when those services are absent.

Design
------
- ``task_always_eager=True`` is NOT used here because the real ``run_experiment``
  task depends on DB + artifact store.  We validate the tasks we can test
  without full infra (cancel, health_check), and explicitly skip the rest.
- The ``override_settings`` fixture no longer exists — env config is controlled
  via ``_test_env`` in conftest.py.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helper — check if celery can reach its broker
# ---------------------------------------------------------------------------


def _broker_available() -> bool:
    """Return True when the Celery broker (Redis) is reachable."""
    try:
        from ive.workers.celery_app import celery_app

        celery_app.control.ping(timeout=1.0)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Task 1: profile_dataset
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_profile_dataset_task_skipped_without_broker() -> None:
    """profile_dataset task requires a real DB + filesystem; skip if no broker."""
    pytest.skip(
        "profile_dataset requires a live PostgreSQL DB and filesystem "
        "not available in headless CI without Docker Compose"
    )


# ---------------------------------------------------------------------------
# Task 2: run_experiment
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_run_experiment_task_raises_without_db(tmp_path) -> None:
    """run_experiment (2-arg: experiment_id, config) must fail fast when DB is
    absent rather than silently returning success.

    This test verifies the task interface: it takes exactly 2 arguments and
    raises (ValueError/psycopg2 error) when the experiment cannot be fetched.
    """
    from ive.workers.tasks import run_experiment

    experiment_id = str(uuid.uuid4())
    config = {"analysis_mode": "demo", "model_types": ["linear"], "cv_folds": 3}

    # The task will attempt a psycopg2 connection.  Without a DB it raises.
    # We confirm it raises rather than silently returning a bad result.
    with pytest.raises((ValueError, RuntimeError, OSError, ConnectionError)):
        run_experiment(experiment_id, config)


# ---------------------------------------------------------------------------
# Task 3: cancel_experiment
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cancel_experiment_returns_cancelled_status() -> None:
    """cancel_experiment should return {status: 'cancelled'} without a real broker.

    The revoke call is mocked so no broker is needed; DB update is also mocked
    since psycopg2 would fail without a live PostgreSQL instance.
    """
    from ive.workers.tasks import cancel_experiment

    task_id = "fake-task-id-abc"
    experiment_id = str(uuid.uuid4())

    with (
        patch("ive.workers.tasks._update_experiment"),
        patch("ive.workers.celery_app.celery_app.control.revoke"),
    ):
        result = cancel_experiment(task_id, experiment_id)

    assert result["experiment_id"] == experiment_id
    assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Task 4: health_check_task
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_health_check_task_returns_worker_healthy() -> None:
    """health_check_task is a trivial task with no I/O — always runnable."""
    from ive.workers.tasks import health_check_task

    result = health_check_task()
    assert result["status"] == "worker_healthy"
    assert "timestamp" in result
