"""
Integration tests — Background Job Processing.

Tests that Celery tasks can be enqueued and execute correctly using
a test Celery worker with the task_always_eager=True setting.
"""

from __future__ import annotations

import uuid

import pytest


@pytest.fixture(autouse=True)
def celery_eager_mode(settings):
    """Run Celery tasks synchronously (eager) during tests."""
    from ive.workers.celery_app import celery_app
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
    yield
    celery_app.conf.update(task_always_eager=False)


@pytest.fixture
def settings(override_settings):
    """Ensure settings override is active for job tests."""
    return None


@pytest.mark.integration
def test_profile_dataset_task_completes(tmp_path) -> None:
    """
    profile_dataset task should complete without raising in eager mode.

    TODO (once implemented):
        - Write a small CSV to tmp_path
        - Call profile_dataset.delay(dataset_id, csv_path)
        - Assert result.status == 'profiled'
    """
    pytest.skip("profile_dataset not yet implemented — skipping")


@pytest.mark.integration
def test_run_experiment_task_completes(tmp_path) -> None:
    """
    run_experiment task should complete in eager mode and return a result dict.

    TODO (once implemented):
        - Write CSV, create DB records
        - Call run_experiment.delay(experiment_id, config, data_path)
        - Assert result['status'] == 'completed'
        - Assert result['n_latent_variables'] >= 0
    """
    from ive.workers.tasks import run_experiment

    experiment_id = str(uuid.uuid4())
    config = {"target_column": "y", "model_types": ["linear"], "cv_folds": 3}
    result = run_experiment(experiment_id, config, "/tmp/nonexistent.csv")
    # Placeholder task returns a dict even without a real file
    assert isinstance(result, dict)
    assert result["experiment_id"] == experiment_id


@pytest.mark.integration
def test_cancel_experiment_task_returns_cancelled() -> None:
    """cancel_experiment should return status=cancelled."""
    from ive.workers.tasks import cancel_experiment

    task_id = "fake-task-id"
    experiment_id = str(uuid.uuid4())
    result = cancel_experiment(task_id, experiment_id)
    assert result["experiment_id"] == experiment_id
    assert result["status"] == "cancelled"
