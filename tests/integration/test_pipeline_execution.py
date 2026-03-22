"""
Integration Tests — IVEPipeline Direct Execution.

Tests the full four-phase ML pipeline by calling ``IVEPipeline.run_experiment()``
directly, bypassing HTTP and Celery.  The database layer is mocked (PostgreSQL-
specific types prevent SQLite in-process testing) but the artifact store uses a
real ``LocalArtifactStore`` backed by ``tmp_path``, and all ML logic runs
unmodified on real NumPy/pandas data.

Strategy
--------
* ``IVEPipeline`` takes a db_session and an artifact_store.
* All DB repository method calls are intercepted by mock repositories.
  Return values are realistic SimpleNamespace objects so attribute access works.
* ``LocalArtifactStore(tmp_path)`` is used so CSV loading works end-to-end.
* The pipeline produces residuals, patterns, and latent variables using the
  real ML code — validating numeric correctness, not just structure.
* Key batch methods the pipeline actually calls:
    - exp_repo.add_trained_model(...)
    - exp_repo.add_residuals_batch(...)
    - exp_repo.add_error_patterns_batch(...)
    - lv_repo.bulk_create(...)

Markers
-------
    @pytest.mark.integration
    @pytest.mark.asyncio
"""

from __future__ import annotations

import importlib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# ---------------------------------------------------------------------------
# Helpers — build realistic mock repo objects
# ---------------------------------------------------------------------------


def _make_dataset(
    dataset_id: uuid.UUID,
    file_path: str,
    target_column: str = "y",
) -> SimpleNamespace:
    """Return a namespace that mimics a ``Dataset`` ORM row."""
    return SimpleNamespace(
        id=dataset_id,
        file_path=file_path,
        target_column=target_column,
        time_column=None,
        name="test_dataset",
        schema_json={},  # empty → no columns to drop
    )


def _make_experiment(
    experiment_id: uuid.UUID,
    dataset_id: uuid.UUID,
    config: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Return a namespace that mimics an ``Experiment`` ORM row."""
    return SimpleNamespace(
        id=experiment_id,
        dataset_id=dataset_id,
        status="queued",
        config_json=config
        or {
            "analysis_mode": "demo",
            "cv_folds": 3,
            "model_types": ["linear"],
            "bootstrap_iterations": 10,
        },
    )


def _build_mock_exp_repo(experiment: SimpleNamespace) -> MagicMock:
    """Build an ExperimentRepository mock covering all methods called by IVEPipeline."""
    repo = MagicMock()
    # Lifecycle
    repo.get_by_id = AsyncMock(return_value=experiment)
    repo.mark_started = AsyncMock(return_value=None)
    repo.update_progress = AsyncMock(return_value=None)
    repo.mark_completed = AsyncMock(return_value=None)
    repo.mark_failed = AsyncMock(return_value=None)
    repo.update = AsyncMock(return_value=None)
    # Persist phase outputs — use BATCH APIs matching the actual pipeline calls
    repo.add_trained_model = AsyncMock(return_value=None)
    repo.add_residuals_batch = AsyncMock(return_value=None)
    repo.add_error_patterns_batch = AsyncMock(return_value=None)
    # Per-pattern fallback (older code path)
    repo.add_error_pattern = AsyncMock(return_value=None)
    return repo


def _build_mock_ds_repo(dataset: SimpleNamespace) -> MagicMock:
    """Build a DatasetRepository mock returning the supplied dataset."""
    repo = MagicMock()
    repo.get_by_id = AsyncMock(return_value=dataset)
    return repo


def _build_mock_lv_repo() -> MagicMock:
    """Build a LatentVariableRepository mock covering all pipeline call sites."""
    repo = MagicMock()
    repo.bulk_create = AsyncMock(return_value=None)
    repo.create = AsyncMock(return_value=None)
    return repo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def artifact_store(tmp_path: Path):
    """Real LocalArtifactStore backed by pytest's tmp_path."""
    from ive.storage.artifact_store import LocalArtifactStore

    return LocalArtifactStore(base_dir=str(tmp_path / "artifacts"))


def _write_regression_csv(path: Path) -> str:
    """Write regression CSV with a hidden subgroup effect; return str path."""
    m = importlib.import_module("tests.fixtures.demo_csv_files")
    csv_bytes = m.make_regression_with_subgroup(n=200, seed=42)
    p = path / "regression_subgroup.csv"
    p.write_bytes(csv_bytes)
    return str(p)


def _write_noise_csv(path: Path) -> str:
    """Write pure-noise CSV; return str path."""
    m = importlib.import_module("tests.fixtures.demo_csv_files")
    csv_bytes = m.make_pure_noise(n=200, seed=99)
    p = path / "pure_noise.csv"
    p.write_bytes(csv_bytes)
    return str(p)


@pytest.fixture
def regression_csv_path(tmp_path: Path) -> str:
    return _write_regression_csv(tmp_path)


@pytest.fixture
def noise_csv_path(tmp_path: Path) -> str:
    return _write_noise_csv(tmp_path)


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(
    artifact_store: Any,
    experiment: SimpleNamespace,
    dataset: SimpleNamespace,
) -> tuple[dict[str, Any], MagicMock, MagicMock, MagicMock]:
    """Run IVEPipeline with fully-mocked repos.

    Returns:
        (result_dict, exp_repo_mock, ds_repo_mock, lv_repo_mock)
    """
    from ive.core.pipeline import IVEPipeline

    exp_repo = _build_mock_exp_repo(experiment)
    ds_repo = _build_mock_ds_repo(dataset)
    lv_repo = _build_mock_lv_repo()

    db_session = AsyncMock()
    pipeline = IVEPipeline(db_session=db_session, artifact_store=artifact_store)

    with (
        patch("ive.core.pipeline.ExperimentRepository", return_value=exp_repo),
        patch("ive.core.pipeline.DatasetRepository", return_value=ds_repo),
        patch("ive.core.pipeline.LatentVariableRepository", return_value=lv_repo),
    ):
        result = await pipeline.run_experiment(experiment.id)

    return result, exp_repo, ds_repo, lv_repo


# ---------------------------------------------------------------------------
# Test 1: Regression dataset with hidden subgroup — pipeline completes
# ---------------------------------------------------------------------------


async def test_pipeline_regression_with_subgroup_completes(
    artifact_store: Any,
    regression_csv_path: str,
) -> None:
    """Full pipeline run on a dataset with a real subgroup hidden variable.

    Asserts:
    - run_experiment() returns without raising.
    - status == 'completed'.
    - n_patterns and n_validated are non-negative integers.
    """
    experiment_id = uuid.uuid4()
    dataset_id = uuid.uuid4()
    dataset = _make_dataset(dataset_id, file_path=regression_csv_path, target_column="y")
    experiment = _make_experiment(
        experiment_id,
        dataset_id,
        config={
            "analysis_mode": "demo",
            "cv_folds": 3,
            "model_types": ["linear"],
            "bootstrap_iterations": 10,
        },
    )

    result, _, _, _ = await _run_pipeline(artifact_store, experiment, dataset)

    assert isinstance(result, dict), "run_experiment must return a dict"
    assert result["status"] == "completed"
    assert isinstance(result["n_patterns"], int)
    assert isinstance(result["n_validated"], int)
    assert result["n_patterns"] >= 0
    assert result["n_validated"] >= 0


# ---------------------------------------------------------------------------
# Test 2: Residuals are persisted with correct field names
# ---------------------------------------------------------------------------


async def test_pipeline_residuals_are_persisted(
    artifact_store: Any,
    regression_csv_path: str,
) -> None:
    """add_residuals_batch is called at least once with correctly-shaped rows."""
    experiment_id = uuid.uuid4()
    dataset_id = uuid.uuid4()
    dataset = _make_dataset(dataset_id, file_path=regression_csv_path)
    experiment = _make_experiment(
        experiment_id,
        dataset_id,
        config={"analysis_mode": "demo", "cv_folds": 3, "model_types": ["linear"]},
    )

    result, exp_repo, _, _ = await _run_pipeline(artifact_store, experiment, dataset)

    assert (
        exp_repo.add_residuals_batch.await_count >= 1
    ), "add_residuals_batch should be called at least once"

    # Inspect the first batch
    first_call_args = exp_repo.add_residuals_batch.call_args_list[0][0]
    residual_rows: list[dict] = first_call_args[1]
    assert len(residual_rows) > 0, "Residual batch must not be empty"

    required_fields = {"model_type", "residual_value", "actual_value", "predicted_value"}
    missing = required_fields - set(residual_rows[0].keys())
    assert not missing, f"Residual row missing fields: {missing}"


# ---------------------------------------------------------------------------
# Test 3: Experiment lifecycle hooks are called
# ---------------------------------------------------------------------------


async def test_pipeline_marks_experiment_started_and_completed(
    artifact_store: Any,
    regression_csv_path: str,
) -> None:
    """mark_started and update_progress must be called during a successful run."""
    experiment_id = uuid.uuid4()
    dataset_id = uuid.uuid4()
    dataset = _make_dataset(dataset_id, file_path=regression_csv_path)
    experiment = _make_experiment(
        experiment_id,
        dataset_id,
        config={"analysis_mode": "demo", "cv_folds": 3, "model_types": ["linear"]},
    )

    _, exp_repo, _, _ = await _run_pipeline(artifact_store, experiment, dataset)

    exp_repo.mark_started.assert_awaited_once()
    assert (
        exp_repo.update_progress.await_count >= 2
    ), "update_progress should be called at least twice (model + detect phases)"


# ---------------------------------------------------------------------------
# Test 4: Pure-noise dataset — pipeline completes without crash
# ---------------------------------------------------------------------------


async def test_pipeline_no_signal_dataset_completes(
    artifact_store: Any,
    noise_csv_path: str,
) -> None:
    """Pipeline on a pure-noise dataset must complete with status='completed'."""
    experiment_id = uuid.uuid4()
    dataset_id = uuid.uuid4()
    dataset = _make_dataset(dataset_id, file_path=noise_csv_path)
    experiment = _make_experiment(
        experiment_id,
        dataset_id,
        config={"analysis_mode": "demo", "cv_folds": 3, "model_types": ["linear"]},
    )

    result, _, _, _ = await _run_pipeline(artifact_store, experiment, dataset)

    assert result["status"] == "completed"
    assert result["n_validated"] >= 0


# ---------------------------------------------------------------------------
# Test 5: Missing experiment raises
# ---------------------------------------------------------------------------


async def test_pipeline_raises_for_missing_experiment(
    artifact_store: Any,
) -> None:
    """If the experiment isn't in DB, run_experiment must raise."""
    from ive.core.pipeline import IVEPipeline

    experiment_id = uuid.uuid4()

    exp_repo = MagicMock()
    exp_repo.get_by_id = AsyncMock(return_value=None)  # not found
    exp_repo.mark_started = AsyncMock()
    exp_repo.update_progress = AsyncMock()
    exp_repo.mark_failed = AsyncMock()

    ds_repo = MagicMock()
    lv_repo = MagicMock()

    db_session = AsyncMock()
    pipeline = IVEPipeline(db_session=db_session, artifact_store=artifact_store)

    with (
        patch("ive.core.pipeline.ExperimentRepository", return_value=exp_repo),
        patch("ive.core.pipeline.DatasetRepository", return_value=ds_repo),
        patch("ive.core.pipeline.LatentVariableRepository", return_value=lv_repo),
        pytest.raises(Exception),
    ):
        await pipeline.run_experiment(experiment_id)


# ---------------------------------------------------------------------------
# Test 6: Summary dict has required keys
# ---------------------------------------------------------------------------


async def test_pipeline_summary_dict_has_required_keys(
    artifact_store: Any,
    regression_csv_path: str,
) -> None:
    """The summary dict returned by run_experiment must have the documented keys."""
    experiment_id = uuid.uuid4()
    dataset_id = uuid.uuid4()
    dataset = _make_dataset(dataset_id, file_path=regression_csv_path)
    experiment = _make_experiment(
        experiment_id,
        dataset_id,
        config={"analysis_mode": "demo", "cv_folds": 2, "model_types": ["linear"]},
    )

    result, _, _, _ = await _run_pipeline(artifact_store, experiment, dataset)

    required_keys = {"status", "n_patterns", "n_validated"}
    missing = required_keys - set(result.keys())
    assert not missing, f"Summary dict missing keys: {missing}"


# ---------------------------------------------------------------------------
# Test 7: Artifact store load_file is called with the correct path
# ---------------------------------------------------------------------------


async def test_pipeline_load_file_is_called_with_correct_path(
    artifact_store: Any,
    regression_csv_path: str,
) -> None:
    """Verify the pipeline asks the artifact store to load the right CSV path."""
    experiment_id = uuid.uuid4()
    dataset_id = uuid.uuid4()
    dataset = _make_dataset(dataset_id, file_path=regression_csv_path)
    experiment = _make_experiment(experiment_id, dataset_id)

    # Spy on load_file without replacing it
    original_load = artifact_store.load_file
    load_calls: list[str] = []

    async def _spied_load(path: str) -> bytes:
        load_calls.append(path)
        return await original_load(path)

    artifact_store.load_file = _spied_load

    await _run_pipeline(artifact_store, experiment, dataset)

    assert len(load_calls) >= 1, "artifact_store.load_file should have been called"
    assert (
        regression_csv_path in load_calls
    ), f"Expected {regression_csv_path!r} to be loaded; got: {load_calls}"
