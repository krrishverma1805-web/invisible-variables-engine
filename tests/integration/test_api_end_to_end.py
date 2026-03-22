"""
Integration Tests — API End-to-End Workflow.

Tests the complete HTTP workflow:
    dataset upload → experiment creation → listing → detail → export endpoints.

Strategy
--------
* Uses the real FastAPI TestClient (full middleware stack: auth, rate-limit, logging).
* Celery dispatch is mocked at the import site so no broker is needed.
* DB and artifact store calls go through the real stack — these tests require
  the Docker Compose services (postgres, redis) to be running.  If the
  connection is unavailable the tests skip themselves gracefully.

Uniqueness
----------
* The real API rejects duplicate file uploads (same SHA-256 checksum) with 400.
* Every upload call uses ``_unique_csv_bytes()`` which appends a UUID comment to
  the CSV header, guaranteeing a fresh checksum each time.  This keeps every test
  independent and deterministic irrespective of run count or run order.
"""

from __future__ import annotations

import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _services_available(client: TestClient) -> bool:
    """Return True when both the API and DB/Redis are reachable."""
    try:
        resp = client.get("/api/v1/health/ready")
        return resp.status_code == 200
    except Exception:
        return False


def _unique_csv_bytes(base_csv_bytes: bytes) -> bytes:
    """Return CSV bytes with a UUID-derived epsilon injected into the first data cell.

    The column structure, types, and row count are completely preserved.  Only the
    value of the first numeric column's first row changes by a tiny amount (~1e-9),
    which is enough to produce a different SHA-256 checksum on every call and bypass
    the API's duplicate-file detection gate.

    This is the correct approach: prepending a comment line would make pandas treat
    the comment as the header, destroying the schema and causing 422 errors.
    """
    import io as _io

    import pandas as pd

    df = pd.read_csv(_io.BytesIO(base_csv_bytes))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        # UUID int gives us ~122 bits of entropy; mod 1_000_000 keeps epsilon tiny
        epsilon = (uuid.uuid4().int % 1_000_000) * 1e-12
        df.at[0, num_cols[0]] = float(df.at[0, num_cols[0]]) + epsilon
    buf = _io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _fake_celery_task(task_id: str = "fake-celery-task-id") -> MagicMock:
    """Return a mock that mimics a Celery AsyncResult returned by .delay()."""
    task = MagicMock()
    task.id = task_id
    return task


def _upload_dataset(
    client: TestClient,
    headers: dict[str, str],
    csv_bytes: bytes,
    filename: str = "test.csv",
    target_column: str = "y",
    name: str | None = None,
) -> str:
    """Upload *csv_bytes* and return the created dataset UUID.

    Automatically makes the payload unique (no duplicate-checksum errors).
    Asserts 201 on success; raises AssertionError with full body on any other status.
    """
    form: dict[str, str] = {"target_column": target_column}
    if name:
        form["name"] = name

    resp = client.post(
        "/api/v1/datasets/",
        headers=headers,
        files={"file": (filename, io.BytesIO(_unique_csv_bytes(csv_bytes)), "text/csv")},
        data=form,
    )
    assert resp.status_code == 201, f"Dataset upload failed ({resp.status_code}): {resp.text}"
    return resp.json()["id"]


def _create_experiment(
    client: TestClient,
    headers: dict[str, str],
    dataset_id: str,
    config: dict | None = None,
    celery_task_id: str = "fixture-task",
) -> str:
    """Create a queued experiment with Celery mocked; return experiment UUID.

    Patches ive.workers.tasks.run_experiment — the object that the endpoint
    obtains via its local ``from ive.workers.tasks import run_experiment as
    celery_run`` statement.  Patching the module-level attribute in the
    endpoint module would raise AttributeError because the symbol is not
    present there at import time.
    """
    with patch(
        "ive.workers.tasks.run_experiment",
        autospec=False,
    ) as mock_task:
        mock_task.delay.return_value = _fake_celery_task(celery_task_id)
        resp = client.post(
            "/api/v1/experiments/",
            headers=headers,
            json={
                "dataset_id": dataset_id,
                "config": config or {"analysis_mode": "demo", "cv_folds": 2},
            },
        )
    assert resp.status_code == 201, f"Experiment creation failed ({resp.status_code}): {resp.text}"
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# 1. Health endpoint smoke tests
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """Basic liveness and readiness checks — no auth required."""

    def test_liveness_returns_200(self, api_client: TestClient) -> None:
        """GET /api/v1/health → 200 with status=healthy."""
        resp = api_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_readiness_returns_valid_status(self, api_client: TestClient) -> None:
        """GET /api/v1/health/ready → 200 or 503 — never an unhandled error."""
        resp = api_client.get("/api/v1/health/ready")
        assert resp.status_code in {200, 503}


# ---------------------------------------------------------------------------
# 2. Authentication middleware tests
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    """Verify API key gate is enforced correctly."""

    def test_missing_key_returns_401(self, api_client: TestClient) -> None:
        resp = api_client.get("/api/v1/datasets/")
        assert resp.status_code == 401

    def test_invalid_key_returns_401(self, api_client: TestClient) -> None:
        resp = api_client.get("/api/v1/datasets/", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_valid_key_passes_auth(
        self, api_client: TestClient, authed_headers: dict[str, str]
    ) -> None:
        """dev-key-1 must not return 401.

        The route may return 500/503 if the DB is unreachable but auth must succeed.
        """
        resp = api_client.get("/api/v1/datasets/", headers=authed_headers)
        assert resp.status_code != 401, f"Expected non-401 (auth passed) but got {resp.status_code}"


# ---------------------------------------------------------------------------
# 3. Dataset upload end-to-end
# ---------------------------------------------------------------------------


class TestDatasetUpload:
    """Dataset ingestion through the real API (requires running DB + artifact store)."""

    def test_upload_valid_csv_returns_201(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> None:
        """Upload a valid CSV and assert the 201 response shape."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        resp = api_client.post(
            "/api/v1/datasets/",
            headers=authed_headers,
            files={
                "file": (
                    "unique_test_data.csv",
                    io.BytesIO(_unique_csv_bytes(small_csv_bytes)),
                    "text/csv",
                )
            },
            data={"target_column": "y", "name": f"Upload Test {uuid.uuid4().hex[:8]}"},
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert uuid.UUID(data["id"])
        assert data["row_count"] > 0
        assert data["col_count"] > 0
        assert data["target_column"] == "y"

    def test_duplicate_upload_returns_400(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> None:
        """Uploading the same file twice must return 400 on the second attempt.

        Documents the real API's duplicate-detection behavior (SHA-256
        checksum guard).  Both uploads use the EXACT same bytes so the
        checksums are identical — this is the scenario the gate is designed
        to catch.
        """

        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        # Build a fixed payload — NOT unique so both requests share the same checksum
        fixed_bytes = _unique_csv_bytes(small_csv_bytes)  # unique vs earlier tests

        # First upload with these bytes: must succeed (or already uploaded)
        resp1 = api_client.post(
            "/api/v1/datasets/",
            headers=authed_headers,
            files={"file": ("dup_test.csv", io.BytesIO(fixed_bytes), "text/csv")},
            data={"target_column": "y"},
        )
        if resp1.status_code == 400:
            # Already uploaded in a prior run — the second upload below will still
            # return 400, which is exactly what we want to assert.
            pass
        else:
            assert (
                resp1.status_code == 201
            ), f"First upload should be 201, got {resp1.status_code}: {resp1.text}"

        # Second upload with the EXACT same bytes: must return 400
        resp2 = api_client.post(
            "/api/v1/datasets/",
            headers=authed_headers,
            files={"file": ("dup_test.csv", io.BytesIO(fixed_bytes), "text/csv")},
            data={"target_column": "y"},
        )
        assert (
            resp2.status_code == 400
        ), f"Expected 400 for duplicate upload, got {resp2.status_code}: {resp2.text}"
        assert "error" in resp2.json()

    def test_upload_missing_target_column_returns_422(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> None:
        """Uploading without a target_column must return 422."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        resp = api_client.post(
            "/api/v1/datasets/",
            headers=authed_headers,
            files={
                "file": (
                    "no_target.csv",
                    io.BytesIO(_unique_csv_bytes(small_csv_bytes)),
                    "text/csv",
                )
            },
        )
        assert resp.status_code == 422

    def test_upload_nonexistent_target_returns_error(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> None:
        """Uploading with a target column that does not exist must return 400 or 422."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        resp = api_client.post(
            "/api/v1/datasets/",
            headers=authed_headers,
            files={
                "file": (
                    "bad_target.csv",
                    io.BytesIO(_unique_csv_bytes(small_csv_bytes)),
                    "text/csv",
                )
            },
            data={"target_column": "this_column_does_not_exist"},
        )
        assert resp.status_code in {400, 422}, resp.text


# ---------------------------------------------------------------------------
# 4. Experiment creation (Celery mocked)
# ---------------------------------------------------------------------------


class TestExperimentCreation:
    """Experiment lifecycle via the API.  Celery dispatch is mocked so no
    broker is required; the DB must still be reachable for these tests."""

    def test_create_experiment_returns_201(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> None:
        """Create an experiment with a valid dataset_id and mocked Celery.

        Asserts:
        - HTTP 201 is returned
        - status is 'queued'
        - id is a valid UUID
        """
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        dataset_id = _upload_dataset(api_client, authed_headers, small_csv_bytes)

        with patch(
            "ive.workers.tasks.run_experiment",
            autospec=False,
        ) as mock_task:
            mock_task.delay.return_value = _fake_celery_task("mocked-task-abc123")
            resp = api_client.post(
                "/api/v1/experiments/",
                headers=authed_headers,
                json={
                    "dataset_id": dataset_id,
                    "config": {
                        "analysis_mode": "demo",
                        "model_types": ["linear"],
                        "cv_folds": 3,
                        "bootstrap_iterations": 10,
                    },
                },
            )

        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["status"] == "queued"
        assert uuid.UUID(data["id"])

    def test_create_experiment_calls_celery_delay(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> None:
        """Assert that Celery's .delay() is invoked exactly once per experiment creation."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        dataset_id = _upload_dataset(api_client, authed_headers, small_csv_bytes)

        with patch(
            "ive.workers.tasks.run_experiment",
            autospec=False,
        ) as mock_task:
            mock_task.delay.return_value = _fake_celery_task("delay-check-task")
            api_client.post(
                "/api/v1/experiments/",
                headers=authed_headers,
                json={
                    "dataset_id": dataset_id,
                    "config": {"analysis_mode": "demo", "cv_folds": 2},
                },
            )

        mock_task.delay.assert_called_once()
        args = mock_task.delay.call_args[0]
        assert isinstance(args[0], str), "First arg to delay() must be experiment_id string"

    def test_create_experiment_bad_dataset_returns_404_or_422(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
    ) -> None:
        """Using a non-existent dataset UUID must return 404 or 422."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        fake_id = str(uuid.uuid4())
        resp = api_client.post(
            "/api/v1/experiments/",
            headers=authed_headers,
            json={"dataset_id": fake_id, "config": {}},
        )
        assert resp.status_code in {404, 422}, resp.text


# ---------------------------------------------------------------------------
# 5. Experiment listing and detail
# ---------------------------------------------------------------------------


class TestExperimentListing:
    """List and detail endpoints — require a queued experiment to exist."""

    @pytest.fixture
    def queued_ids(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> dict[str, str]:
        """Upload a unique dataset and create a queued experiment; return both IDs."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        dataset_id = _upload_dataset(
            api_client,
            authed_headers,
            small_csv_bytes,
            filename="listing_fixture.csv",
        )
        experiment_id = _create_experiment(
            api_client,
            authed_headers,
            dataset_id,
            celery_task_id="list-test-task",
        )
        return {"dataset_id": dataset_id, "experiment_id": experiment_id}

    def test_list_experiments_contains_created(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_ids: dict[str, str],
    ) -> None:
        """The created experiment must appear in the listing."""
        resp = api_client.get("/api/v1/experiments/", headers=authed_headers)
        assert resp.status_code == 200
        data = resp.json()
        all_ids = {e["id"] for e in data.get("experiments", [])}
        assert queued_ids["experiment_id"] in all_ids

    def test_list_experiments_filter_by_dataset(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_ids: dict[str, str],
    ) -> None:
        """Filtering by dataset_id returns only experiments for that dataset."""
        resp = api_client.get(
            "/api/v1/experiments/",
            headers=authed_headers,
            params={"dataset_id": queued_ids["dataset_id"]},
        )
        assert resp.status_code == 200
        for exp in resp.json().get("experiments", []):
            assert exp["dataset_id"] == queued_ids["dataset_id"]

    def test_list_experiments_filter_by_status(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_ids: dict[str, str],
    ) -> None:
        """Filtering by status=queued returns only queued experiments."""
        resp = api_client.get(
            "/api/v1/experiments/",
            headers=authed_headers,
            params={"status": "queued"},
        )
        assert resp.status_code == 200
        for exp in resp.json().get("experiments", []):
            assert exp["status"] == "queued"

    def test_experiment_detail_has_required_fields(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_ids: dict[str, str],
    ) -> None:
        """GET /experiments/{id} must return id, status, and config."""
        exp_id = queued_ids["experiment_id"]
        resp = api_client.get(f"/api/v1/experiments/{exp_id}", headers=authed_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "status" in data
        # config may be serialised as config_json or config depending on API version
        assert "config_json" in data or "config" in data

    def test_experiment_progress_endpoint(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_ids: dict[str, str],
    ) -> None:
        """GET /experiments/{id}/progress must return status."""
        exp_id = queued_ids["experiment_id"]
        resp = api_client.get(f"/api/v1/experiments/{exp_id}/progress", headers=authed_headers)
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_nonexistent_experiment_returns_404(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
    ) -> None:
        """Fetching a nonexistent experiment UUID must return 404 (requires DB)."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")
        resp = api_client.get(f"/api/v1/experiments/{uuid.uuid4()}", headers=authed_headers)
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 6. Datasets listing
# ---------------------------------------------------------------------------


class TestDatasetListing:
    """Verify the dataset list endpoint returns the correct response shape."""

    def test_list_datasets_returns_200_with_datasets_key(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
    ) -> None:
        """GET /api/v1/datasets/ → 200 with 'datasets' list key (requires DB)."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        resp = api_client.get("/api/v1/datasets/", headers=authed_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert (
            "datasets" in data
        ), f"Expected 'datasets' key in response but got keys: {list(data.keys())}"
        assert isinstance(data["datasets"], list)
        assert "total" in data


# ---------------------------------------------------------------------------
# 7. Export and report endpoints — available before experiment completes
# ---------------------------------------------------------------------------


class TestReportAndExportEndpoints:
    """Assert that report/export endpoints respond with correct content types
    even when an experiment has not yet produced any patterns or latent variables.
    """

    @pytest.fixture
    def queued_experiment_id(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        small_csv_bytes: bytes,
    ) -> str:
        """Upload a unique dataset and create a queued experiment; return experiment ID."""
        if not _services_available(api_client):
            pytest.skip("Docker services not available")

        dataset_id = _upload_dataset(
            api_client,
            authed_headers,
            small_csv_bytes,
            filename="report_fixture.csv",
        )
        return _create_experiment(
            api_client,
            authed_headers,
            dataset_id,
            config={"cv_folds": 2},
            celery_task_id="report-test-task",
        )

    def test_summary_endpoint_responds(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_experiment_id: str,
    ) -> None:
        """GET /experiments/{id}/summary → 200 with a JSON dict body."""
        resp = api_client.get(
            f"/api/v1/experiments/{queued_experiment_id}/summary",
            headers=authed_headers,
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_report_endpoint_returns_json(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_experiment_id: str,
    ) -> None:
        """GET /experiments/{id}/report → 200 JSON."""
        resp = api_client.get(
            f"/api/v1/experiments/{queued_experiment_id}/report",
            headers=authed_headers,
        )
        assert resp.status_code == 200
        assert "application/json" in resp.headers.get("content-type", "")
        assert isinstance(resp.json(), dict)

    def test_patterns_export_returns_csv(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_experiment_id: str,
    ) -> None:
        """GET /experiments/{id}/patterns/export → 200 CSV."""
        resp = api_client.get(
            f"/api/v1/experiments/{queued_experiment_id}/patterns/export",
            headers=authed_headers,
        )
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/csv" in ct or "application/octet-stream" in ct

    def test_latent_variables_export_returns_csv(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_experiment_id: str,
    ) -> None:
        """GET /experiments/{id}/latent-variables/export → 200 CSV."""
        resp = api_client.get(
            f"/api/v1/experiments/{queued_experiment_id}/latent-variables/export",
            headers=authed_headers,
        )
        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/csv" in ct or "application/octet-stream" in ct

    def test_latent_variables_list_responds(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_experiment_id: str,
    ) -> None:
        """GET /experiments/{id}/latent-variables?status=validated → 200."""
        resp = api_client.get(
            f"/api/v1/experiments/{queued_experiment_id}/latent-variables",
            headers=authed_headers,
            params={"status": "validated"},
        )
        assert resp.status_code == 200

    def test_cancel_queued_experiment(
        self,
        api_client: TestClient,
        authed_headers: dict[str, str],
        queued_experiment_id: str,
    ) -> None:
        """POST /experiments/{id}/cancel → 200 for a queued experiment."""
        with patch(
            "ive.workers.tasks.cancel_experiment",
            autospec=False,
        ) as mock_c:
            mock_c.delay.return_value = _fake_celery_task("cancel-task")
            resp = api_client.post(
                f"/api/v1/experiments/{queued_experiment_id}/cancel",
                headers=authed_headers,
            )
        assert resp.status_code == 200
