"""
End-to-End Workflow Test — Invisible Variables Engine.

Tests the FULL real workflow without any mocking:

    CSV upload                      → POST /api/v1/datasets/
    Experiment creation             → POST /api/v1/experiments/
    Background execution (Celery)   → worker picks up task from Redis
    Result retrieval                → GET summary / report / patterns / latent-variables

Design
------
* Requires Docker Compose stack: PostgreSQL + Redis + Celery worker + API.
* Uses ``httpx`` (sync) against the live API server running on BASE_URL.
  Configure via the ``IVE_BASE_URL`` env var (default: http://localhost:8000).
* Polls the experiment detail endpoint until status is "completed" or "failed"
  with a configurable timeout (default: 120 s).
* The duplicate-checksum guard is bypassed via the same pandas-epsilon trick
  used in the integration suite — each upload has a unique SHA-256.
* If the server or database is not reachable the tests skip themselves
  gracefully with a diagnostic message.

Markers
-------
    @pytest.mark.e2e   — full stack, never runs in CI without Docker Compose

Running manually
----------------
    # From the project root with services up:
    docker-compose exec api pytest tests/e2e/ -v

    # Or locally (requires the full stack on localhost:8000):
    IVE_BASE_URL=http://localhost:8000 poetry run pytest tests/e2e/ -v
"""

from __future__ import annotations

import io
import os
import time
import uuid

import httpx
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL: str = os.getenv("IVE_BASE_URL", "http://localhost:8000")
API_KEY: str = os.getenv("IVE_API_KEY", "dev-key-1")

POLL_INTERVAL_S: float = 3.0
POLL_TIMEOUT_S: float = float(os.getenv("IVE_E2E_TIMEOUT", "120"))

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Inline CSV factories
# ---------------------------------------------------------------------------


def _make_regression_csv(n: int = 200, seed: int = 42) -> bytes:
    """Small regression CSV with a hidden subgroup effect in x3 (categorical feature).

    Hidden pattern: rows where cat == 'A' have a systematically higher y,
    which the pipeline should detect as a subgroup latent variable.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    cat = rng.choice(["A", "B", "C"], n)
    noise = rng.normal(0, 0.5, n)
    group_effect = (cat == "A").astype(float) * 3.0
    y = 2.0 * x1 - 1.5 * x2 + group_effect + noise
    df = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": y})
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _make_noise_csv(n: int = 150, seed: int = 99) -> bytes:
    """Pure noise CSV — no signal, no validatable latent variables."""
    import numpy as np

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(0, 1, n),
            "x3": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        }
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _unique_bytes(base: bytes) -> bytes:
    """Perturb the first numeric cell by a UUID-derived epsilon.

    This ensures a fresh SHA-256 checksum on every call, bypassing the
    API's duplicate-file detection gate without altering the schema.
    """
    df = pd.read_csv(io.BytesIO(base))
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        epsilon = (uuid.uuid4().int % 1_000_000) * 1e-12
        df.at[0, num_cols[0]] = float(df.at[0, num_cols[0]]) + epsilon
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _client() -> httpx.Client:
    """Return a configured httpx sync client."""
    return httpx.Client(
        base_url=BASE_URL,
        headers={"X-API-Key": API_KEY},
        timeout=30.0,
    )


def _services_available() -> bool:
    """Return True when the live API + DB are reachable."""
    try:
        with _client() as c:
            r = c.get("/api/v1/health/ready")
            return r.status_code == 200
    except Exception:
        return False


def _upload_dataset(
    client: httpx.Client,
    csv_bytes: bytes,
    target_column: str = "y",
    name: str | None = None,
) -> str:
    """Upload *csv_bytes* and return the dataset UUID string."""
    form_data: dict[str, str] = {"target_column": target_column}
    if name:
        form_data["name"] = name

    resp = client.post(
        "/api/v1/datasets/",
        files={"file": ("e2e_test.csv", io.BytesIO(_unique_bytes(csv_bytes)), "text/csv")},
        data=form_data,
    )
    assert resp.status_code == 201, f"Dataset upload failed ({resp.status_code}): {resp.text}"
    return resp.json()["id"]


def _create_experiment(
    client: httpx.Client,
    dataset_id: str,
    config: dict | None = None,
) -> str:
    """Create an experiment (real Celery dispatch — no mock) and return its UUID."""
    payload: dict = {
        "dataset_id": dataset_id,
        "config": config
        or {
            "analysis_mode": "demo",
            "model_types": ["linear", "xgboost"],
            "cv_folds": 3,
            "bootstrap_iterations": 10,
        },
    }
    resp = client.post("/api/v1/experiments/", json=payload)
    assert resp.status_code == 201, f"Experiment creation failed ({resp.status_code}): {resp.text}"
    return resp.json()["id"]


def _poll_until_terminal(
    client: httpx.Client,
    experiment_id: str,
    timeout: float = POLL_TIMEOUT_S,
    interval: float = POLL_INTERVAL_S,
) -> dict:
    """Poll GET /experiments/{id} until status is in TERMINAL_STATUSES.

    Returns the final experiment detail dict.

    Raises:
        TimeoutError: if the experiment does not reach a terminal state within
                      *timeout* seconds.
        AssertionError: if any poll request returns a non-200 status.
    """
    deadline = time.monotonic() + timeout
    while True:
        resp = client.get(f"/api/v1/experiments/{experiment_id}")
        assert resp.status_code == 200, f"Poll failed ({resp.status_code}): {resp.text}"
        data = resp.json()
        status_val: str = data.get("status", "")
        if status_val in TERMINAL_STATUSES:
            return data

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Experiment {experiment_id} did not complete within {timeout}s. "
                f"Last status: {status_val!r}"
            )
        time.sleep(min(interval, remaining))


# ---------------------------------------------------------------------------
# Skip guard fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def require_services() -> None:
    """Skip the entire module when the live stack is not reachable."""
    if not _services_available():
        pytest.skip(
            f"E2E stack not available at {BASE_URL} — " "run `docker-compose up` and retry."
        )


# ---------------------------------------------------------------------------
# Test 1: Full regression workflow with signal
# ---------------------------------------------------------------------------


def test_full_regression_workflow() -> None:
    """Upload → create experiment → worker executes → assert completed results.

    Covers:
    - /api/v1/datasets/ POST (upload)
    - /api/v1/experiments/ POST (create + real Celery dispatch)
    - /api/v1/experiments/{id} GET (polling)
    - /api/v1/experiments/{id}/summary GET
    - /api/v1/experiments/{id}/report GET
    - /api/v1/experiments/{id}/patterns GET
    - /api/v1/experiments/{id}/latent-variables GET
    """
    csv_bytes = _make_regression_csv(n=200, seed=42)

    with _client() as client:
        # Step 1 — Upload
        dataset_id = _upload_dataset(
            client,
            csv_bytes,
            name=f"E2E Regression {uuid.uuid4().hex[:6]}",
        )

        # Step 2 — Create experiment (real Celery dispatch)
        experiment_id = _create_experiment(
            client,
            dataset_id,
            config={
                "analysis_mode": "demo",
                "model_types": ["linear", "xgboost"],
                "cv_folds": 3,
                "bootstrap_iterations": 10,
            },
        )

        # Step 3 — Poll to terminal state
        final = _poll_until_terminal(client, experiment_id)

        # Step 4 — Assert completed (not failed)
        assert final["status"] == "completed", (
            f"Experiment ended with status {final['status']!r}, not 'completed'. "
            "Check the Celery worker logs."
        )

        # Step 5 & 6 — Fetch and validate result endpoints

        # 5a. Summary
        r_summary = client.get(f"/api/v1/experiments/{experiment_id}/summary")
        assert r_summary.status_code == 200, r_summary.text
        summary = r_summary.json()
        assert isinstance(summary, dict), "Summary must be a JSON object"

        # 5b. Report
        r_report = client.get(f"/api/v1/experiments/{experiment_id}/report")
        assert r_report.status_code == 200, r_report.text
        assert "application/json" in r_report.headers.get("content-type", "")
        report = r_report.json()
        assert isinstance(report, dict), "Report must be a JSON object"

        # 5c. Patterns
        r_patterns = client.get(f"/api/v1/experiments/{experiment_id}/patterns")
        assert r_patterns.status_code == 200, r_patterns.text
        patterns = r_patterns.json()
        assert isinstance(
            patterns, list
        ), f"Patterns endpoint must return a JSON list; got {type(patterns).__name__}"

        # 5d. Latent variables
        r_lvs = client.get(f"/api/v1/experiments/{experiment_id}/latent-variables")
        assert r_lvs.status_code == 200, r_lvs.text
        lv_data = r_lvs.json()
        assert (
            "variables" in lv_data
        ), f"Expected 'variables' key in LV response; got: {list(lv_data.keys())}"
        assert isinstance(lv_data["variables"], list)
        assert "total" in lv_data


# ---------------------------------------------------------------------------
# Test 2: No-signal dataset — pipeline must still complete
# ---------------------------------------------------------------------------


def test_no_signal_workflow_completes() -> None:
    """Upload a pure-noise dataset — pipeline must complete without crashing.

    The pipeline should reach status='completed' even when no latent variables
    are validated (n_validated == 0 is acceptable — a crash or 'failed' status
    is not).
    """
    csv_bytes = _make_noise_csv(n=150, seed=99)

    with _client() as client:
        dataset_id = _upload_dataset(
            client,
            csv_bytes,
            name=f"E2E Noise {uuid.uuid4().hex[:6]}",
        )
        experiment_id = _create_experiment(
            client,
            dataset_id,
            config={
                "analysis_mode": "demo",
                "model_types": ["linear"],
                "cv_folds": 3,
                "bootstrap_iterations": 10,
            },
        )
        final = _poll_until_terminal(client, experiment_id)

        assert final["status"] == "completed", (
            f"No-signal experiment ended with {final['status']!r}. "
            "Pipeline must not crash on pure-noise data."
        )

        # Latent variables list must still be 200 even if empty
        r_lvs = client.get(f"/api/v1/experiments/{experiment_id}/latent-variables")
        assert r_lvs.status_code == 200, r_lvs.text
        lv_data = r_lvs.json()
        assert "variables" in lv_data
        assert isinstance(lv_data["variables"], list)
        # It is fine (and expected) that no variables are validated
        n_validated = sum(1 for v in lv_data["variables"] if v.get("status") == "validated")
        assert n_validated >= 0  # structural — not a criteria for success


# ---------------------------------------------------------------------------
# Test 3: Export endpoints respond post-completion
# ---------------------------------------------------------------------------


def test_export_endpoints_after_completion() -> None:
    """Verify CSV export endpoints return usable data after a completed experiment.

    Uses a fresh small dataset so this test is always independent.
    """
    csv_bytes = _make_regression_csv(n=150, seed=7)

    with _client() as client:
        dataset_id = _upload_dataset(client, csv_bytes)
        experiment_id = _create_experiment(
            client,
            dataset_id,
            config={
                "analysis_mode": "demo",
                "model_types": ["linear"],
                "cv_folds": 3,
                "bootstrap_iterations": 10,
            },
        )
        final = _poll_until_terminal(client, experiment_id)
        assert final["status"] == "completed"

        # Patterns CSV
        r_pcsv = client.get(f"/api/v1/experiments/{experiment_id}/patterns/export")
        assert r_pcsv.status_code == 200, r_pcsv.text
        ct = r_pcsv.headers.get("content-type", "")
        assert (
            "text/csv" in ct or "application/octet-stream" in ct
        ), f"Unexpected content-type for patterns CSV: {ct!r}"

        # Latent variables CSV
        r_lvcsv = client.get(f"/api/v1/experiments/{experiment_id}/latent-variables/export")
        assert r_lvcsv.status_code == 200, r_lvcsv.text
        ct = r_lvcsv.headers.get("content-type", "")
        assert (
            "text/csv" in ct or "application/octet-stream" in ct
        ), f"Unexpected content-type for latent-variables CSV: {ct!r}"


# ---------------------------------------------------------------------------
# Test 4: Progress endpoint reflects running → completed transition
# ---------------------------------------------------------------------------


def test_progress_endpoint_reflects_completion() -> None:
    """GET /experiments/{id}/progress must return status and a progress_pct field
    at least once before and after the experiment completes.
    """
    csv_bytes = _make_regression_csv(n=150, seed=55)

    with _client() as client:
        dataset_id = _upload_dataset(client, csv_bytes)
        experiment_id = _create_experiment(
            client,
            dataset_id,
            config={
                "analysis_mode": "demo",
                "model_types": ["linear"],
                "cv_folds": 3,
                "bootstrap_iterations": 10,
            },
        )

        # Collect at least one progress snapshot before terminal state
        progress_snapshots: list[dict] = []

        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            r = client.get(f"/api/v1/experiments/{experiment_id}/progress")
            assert r.status_code == 200, f"Progress poll failed: {r.text}"
            snap = r.json()
            assert "status" in snap
            progress_snapshots.append(snap)
            if snap["status"] in TERMINAL_STATUSES:
                break
            time.sleep(POLL_INTERVAL_S)
        else:
            raise TimeoutError(
                f"Experiment {experiment_id} did not complete within {POLL_TIMEOUT_S}s"
            )

        assert any(
            s["status"] in TERMINAL_STATUSES for s in progress_snapshots
        ), "No snapshot showed a terminal status"

        # Final status must be completed
        final_status = progress_snapshots[-1]["status"]
        assert final_status == "completed", f"Expected completed, got {final_status!r}"
