"""
Integration tests — API Endpoints.

Tests all FastAPI endpoints with a real TestClient, verifying HTTP
status codes, response schema conformance, and authentication.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestHealthEndpoints:
    def test_health_liveness(self, api_client: TestClient) -> None:
        """GET /api/v1/health should return 200 with status=ok."""
        resp = api_client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_readiness_without_db(self, api_client: TestClient) -> None:
        """GET /api/v1/health/ready may return 503 when DB is not available."""
        resp = api_client.get("/api/v1/health/ready")
        assert resp.status_code in (200, 503)


@pytest.mark.integration
class TestAuthMiddleware:
    def test_missing_api_key_returns_401(self, api_client: TestClient) -> None:
        """Requests without X-API-Key should return 401."""
        resp = api_client.get("/api/v1/datasets")
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, api_client: TestClient) -> None:
        """Requests with an invalid API key should return 401."""
        resp = api_client.get("/api/v1/datasets", headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401

    def test_valid_api_key_passes_auth(self, api_client: TestClient, authed_headers) -> None:
        """Requests with a valid API key should not return 401."""
        resp = api_client.get("/api/v1/datasets", headers=authed_headers)
        assert resp.status_code != 401


@pytest.mark.integration
class TestDatasetEndpoints:
    def test_list_datasets_returns_200(self, api_client: TestClient, authed_headers) -> None:
        """GET /api/v1/datasets should return 200 with items list."""
        resp = api_client.get("/api/v1/datasets", headers=authed_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_get_nonexistent_dataset_returns_404(
        self, api_client: TestClient, authed_headers
    ) -> None:
        """GET /api/v1/datasets/{nonexistent_id} should return 404."""
        resp = api_client.get(
            "/api/v1/datasets/00000000-0000-0000-0000-000000000000",
            headers=authed_headers,
        )
        assert resp.status_code == 404


@pytest.mark.integration
class TestExperimentEndpoints:
    def test_list_experiments_returns_200(self, api_client: TestClient, authed_headers) -> None:
        """GET /api/v1/experiments should return 200."""
        resp = api_client.get("/api/v1/experiments", headers=authed_headers)
        assert resp.status_code == 200

    def test_create_experiment_missing_dataset_returns_422_or_404(
        self, api_client: TestClient, authed_headers
    ) -> None:
        """POST /api/v1/experiments with a non-existent dataset ID returns 404 or 422."""
        payload = {
            "dataset_id": "00000000-0000-0000-0000-000000000000",
            "name": "Test Experiment",
            "config": {"target_column": "y", "model_types": ["linear"], "cv_folds": 3},
        }
        resp = api_client.post("/api/v1/experiments", json=payload, headers=authed_headers)
        assert resp.status_code in (404, 422)
