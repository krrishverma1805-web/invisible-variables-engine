"""
Pytest Configuration and Shared Fixtures.

Provides fixtures used across all test modules:
    - Synthetic DataFrame generators for various dataset shapes
    - Pre-built PipelineContext instances for phase testing
    - AsyncSession mocks for DB layer testing
    - TestClient for FastAPI endpoint testing
    - Fake ArtifactStore backed by tmp_path
"""

from __future__ import annotations

import asyncio
import uuid
from typing import AsyncGenerator, Generator

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Pytest asyncio configuration
# ---------------------------------------------------------------------------
# All async tests use the same event loop per session
@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Settings override — use test values for all tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def override_settings(monkeypatch, tmp_path):
    """Override settings with safe test values before each test."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-32chars-minimum-x")
    monkeypatch.setenv("VALID_API_KEYS", "test-api-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://ive:ive@localhost:5432/ive_test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")
    monkeypatch.setenv("ARTIFACT_BASE_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ARTIFACT_STORE_TYPE", "local")

    # Clear the cached settings so the monkeypatched env is used
    from ive.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Synthetic DataFrame fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_regression_df() -> pd.DataFrame:
    """
    Small (200 × 5) regression dataset with a known latent variable.

    y = 2*x1 + 3*x2 + hidden_group_effect + noise
    where hidden_group_effect depends on a binary latent variable.
    """
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    latent = (rng.integers(0, 2, n) * 5).astype(float)  # hidden: 0 or 5
    noise = rng.normal(0, 0.5, n)
    y = 2 * x1 + 3 * x2 + latent + noise

    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def large_regression_df() -> pd.DataFrame:
    """
    Larger (2000 × 10) regression dataset for performance tests.
    """
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(0, 1, (n, 9))
    y = X[:, 0] * 2 + X[:, 1] * -1.5 + rng.normal(0, 1, n)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(9)])
    df["target"] = y
    return df


@pytest.fixture
def categorical_df() -> pd.DataFrame:
    """Dataset with a mix of numeric and categorical columns."""
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "age":        rng.integers(18, 80, n),
        "income":     rng.normal(50000, 15000, n),
        "region":     rng.choice(["north", "south", "east", "west"], n),
        "education":  rng.choice(["high_school", "bachelor", "master", "phd"], n),
        "purchased":  rng.integers(0, 2, n),   # binary target
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """Dataset with intentional missing values."""
    rng = np.random.default_rng(13)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
        "y":  rng.normal(0, 1, n),
    })
    # Introduce ~15% missingness in x2
    mask = rng.random(n) < 0.15
    df.loc[mask, "x2"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Model / ML fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_X_y(simple_regression_df):
    """Return feature matrix and target array from simple_regression_df."""
    X = simple_regression_df[["x1", "x2"]].to_numpy()
    y = simple_regression_df["y"].to_numpy()
    return X, y


@pytest.fixture
def residuals_array(small_X_y):
    """Fake OOF residuals for a simple linear fit."""
    X, y = small_X_y
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    preds = cross_val_predict(Ridge(), X, y, cv=3)
    return y - preds


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client() -> Generator[TestClient, None, None]:
    """Synchronous test client for FastAPI endpoints."""
    from ive.main import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def authed_headers() -> dict[str, str]:
    """Headers with test API key pre-set."""
    return {"X-API-Key": "test-api-key"}


# ---------------------------------------------------------------------------
# Artifact store fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def artifact_store(tmp_path):
    """Local artifact store backed by a temporary directory."""
    from ive.storage.artifact_store import LocalArtifactStore
    return LocalArtifactStore(base_dir=str(tmp_path / "artifacts"))
