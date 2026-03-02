"""
Shared pytest fixtures for Invisible Variables Engine tests.

Provides lightweight, DB/Redis-free fixtures for unit testing.
All fixtures are function-scoped (default) for test isolation.
"""

from __future__ import annotations

import io
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Custom pytest markers
# ---------------------------------------------------------------------------

def pytest_configure(config):  # noqa: D401
    """Register custom marks used throughout the test suite."""
    config.addinivalue_line("markers", "unit: fast, in-process unit tests (no DB/Redis)")
    config.addinivalue_line("markers", "integration: tests that require Docker services")
    config.addinivalue_line("markers", "statistical: property-based / numerical accuracy tests")


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_regression_df() -> pd.DataFrame:
    """500-row regression DataFrame with known feature→target relationships.

    Relationship: price ≈ 50,000 + 5,000·feature_a + 100·feature_b + noise
    """
    rng = np.random.default_rng(42)
    n = 500
    feature_a = rng.standard_normal(n)
    feature_b = rng.uniform(0, 100, n)
    noise = rng.standard_normal(n) * 1000
    df = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "category": rng.choice(["cat_1", "cat_2", "cat_3"], n),
            "price": 50_000 + 5_000 * feature_a + 100 * feature_b + noise,
        }
    )
    return df


@pytest.fixture
def sample_classification_df() -> pd.DataFrame:
    """500-row binary classification DataFrame (70/30 split)."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "income": rng.normal(50_000, 20_000, n),
            "category": rng.choice(["A", "B", "C"], n),
            "target": rng.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def sample_large_df() -> pd.DataFrame:
    """1,000-row dataset suitable for correlation and profiling tests."""
    rng = np.random.default_rng(0)
    n = 1_000
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = x1 * 0.97 + rng.standard_normal(n) * 0.1   # highly correlated with x1
    return pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "cat": rng.choice(["a", "b", "c", "d"], n),
            "target": 2 * x1 - x2 + rng.standard_normal(n) * 0.5,
        }
    )


# ---------------------------------------------------------------------------
# CSV bytes helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_bytes(sample_regression_df: pd.DataFrame) -> bytes:
    """UTF-8 CSV bytes of the sample regression DataFrame."""
    return sample_regression_df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def sample_classification_csv_bytes(sample_classification_df: pd.DataFrame) -> bytes:
    """UTF-8 CSV bytes of the sample classification DataFrame."""
    return sample_classification_df.to_csv(index=False).encode("utf-8")


@pytest.fixture
def bad_csv_bytes() -> dict[str, bytes]:
    """Named bad CSV byte strings for negative-path tests."""
    return {
        "empty": b"",
        "whitespace_only": b"   \n  \n",
        "header_only": b"col1,col2,col3\n",
        "single_data_row": b"col1,col2\n1,2\n",
        "wrong_encoding": "col1,col2\nvalue,données".encode("latin-1"),
        "no_target": b"col1,col2\n" + b"1,2\n" * 150,
    }


def _make_csv(n: int, target_col: str = "y", delimiter: str = ",") -> bytes:
    """Build a minimal valid CSV with ``n`` data rows."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            target_col: rng.standard_normal(n),
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=delimiter)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def minimal_valid_csv() -> bytes:
    """150-row valid CSV — just above the _MIN_ROWS=100 threshold."""
    return _make_csv(150)


# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory (delegates to pytest's ``tmp_path``)."""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Mock DB session
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_session():
    """Async SQLAlchemy session mock.

    Stubs the execute / flush / commit / rollback / close / add chain so
    that DataIngestionService can be tested without a real database.
    """
    session = AsyncMock()
    session.commit = AsyncMock(return_value=None)
    session.rollback = AsyncMock(return_value=None)
    session.close = AsyncMock(return_value=None)
    session.add = MagicMock()
    session.flush = AsyncMock(return_value=None)

    # execute() returns a result that supports scalar_one_or_none()
    exec_result = MagicMock()
    exec_result.scalar_one_or_none.return_value = None   # no duplicate found
    session.execute = AsyncMock(return_value=exec_result)
    return session


# ---------------------------------------------------------------------------
# Mock artifact store
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_artifact_store():
    """Async mock for LocalArtifactStore / S3ArtifactStore."""
    store = AsyncMock()
    store.save_file = AsyncMock(return_value="/artifacts/datasets/test.csv")
    store.delete_file = AsyncMock(return_value=None)
    store.file_exists = AsyncMock(return_value=False)
    return store
