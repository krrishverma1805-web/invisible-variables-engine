"""Unit tests for data profiler module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ive.data.profiler import DataProfiler


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "numeric_1": rng.normal(0, 1, 100),
        "numeric_2": rng.integers(0, 100, 100).astype(float),
        "binary_cat": rng.choice(["A", "B"], 100),
        "target": rng.normal(5, 2, 100),
    })


class TestDataProfiler:
    def test_profile_returns_dict(self, mixed_df: pd.DataFrame) -> None:
        """profile() should return a dict with expected top-level keys."""
        profiler = DataProfiler()
        result = profiler.profile(mixed_df)
        assert isinstance(result, dict)
        assert "schema" in result
        assert "warnings" in result

    def test_infer_column_types_returns_dict(self, mixed_df: pd.DataFrame) -> None:
        """infer_column_types() should return a dict with one entry per column."""
        profiler = DataProfiler()
        col_types = profiler.infer_column_types(mixed_df)
        assert isinstance(col_types, dict)
        assert set(col_types.keys()) == set(mixed_df.columns)

    def test_infer_handles_none_df(self) -> None:
        """infer_column_types() on None should return empty dict."""
        profiler = DataProfiler()
        result = profiler.infer_column_types(None)
        assert result == {}

    def test_compute_column_stats_returns_dict(self, mixed_df: pd.DataFrame) -> None:
        """compute_column_stats() should return a dict with numeric keys."""
        profiler = DataProfiler()
        stats = profiler.compute_column_stats(mixed_df, "numeric_1")
        assert isinstance(stats, dict)
        assert "mean" in stats

    def test_profile_none_df(self) -> None:
        """profile() on None should return dict without raising."""
        profiler = DataProfiler()
        result = profiler.profile(None)
        assert isinstance(result, dict)
