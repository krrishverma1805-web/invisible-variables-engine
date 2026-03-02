"""Unit tests for data ingestion module."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from ive.data.ingestion import DataIngestion


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    """Create a temporary CSV file for testing."""
    path = tmp_path / "test.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x1", "x2", "y"])
        writer.writeheader()
        for i in range(50):
            writer.writerow({"x1": i, "x2": i * 2, "y": i * 3})
    return path


@pytest.fixture
def parquet_file(tmp_path: Path, csv_file: Path) -> Path:
    """Create a temporary Parquet file for testing."""
    df = pd.read_csv(csv_file)
    path = tmp_path / "test.parquet"
    df.to_parquet(path, index=False)
    return path


class TestDataIngestion:
    def test_load_csv_returns_dataframe(self, csv_file: Path) -> None:
        """Should return a non-empty DataFrame from a valid CSV."""
        ingestion = DataIngestion()
        df = ingestion.load(csv_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert list(df.columns) == ["x1", "x2", "y"]

    def test_load_parquet_returns_dataframe(self, parquet_file: Path) -> None:
        """Should return a non-empty DataFrame from a valid Parquet file."""
        ingestion = DataIngestion()
        df = ingestion.load(parquet_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_max_rows_limit(self, csv_file: Path) -> None:
        """Should respect the max_rows limit."""
        ingestion = DataIngestion(max_rows=20)
        df = ingestion.load(csv_file)
        assert len(df) == 20

    def test_file_not_found_raises(self) -> None:
        """Should raise FileNotFoundError for non-existent paths."""
        ingestion = DataIngestion()
        with pytest.raises(FileNotFoundError):
            ingestion.load("/non/existent/file.csv")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Should raise ValueError for unsupported file formats."""
        bad_file = tmp_path / "data.xlsx"
        bad_file.write_text("not a csv")
        ingestion = DataIngestion()
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingestion.load(bad_file)

    def test_column_names_are_stripped(self, tmp_path: Path) -> None:
        """Column names with leading/trailing whitespace should be stripped."""
        path = tmp_path / "spaced.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[" x1 ", " y "])
            writer.writeheader()
            writer.writerow({" x1 ": 1, " y ": 2})
        ingestion = DataIngestion()
        df = ingestion.load(path)
        assert "x1" in df.columns
        assert "y" in df.columns
