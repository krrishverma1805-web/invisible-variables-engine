"""
Unit tests for DataIngestionService — Invisible Variables Engine.

Tests are grouped into:
    * CSV parsing         — encoding, delimiter, BOM, whitespace, empty files
    * Type detection      — numeric, categorical, datetime, boolean, id, text
    * Validation          — min rows, column count, target presence, null/variance
    * Task type detection — regression vs classification
    * Full ingest         — end-to-end with mocked artifact store + no DB session

No real database or filesystem access required.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from ive.data.ingestion import DataIngestionService, DatasetValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 150, target: str = "y") -> pd.DataFrame:
    """Minimal valid DataFrame with n rows."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"x1": rng.standard_normal(n), "x2": rng.standard_normal(n), target: rng.standard_normal(n)}
    )


def _df_to_csv_bytes(df: pd.DataFrame, sep: str = ",", encoding: str = "utf-8") -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=sep)
    return buf.getvalue().encode(encoding)


def _make_svc() -> DataIngestionService:
    """Build a DataIngestionService with a mocked artifact store."""
    svc = DataIngestionService.__new__(DataIngestionService)
    from ive.utils.logging import get_logger

    svc.logger = get_logger("test.ingestion")
    store = AsyncMock()
    store.save_file = AsyncMock(return_value="/artifacts/test.csv")
    store.delete_file = AsyncMock()
    svc.store = store
    from ive.config import get_settings

    svc.settings = get_settings()
    return svc


# ===========================================================================
# CSV Parsing
# ===========================================================================


@pytest.mark.unit
class TestCSVParsing:
    """Tests for DataIngestionService._parse_csv."""

    def test_parse_valid_comma_csv(self) -> None:
        """Standard comma-delimited CSV parses without error."""
        svc = _make_svc()
        df = _make_df()
        data = _df_to_csv_bytes(df)
        result = svc._parse_csv(data, "test.csv")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert set(result.columns) == set(df.columns)

    @pytest.mark.parametrize("sep", [",", ";", "\t"])
    def test_parse_different_delimiters(self, sep: str) -> None:
        """Parser auto-detects comma, semicolon, and tab delimiters."""
        svc = _make_svc()
        df = _make_df()
        data = _df_to_csv_bytes(df, sep=sep)
        result = svc._parse_csv(data, "test.csv")
        assert len(result) == len(df), f"Failed with delimiter {sep!r}"
        assert "x1" in result.columns

    def test_parse_utf8_bom(self) -> None:
        """UTF-8 BOM (\\ufeff) is stripped and the CSV parses correctly."""
        svc = _make_svc()
        df = _make_df()
        raw = _df_to_csv_bytes(df)
        bom_data = b"\xef\xbb\xbf" + raw  # UTF-8 BOM
        result = svc._parse_csv(bom_data, "bom.csv")
        assert "x1" in result.columns, "BOM not stripped — column name mangled"

    def test_parse_latin1_encoding(self) -> None:
        """Latin-1 encoded CSV (with non-ASCII chars) is decoded without error."""
        svc = _make_svc()
        # Build a CSV with an accented character value
        csv_text = "col1,col2\ncafé,123\ndonnées,456\n" * 60  # 120 rows
        raw = csv_text.encode("latin-1")
        result = svc._parse_csv(raw, "latin.csv")
        assert len(result) > 0

    def test_parse_empty_file_raises(self) -> None:
        """Empty bytes raise DatasetValidationError."""
        svc = _make_svc()
        with pytest.raises(DatasetValidationError):
            svc._parse_csv(b"", "empty.csv")

    def test_parse_whitespace_only_raises(self) -> None:
        """All-whitespace bytes raise DatasetValidationError."""
        svc = _make_svc()
        with pytest.raises(DatasetValidationError):
            svc._parse_csv(b"   \n  \n  ", "blank.csv")

    def test_parse_strips_whitespace_from_headers(self) -> None:
        """Column names with surrounding spaces are stripped."""
        svc = _make_svc()
        csv_data = b" x1 , x2 , y \n1.0,2.0,3.0\n" * 80
        result = svc._parse_csv(csv_data, "spaced.csv")
        assert "x1" in result.columns, f"Got columns: {list(result.columns)}"
        assert "y" in result.columns

    def test_parse_removes_fully_empty_rows(self) -> None:
        """Trailing empty rows are dropped."""
        svc = _make_svc()
        df = _make_df(n=100)
        raw = _df_to_csv_bytes(df)
        # Append empty lines
        raw_with_blanks = raw + b"\n\n\n"
        result = svc._parse_csv(raw_with_blanks, "blanks.csv")
        assert len(result) == len(df)

    def test_parse_header_only_raises(self) -> None:
        """A CSV with only a header row (no data) raises DatasetValidationError."""
        svc = _make_svc()
        with pytest.raises(DatasetValidationError):
            svc._parse_csv(b"col1,col2,col3\n", "header_only.csv")


# ===========================================================================
# Column Type Detection
# ===========================================================================


@pytest.mark.unit
class TestColumnTypeDetection:
    """Tests for DataIngestionService._detect_column_types."""

    def _detect(self, df: pd.DataFrame):
        svc = _make_svc()
        return {c.name: c for c in svc._detect_column_types(df)}

    def test_detect_numeric_columns(self) -> None:
        """Float and integer columns are classified as 'numeric'."""
        df = pd.DataFrame({"val": [1.1, 2.2, 3.3] * 100, "y": range(300)})
        types = self._detect(df)
        assert types["val"].detected_type == "numeric", f"Got {types['val'].detected_type}"

    def test_detect_categorical_columns(self) -> None:
        """Low-cardinality string columns are 'categorical'."""
        df = pd.DataFrame({"cat": ["A", "B", "C"] * 200, "y": range(600)})
        types = self._detect(df)
        assert types["cat"].detected_type == "categorical"

    def test_detect_boolean_columns(self) -> None:
        """Columns with only true/false strings are 'boolean'."""
        df = pd.DataFrame({"flag": ["true", "false", "yes", "no"] * 100, "y": range(400)})
        types = self._detect(df)
        assert types["flag"].detected_type == "boolean"

    def test_detect_datetime_columns(self) -> None:
        """ISO-8601 date strings are classified as 'datetime'."""
        dates = pd.date_range("2020-01-01", periods=200, freq="D").astype(str).tolist()
        df = pd.DataFrame({"date": dates, "x": range(200), "y": range(200)})
        types = self._detect(df)
        assert types["date"].detected_type == "datetime"

    def test_detect_id_columns(self) -> None:
        """Columns named '*_id' or 'id' with high cardinality become 'id'."""
        df = pd.DataFrame({"user_id": range(300), "x1": range(300), "y": range(300)})
        types = self._detect(df)
        assert types["user_id"].detected_type == "id"

    def test_detect_text_columns(self) -> None:
        """High-cardinality string columns with no obvious pattern are 'text'."""
        rng = np.random.default_rng(7)
        # Unique sentences — high cardinality, not IDs
        sentences = [f"sentence {i} blah blah" for i in range(300)]
        df = pd.DataFrame({"description": sentences, "x": range(300), "y": range(300)})
        types = self._detect(df)
        assert types["description"].detected_type in ("text", "categorical")

    def test_detect_null_pct_accuracy(self) -> None:
        """null_pct is computed correctly."""
        n = 200
        vals = [1.0] * 150 + [float("nan")] * 50
        df = pd.DataFrame({"x": vals, "y": range(n)})
        types = self._detect(df)
        assert abs(types["x"].null_pct - 25.0) < 0.5, f"Expected ~25%, got {types['x'].null_pct}"

    def test_detect_unique_count(self) -> None:
        """unique_count is the count of non-null distinct values."""
        df = pd.DataFrame({"x": [1, 2, 3, 1, 2, None] * 50, "y": range(300)})
        types = self._detect(df)
        assert types["x"].unique_count == 3


# ===========================================================================
# Validation
# ===========================================================================


@pytest.mark.unit
class TestValidation:
    """Tests for DataIngestionService._validate."""

    def _run_validate(self, df, target_column="y", time_column=None, file_bytes=1000):
        svc = _make_svc()
        return svc._validate(
            df, svc._detect_column_types(df), target_column, time_column, file_bytes
        )

    def test_validate_passes_for_valid_dataset(self, sample_regression_df: pd.DataFrame) -> None:
        """A clean, sufficiently large dataset generates no errors."""
        df = sample_regression_df.rename(columns={"price": "y"})
        # Should not raise
        warnings = self._run_validate(df, target_column="y")
        assert isinstance(warnings, list)

    def test_validate_minimum_rows(self) -> None:
        """DatasetValidationError raised for fewer than 100 rows."""
        df = _make_df(n=50)
        svc = _make_svc()
        with pytest.raises(DatasetValidationError, match="row"):
            svc._validate(df, svc._detect_column_types(df), "y", None, 1000)

    def test_validate_minimum_feature_columns(self) -> None:
        """DatasetValidationError raised when only 1 feature besides target."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x1": rng.standard_normal(200), "y": rng.standard_normal(200)})
        svc = _make_svc()
        with pytest.raises(DatasetValidationError):
            svc._validate(df, svc._detect_column_types(df), "y", None, 1000)

    def test_validate_target_must_exist(self) -> None:
        """DatasetValidationError raised when target column is missing."""
        df = _make_df(n=200)
        svc = _make_svc()
        with pytest.raises(DatasetValidationError, match="target"):
            svc._validate(df, svc._detect_column_types(df), "nonexistent_col", None, 1000)

    def test_validate_target_not_all_null(self) -> None:
        """DatasetValidationError raised when target is entirely NaN."""
        df = _make_df(n=200)
        df["y"] = float("nan")
        svc = _make_svc()
        with pytest.raises(DatasetValidationError, match="null"):
            svc._validate(df, svc._detect_column_types(df), "y", None, 1000)

    def test_validate_target_must_have_variance(self) -> None:
        """DatasetValidationError raised when target is a constant."""
        df = _make_df(n=200)
        df["y"] = 42.0  # zero variance
        svc = _make_svc()
        with pytest.raises(DatasetValidationError, match="[Vv]ariance|constant"):
            svc._validate(df, svc._detect_column_types(df), "y", None, 1000)

    def test_validate_time_column_must_exist_if_specified(self) -> None:
        """DatasetValidationError raised when specified time_column is absent."""
        df = _make_df(n=200)
        svc = _make_svc()
        with pytest.raises(DatasetValidationError, match="time"):
            svc._validate(df, svc._detect_column_types(df), "y", "missing_time_col", 1000)

    def test_validate_warns_on_high_null_columns(self) -> None:
        """A warning (not error) is returned for columns with >95% nulls."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "almost_empty": [float("nan")] * 196 + [1.0, 2.0, 3.0, 4.0],
                "y": rng.standard_normal(n),
            }
        )
        svc = _make_svc()
        warnings = svc._validate(df, svc._detect_column_types(df), "y", None, 1000)
        assert any(
            "almost_empty" in w for w in warnings
        ), f"Expected a warning about 'almost_empty', got: {warnings}"


# ===========================================================================
# Task type detection
# ===========================================================================


@pytest.mark.unit
class TestTaskTypeDetection:
    """Tests for schema_json[detected_task] emitted by ingestion."""

    def _get_schema(self, df: pd.DataFrame, target: str) -> dict:
        svc = _make_svc()
        types = svc._detect_column_types(df)
        return svc._build_schema(df, types, target, None)

    def test_detect_regression_task(self) -> None:
        """Continuous float target → regression."""
        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        schema = self._get_schema(df, "y")
        assert schema["detected_task"] == "regression"

    def test_detect_binary_classification_task(self) -> None:
        """Binary 0/1 integer target → classification."""
        rng = np.random.default_rng(2)
        n = 200
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y": rng.integers(0, 2, n),  # 0 or 1
            }
        )
        schema = self._get_schema(df, "y")
        assert schema["detected_task"] == "classification"

    def test_detect_multiclass_classification_task(self) -> None:
        """Integer target with 3 unique values → classification."""
        rng = np.random.default_rng(3)
        n = 300
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "label": rng.integers(0, 3, n),  # 0, 1, 2
            }
        )
        schema = self._get_schema(df, "label")
        assert schema["detected_task"] == "classification"


# ===========================================================================
# Full ingestion (mocked store, no real DB)
# ===========================================================================


@pytest.mark.unit
class TestFullIngestion:
    """End-to-end ingest() with mocked artifact store and no database session."""

    @pytest.mark.asyncio
    async def test_full_ingestion_regression_dataset(self, sample_csv_bytes: bytes) -> None:
        """ingest() returns a valid IngestionResult for a regression CSV."""
        svc = _make_svc()
        with patch("ive.data.ingestion.get_artifact_store", return_value=svc.store):
            result = await svc.ingest(
                file_content=sample_csv_bytes,
                filename="regression.csv",
                target_column="price",
                session=None,  # no DB
            )
        assert result.row_count > 0
        assert result.col_count >= 2
        assert result.target_column == "price"
        assert result.checksum  # non-empty hex string
        assert len(result.columns) == result.col_count

    @pytest.mark.asyncio
    async def test_full_ingestion_classification_dataset(
        self, sample_classification_csv_bytes: bytes
    ) -> None:
        """ingest() returns detected_task='classification' for a binary target."""
        svc = _make_svc()
        with patch("ive.data.ingestion.get_artifact_store", return_value=svc.store):
            result = await svc.ingest(
                file_content=sample_classification_csv_bytes,
                filename="classification.csv",
                target_column="target",
                session=None,
            )
        assert result.detected_task == "classification"

    @pytest.mark.asyncio
    async def test_full_ingestion_empty_file_raises(self) -> None:
        """ingest() raises DatasetValidationError for empty bytes."""
        svc = _make_svc()
        with patch("ive.data.ingestion.get_artifact_store", return_value=svc.store):
            with pytest.raises(DatasetValidationError):
                await svc.ingest(b"", "empty.csv", "y", session=None)

    @pytest.mark.asyncio
    async def test_ingestion_result_has_schema_json(self, sample_csv_bytes: bytes) -> None:
        """schema_json contains 'columns' and 'detected_task' keys."""
        svc = _make_svc()
        with patch("ive.data.ingestion.get_artifact_store", return_value=svc.store):
            result = await svc.ingest(
                file_content=sample_csv_bytes,
                filename="test.csv",
                target_column="price",
                session=None,
            )
        assert "columns" in result.schema_json
        assert "detected_task" in result.schema_json

    @pytest.mark.parametrize("sep", [",", ";", "\t"])
    @pytest.mark.asyncio
    async def test_ingestion_delimiter_variants(self, sep: str) -> None:
        """ingest() succeeds for CSV files with different delimiters."""
        rng = np.random.default_rng(99)
        n = 150
        df = pd.DataFrame(
            {
                "x1": rng.standard_normal(n),
                "x2": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False, sep=sep)
        csv_bytes = buf.getvalue().encode("utf-8")

        svc = _make_svc()
        with patch("ive.data.ingestion.get_artifact_store", return_value=svc.store):
            result = await svc.ingest(csv_bytes, "delimit.csv", "y", session=None)
        assert (
            result.row_count == n
        ), f"Delimiter {sep!r}: expected {n} rows, got {result.row_count}"
