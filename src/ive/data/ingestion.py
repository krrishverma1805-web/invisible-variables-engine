"""
Data Ingestion Service — Invisible Variables Engine.

Responsible for the complete ingestion pipeline:

    1. Parse raw CSV bytes (handle encoding, delimiters, BOMs)
    2. Auto-detect column types (numeric, categorical, datetime, boolean, text, id)
    3. Validate dataset against minimum requirements
    4. Persist dataset file to the artifact store
    5. Create the ``Dataset`` DB record
    6. Return a structured ``IngestionResult``

The service is **stateless** — every call to :meth:`DataIngestionService.ingest`
is self-contained.  Large files (>50 MB) are parsed with Polars for speed;
smaller files use Pandas.

Usage::

    service = DataIngestionService()
    result = await service.ingest(
        file_content=raw_bytes,
        filename="housing.csv",
        target_column="price",
        session=session,
    )
"""

from __future__ import annotations

import csv
import io
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from ive.config import get_settings
from ive.db.models import Dataset
from ive.db.repositories.dataset_repo import DatasetRepository
from ive.storage.artifact_store import compute_checksum, get_artifact_store
from ive.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_ROWS = 100
_MIN_FEATURE_COLS = 2
_MAX_FILE_BYTES = 500 * 1024 * 1024  # 500 MB
_LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50 MB — switch to Polars
_HIGH_NULL_PCT = 95.0
_SNIFF_BYTES = 10_240
_SAMPLE_VALUES = 5
_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1")
_BOOLEAN_TRUES = {"true", "yes", "y", "1", "1.0", "t"}
_BOOLEAN_FALSES = {"false", "no", "n", "0", "0.0", "f"}
_BOOLEAN_PAIRS = _BOOLEAN_TRUES | _BOOLEAN_FALSES
_ID_PATTERNS = re.compile(r"(?:^|_)(id|index|key|uuid|pk)(?:$|_)", re.IGNORECASE)
_MAX_CATEGORICAL_UNIQUE = 50
_CATEGORICAL_UNIQUE_PCT = 5.0
_DATETIME_SUCCESS_RATE = 0.80
_MAX_CLASSIFICATION_UNIQUE = 20


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DatasetValidationError(Exception):
    """Raised when a dataset fails one or more validation rules.

    Attributes:
        errors: List of human-readable validation error strings.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Dataset validation failed: {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnTypeInfo:
    """Type-detection result for a single column."""

    name: str
    detected_type: str  # "numeric" | "categorical" | "datetime" | "boolean" | "text" | "id"
    dtype: str  # original pandas dtype string
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    sample_values: list[Any]  # up to 5 random non-null values

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON storage in ``schema_json``."""
        return {
            "name": self.name,
            "type": self.detected_type,
            "dtype": self.dtype,
            "null_pct": round(self.null_pct, 2),
            "unique_count": self.unique_count,
            "unique_pct": round(self.unique_pct, 2),
            "sample_values": [str(v) for v in self.sample_values],
        }


@dataclass(slots=True)
class IngestionResult:
    """Structured result returned by :meth:`DataIngestionService.ingest`."""

    dataset_id: str  # UUID as string
    file_path: str  # artefact store path
    checksum: str  # SHA-256 hex digest
    row_count: int
    col_count: int
    target_column: str
    time_column: str | None
    columns: list[ColumnTypeInfo]
    warnings: list[str] = field(default_factory=list)
    schema_json: dict[str, Any] = field(default_factory=dict)

    @property
    def detected_task(self) -> str:
        """``"classification"`` or ``"regression"`` based on schema_json."""
        return self.schema_json.get("detected_task", "regression")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DataIngestionService:
    """Stateless service orchestrating the full CSV → DB ingestion pipeline.

    Each call to :meth:`ingest` is independent.  Configuration is read via
    :func:`ive.config.get_settings`, the artifact store via
    :func:`ive.storage.artifact_store.get_artifact_store`, and the database
    session is passed explicitly.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = get_artifact_store()
        self.logger = logger

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def ingest(
        self,
        file_content: bytes,
        filename: str,
        target_column: str,
        time_column: str | None = None,
        session: AsyncSession | None = None,
    ) -> IngestionResult:
        """Run the full ingestion pipeline.

        Args:
            file_content:  Raw CSV bytes.
            filename:      Original upload filename.
            target_column: Name of the target / label column.
            time_column:   Optional datetime column for temporal analysis.
            session:       Active ``AsyncSession`` (required for DB writes).

        Returns:
            :class:`IngestionResult` containing all metadata and the dataset ID.

        Raises:
            DatasetValidationError: When the dataset fails validation.
            ValueError:             On duplicate file (same checksum).
        """
        self.logger.info(
            "ingestion.start",
            filename=filename,
            size_bytes=len(file_content),
            target=target_column,
        )

        # ── 1. Parse CSV ───────────────────────────────────────────────
        df = self._parse_csv(file_content, filename)
        self.logger.info(
            "ingestion.parsed",
            rows=len(df),
            cols=len(df.columns),
        )

        # ── 2. Detect column types ─────────────────────────────────────
        columns = self._detect_column_types(df)
        self.logger.debug("ingestion.types_detected", columns=len(columns))

        # ── 3. Validate ────────────────────────────────────────────────
        warnings = self._validate(df, columns, target_column, time_column, len(file_content))

        # ── 4. Store file ──────────────────────────────────────────────
        file_path, checksum = await self._store_file(file_content, filename, session)

        # ── 5. Build schema_json ───────────────────────────────────────
        schema = self._build_schema(df, columns, target_column, time_column)

        # ── 6. Create DB record ────────────────────────────────────────
        dataset_id: str | None = None
        if session is not None:
            dataset_id = await self._create_db_record(
                session=session,
                filename=filename,
                file_path=file_path,
                file_size_bytes=len(file_content),
                target_column=target_column,
                time_column=time_column,
                checksum=checksum,
                row_count=len(df),
                col_count=len(df.columns),
                schema_json=schema,
            )
        else:
            dataset_id = str(uuid.uuid4())
            self.logger.warning("ingestion.no_session", note="DB record not created")

        # ── 7. Assemble result ─────────────────────────────────────────
        result = IngestionResult(
            dataset_id=dataset_id,
            file_path=file_path,
            checksum=checksum,
            row_count=len(df),
            col_count=len(df.columns),
            target_column=target_column,
            time_column=time_column,
            columns=columns,
            warnings=warnings,
            schema_json=schema,
        )

        self.logger.info(
            "ingestion.complete",
            dataset_id=dataset_id,
            rows=result.row_count,
            cols=result.col_count,
            task=result.detected_task,
            warnings=len(warnings),
        )
        return result

    # ------------------------------------------------------------------
    # Step 1: CSV Parsing
    # ------------------------------------------------------------------

    def _parse_csv(self, file_content: bytes, filename: str) -> pd.DataFrame:
        """Parse raw CSV bytes into a cleaned Pandas DataFrame.

        Tries multiple encodings, auto-detects the delimiter, strips BOM,
        and removes fully empty rows/columns.  Large files (>50 MB) are
        parsed with Polars then converted to Pandas for uniformity.

        Args:
            file_content: Raw bytes of the CSV file.
            filename:     Original filename (for logging only).

        Returns:
            A cleaned ``pd.DataFrame``.

        Raises:
            DatasetValidationError: If the file cannot be parsed at all.
        """
        if not file_content or len(file_content.strip()) == 0:
            raise DatasetValidationError(["File is empty — no data to ingest."])

        # ── Detect encoding ────────────────────────────────────────────
        text: str | None = None
        used_encoding: str = "utf-8"
        for enc in _ENCODINGS:
            try:
                text = file_content.decode(enc)
                used_encoding = enc
                break
            except (UnicodeDecodeError, ValueError):
                continue

        if text is None:
            raise DatasetValidationError(
                ["Could not decode file with any supported encoding " f"({', '.join(_ENCODINGS)})."]
            )

        # Strip UTF-8 BOM if present
        if text.startswith("\ufeff"):
            text = text[1:]

        self.logger.debug("ingestion.encoding", encoding=used_encoding)

        # ── Detect delimiter ───────────────────────────────────────────
        delimiter = ","
        try:
            sniff_sample = text[:_SNIFF_BYTES]
            dialect = csv.Sniffer().sniff(sniff_sample, delimiters=",;\t|")
            delimiter = dialect.delimiter
        except csv.Error:
            self.logger.debug("ingestion.sniffer_fallback", delimiter=",")

        self.logger.debug("ingestion.delimiter", delimiter=repr(delimiter))

        # ── Parse ──────────────────────────────────────────────────────
        use_polars = len(file_content) > _LARGE_FILE_THRESHOLD

        if use_polars:
            df = self._parse_with_polars(file_content, delimiter, used_encoding)
        else:
            df = self._parse_with_pandas(text, delimiter)

        # ── Clean ──────────────────────────────────────────────────────
        df.columns = df.columns.str.strip()

        # Remove fully empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        # Remove duplicate column names (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        if df.empty or len(df.columns) == 0:
            raise DatasetValidationError(
                ["File contains no usable data after removing empty rows/columns."]
            )

        return df

    def _parse_with_pandas(self, text: str, delimiter: str) -> pd.DataFrame:
        """Parse CSV text using Pandas."""
        try:
            return pd.read_csv(
                io.StringIO(text),
                delimiter=delimiter,
                engine="python",
                on_bad_lines="warn",
                skip_blank_lines=True,
            )
        except Exception as exc:
            raise DatasetValidationError([f"Failed to parse CSV with Pandas: {exc}"]) from exc

    def _parse_with_polars(self, raw: bytes, delimiter: str, encoding: str) -> pd.DataFrame:
        """Parse large CSV using Polars and convert to Pandas."""
        try:
            import polars as pl

            df_pl = pl.read_csv(
                raw,
                separator=delimiter,
                encoding=encoding if encoding in ("utf-8", "utf-8-sig") else "utf8",
                ignore_errors=True,
                truncate_ragged_lines=True,
            )
            return df_pl.to_pandas()
        except ImportError:
            self.logger.warning("ingestion.polars_unavailable", fallback="pandas")
            text = raw.decode(encoding, errors="replace")
            return self._parse_with_pandas(text, delimiter)
        except Exception as exc:
            raise DatasetValidationError([f"Failed to parse CSV with Polars: {exc}"]) from exc

    # ------------------------------------------------------------------
    # Step 2: Column Type Detection
    # ------------------------------------------------------------------

    def _detect_column_types(self, df: pd.DataFrame) -> list[ColumnTypeInfo]:
        """Detect the semantic type of every column in the DataFrame.

        Detection order (first match wins):
            1. ID — column name matches id/index/key pattern AND >90 % unique
            2. Datetime — parseable as datetime with >80 % success rate
            3. Boolean — values ⊆ {0,1,True,False,yes,no,Y,N}
            4. Numeric — int64 / float64 dtype
            5. Categorical — object/string with <50 unique OR <5 % unique%
            6. Text — everything else (high-cardinality strings)

        Args:
            df: Cleaned DataFrame.

        Returns:
            A ``ColumnTypeInfo`` per column.
        """
        results: list[ColumnTypeInfo] = []
        n_rows = len(df)

        for col_name in df.columns:
            series = df[col_name]
            dtype_str = str(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0.0
            non_null = series.dropna()
            unique_count = int(non_null.nunique())
            unique_pct = (unique_count / len(non_null) * 100) if len(non_null) > 0 else 0.0

            # Sample values
            sample: list[Any] = []
            if len(non_null) > 0:
                sample_n = min(_SAMPLE_VALUES, len(non_null))
                sample = non_null.sample(n=sample_n, random_state=42).tolist()

            detected = self._classify_column(
                col_name, series, dtype_str, unique_count, unique_pct, n_rows
            )

            results.append(
                ColumnTypeInfo(
                    name=col_name,
                    detected_type=detected,
                    dtype=dtype_str,
                    null_count=null_count,
                    null_pct=round(null_pct, 2),
                    unique_count=unique_count,
                    unique_pct=round(unique_pct, 2),
                    sample_values=sample,
                )
            )

        self.logger.debug(
            "ingestion.types_summary",
            types={
                ct.detected_type: sum(1 for c in results if c.detected_type == ct.detected_type)
                for ct in results
            },
        )
        return results

    def _classify_column(
        self,
        name: str,
        series: pd.Series,
        dtype_str: str,
        unique_count: int,
        unique_pct: float,
        n_rows: int,
    ) -> str:
        """Return the semantic type string for a single column."""
        non_null = series.dropna()

        # 1. ID columns
        if _ID_PATTERNS.search(name) and unique_pct > 90.0:
            return "id"

        # 2. Datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if series.dtype == object and len(non_null) > 0:
            try:
                converted = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
                success_rate = converted.notna().sum() / len(non_null)
                if success_rate >= _DATETIME_SUCCESS_RATE:
                    return "datetime"
            except Exception:
                pass

        # 3. Boolean
        if series.dtype == bool or series.dtype == "boolean":
            return "boolean"
        if series.dtype == object and len(non_null) > 0:
            lower_vals = {str(v).strip().lower() for v in non_null.unique()}
            if lower_vals and lower_vals <= _BOOLEAN_PAIRS:
                return "boolean"
        if pd.api.types.is_numeric_dtype(series) and len(non_null) > 0:
            vals = set(non_null.unique())
            if vals <= {0, 1, 0.0, 1.0}:
                return "boolean"

        # 4. Numeric
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        # 5. Categorical vs Text
        if unique_count < _MAX_CATEGORICAL_UNIQUE or unique_pct < _CATEGORICAL_UNIQUE_PCT:
            return "categorical"

        return "text"

    # ------------------------------------------------------------------
    # Step 3: Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        df: pd.DataFrame,
        columns: list[ColumnTypeInfo],
        target_column: str,
        time_column: str | None,
        file_size: int,
    ) -> list[str]:
        """Validate the dataset against business rules.

        Collects **all** errors before raising, so the user can fix
        everything in a single iteration.

        Args:
            df:            The parsed DataFrame.
            columns:       Column type info list.
            target_column: User-specified target column name.
            time_column:   Optional datetime column name.
            file_size:     Raw file size in bytes.

        Returns:
            List of non-fatal warning strings.

        Raises:
            DatasetValidationError: If any hard errors are found.
        """
        errors: list[str] = []
        warnings: list[str] = []

        # ── Row count ────────────────────────────────────────────────
        if len(df) < _MIN_ROWS:
            errors.append(f"Dataset has {len(df)} rows — minimum is {_MIN_ROWS}.")

        # ── Feature column count (excluding target) ──────────────────
        feature_cols = [c for c in df.columns if c != target_column]
        if len(feature_cols) < _MIN_FEATURE_COLS:
            errors.append(
                f"Dataset needs at least {_MIN_FEATURE_COLS} feature columns "
                f"(besides '{target_column}'); found {len(feature_cols)}."
            )

        # ── Target column existence ──────────────────────────────────
        if target_column not in df.columns:
            errors.append(
                f"Target column '{target_column}' not found. "
                f"Available columns: {list(df.columns[:20])}"
            )
        else:
            target = df[target_column]
            if target.isna().all():
                errors.append(f"Target column '{target_column}' is entirely null.")
            elif target.dropna().nunique() < 2:
                errors.append(
                    f"Target column '{target_column}' has no variance "
                    f"(only one unique value: {target.dropna().unique()[:3].tolist()})."
                )

        # ── Time column ──────────────────────────────────────────────
        if time_column is not None:
            if time_column not in df.columns:
                errors.append(f"Time column '{time_column}' not found in dataset.")
            else:
                try:
                    converted = pd.to_datetime(df[time_column], errors="coerce")
                    if converted.isna().all():
                        errors.append(
                            f"Time column '{time_column}' could not be parsed as datetime."
                        )
                except Exception:
                    errors.append(f"Time column '{time_column}' is not parseable as datetime.")

        # ── Duplicate column names ───────────────────────────────────
        dup_cols = [c for c in df.columns[df.columns.duplicated()]]
        if dup_cols:
            errors.append(f"Duplicate column names detected: {dup_cols}")

        # ── File size ────────────────────────────────────────────────
        if file_size > _MAX_FILE_BYTES:
            errors.append(
                f"File size ({file_size / (1024 * 1024):.0f} MB) exceeds "
                f"maximum ({_MAX_FILE_BYTES / (1024 * 1024):.0f} MB)."
            )

        # ── At least some numeric/categorical features ───────────────
        usable_types = {"numeric", "categorical", "boolean"}
        usable = [c for c in columns if c.detected_type in usable_types and c.name != target_column]
        if len(usable) == 0:
            errors.append(
                "No numeric, categorical, or boolean feature columns detected. "
                "At least one is required."
            )

        # ── High-null warnings ───────────────────────────────────────
        for col_info in columns:
            if col_info.null_pct > _HIGH_NULL_PCT:
                warnings.append(
                    f"Column '{col_info.name}' is {col_info.null_pct:.1f}% null "
                    f"— may be dropped during preprocessing."
                )

        # ── Raise if errors ──────────────────────────────────────────
        if errors:
            self.logger.error("ingestion.validation_failed", errors=errors)
            raise DatasetValidationError(errors)

        if warnings:
            self.logger.warning("ingestion.validation_warnings", warnings=warnings)

        return warnings

    # ------------------------------------------------------------------
    # Step 4: Store File
    # ------------------------------------------------------------------

    async def _store_file(
        self,
        file_content: bytes,
        filename: str,
        session: AsyncSession | None,
    ) -> tuple[str, str]:
        """Persist the raw file and return ``(file_path, checksum)``.

        Performs SHA-256 checksum dedup if a session is available.

        Args:
            file_content: Raw bytes of the CSV file.
            filename:     Original upload filename.
            session:      Optional DB session for dedup checks.

        Returns:
            Tuple of ``(artefact_store_path, sha256_hex_digest)``.

        Raises:
            ValueError: If a dataset with the same checksum already exists.
        """
        checksum = compute_checksum(file_content)
        self.logger.debug("ingestion.checksum", checksum=checksum[:12] + "...")

        # Duplicate detection
        if session is not None:
            repo = DatasetRepository(session, Dataset)
            existing = await repo.get_by_checksum(checksum)
            if existing is not None:
                raise ValueError(
                    f"Duplicate file detected — identical file already uploaded "
                    f"as dataset '{existing.name}' (id={existing.id})."
                )

        file_path = await self.store.save_file(
            file_content,
            category="datasets",
            filename=filename,
        )
        return file_path, checksum

    # ------------------------------------------------------------------
    # Step 5: Build schema_json
    # ------------------------------------------------------------------

    def _build_schema(
        self,
        df: pd.DataFrame,
        columns: list[ColumnTypeInfo],
        target_column: str,
        time_column: str | None,
    ) -> dict[str, Any]:
        """Build the ``schema_json`` dict stored in the ``Dataset`` DB record.

        Includes per-column type info, target statistics, and the
        auto-detected task type (regression vs classification).

        Args:
            df:            The parsed DataFrame.
            columns:       Column type info list.
            target_column: Name of the target column.
            time_column:   Optional time column name.

        Returns:
            Serialisable dict for ``Dataset.schema_json``.
        """
        # Target statistics
        target_stats: dict[str, Any] = {}
        if target_column in df.columns:
            target = df[target_column].dropna()
            target_col_info = next((c for c in columns if c.name == target_column), None)
            target_type = target_col_info.detected_type if target_col_info else "numeric"

            if pd.api.types.is_numeric_dtype(target):
                target_stats = {
                    "mean": float(target.mean()),
                    "std": float(target.std()),
                    "min": float(target.min()),
                    "max": float(target.max()),
                    "median": float(target.median()),
                }
            elif target.dtype == object:
                value_counts = target.value_counts().head(10)
                target_stats = {
                    "value_counts": value_counts.to_dict(),
                }

        # Task detection
        detected_task = self._detect_task(df, target_column)

        return {
            "columns": [c.to_dict() for c in columns],
            "target": {
                "name": target_column,
                "type": target_type if target_column in df.columns else "unknown",
                "stats": target_stats,
            },
            "time_column": time_column,
            "detected_task": detected_task,
            "row_count": len(df),
            "col_count": len(df.columns),
        }

    def _detect_task(self, df: pd.DataFrame, target_column: str) -> str:
        """Infer whether the task is classification or regression.

        Heuristic: if the target has ≤ 20 unique values AND is integer /
        categorical then it's classification.  Otherwise regression.

        Args:
            df:            The parsed DataFrame.
            target_column: Name of the target column.

        Returns:
            ``"classification"`` or ``"regression"``.
        """
        if target_column not in df.columns:
            return "regression"
        target = df[target_column].dropna()
        n_unique = target.nunique()
        is_int_like = pd.api.types.is_integer_dtype(target) or (
            pd.api.types.is_float_dtype(target) and (target == target.astype(int)).all()
        )
        if n_unique <= _MAX_CLASSIFICATION_UNIQUE and (is_int_like or target.dtype == object):
            return "classification"
        return "regression"

    # ------------------------------------------------------------------
    # Step 6: Create DB Record
    # ------------------------------------------------------------------

    async def _create_db_record(
        self,
        *,
        session: AsyncSession,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        target_column: str,
        time_column: str | None,
        checksum: str,
        row_count: int,
        col_count: int,
        schema_json: dict,
    ) -> str:
        """Create a ``Dataset`` row in the database.

        Args:
            session:         Active ``AsyncSession``.
            filename:        Original filename.
            file_path:       Artefact store path.
            file_size_bytes: Raw file size.
            target_column:   Target column name.
            time_column:     Optional time column.
            checksum:        SHA-256 hex digest.
            row_count:       Number of rows.
            col_count:       Number of columns.
            schema_json:     Schema metadata dict.

        Returns:
            Dataset UUID as a string.
        """
        repo = DatasetRepository(session, Dataset)
        dataset = await repo.create(
            name=filename,
            original_filename=filename,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            target_column=target_column,
            time_column=time_column,
            checksum=checksum,
            row_count=row_count,
            col_count=col_count,
            schema_json=schema_json,
        )
        self.logger.info("ingestion.db_record_created", dataset_id=str(dataset.id))
        return str(dataset.id)
