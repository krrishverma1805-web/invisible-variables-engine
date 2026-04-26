"""
Dataset API Schemas — Invisible Variables Engine.

Pydantic models for the dataset upload, list, detail, and profile endpoints.
All models used as FastAPI response bodies are annotated with
``model_config = ConfigDict(from_attributes=True)`` so they can be
constructed directly from SQLAlchemy ORM instances.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Column-level info (part of dataset detail response)
# ---------------------------------------------------------------------------


class ColumnInfo(BaseModel):  # type: ignore[misc]
    """Compact per-column metadata returned in the dataset detail response."""

    name: str
    detected_type: str  # "numeric" | "categorical" | "datetime" | "boolean" | "text" | "id"
    dtype: str  # original pandas dtype string, e.g. "float64"
    null_pct: float
    unique_count: int
    sample_values: list[Any] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core dataset response
# ---------------------------------------------------------------------------


class DatasetResponse(BaseModel):  # type: ignore[misc]
    """Full dataset metadata returned by upload, detail, and list endpoints."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    original_filename: str
    row_count: int
    col_count: int
    target_column: str
    time_column: str | None
    file_size_bytes: int
    checksum: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    quality_score: float | None = None
    created_at: datetime

    @classmethod
    def from_dataset(cls, ds: Any) -> DatasetResponse:
        """Construct from a ``Dataset`` ORM instance.

        Extracts ``columns`` and ``quality_score`` from ``schema_json``.
        """
        schema: dict[str, Any] = ds.schema_json or {}
        raw_cols: list[dict[str, Any]] = schema.get("columns", [])
        columns = [
            ColumnInfo(
                name=c.get("name", ""),
                detected_type=c.get("type", "unknown"),
                dtype=c.get("dtype", ""),
                null_pct=c.get("null_pct", 0.0),
                unique_count=c.get("unique_count", 0),
                sample_values=c.get("sample_values", []),
            )
            for c in raw_cols
        ]
        return cls(
            id=ds.id,
            name=ds.name,
            original_filename=ds.original_filename,
            row_count=ds.row_count,
            col_count=ds.col_count,
            target_column=ds.target_column,
            time_column=ds.time_column,
            file_size_bytes=ds.file_size_bytes,
            checksum=ds.checksum,
            columns=columns,
            quality_score=schema.get("quality_score"),
            created_at=ds.created_at,
        )


# ---------------------------------------------------------------------------
# List response
# ---------------------------------------------------------------------------


class DatasetListResponse(BaseModel):  # type: ignore[misc]
    """Paginated list of datasets."""

    datasets: list[DatasetResponse]
    total: int
    skip: int
    limit: int


# ---------------------------------------------------------------------------
# Profile response
# ---------------------------------------------------------------------------


class DatasetProfileResponse(BaseModel):  # type: ignore[misc]
    """Statistical profile returned by ``GET /datasets/{id}/profile``."""

    dataset_id: UUID
    row_count: int
    col_count: int
    memory_usage_mb: float
    target_stats: dict[str, Any]
    column_profiles: list[dict[str, Any]]
    quality_score: float
    quality_issues: list[dict[str, Any]]
    recommendations: list[str]
    top_correlations: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Delete response
# ---------------------------------------------------------------------------


class DeleteResponse(BaseModel):  # type: ignore[misc]
    """Acknowledgment returned after a successful delete."""

    message: str
    dataset_id: UUID
