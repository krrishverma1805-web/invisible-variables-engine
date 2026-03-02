"""
Pydantic schemas for Dataset API requests and responses.

These schemas define the shape of data entering and leaving the API layer.
They are separate from the SQLAlchemy ORM models (db/models.py).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DatasetCreateResponse(BaseModel):
    """Response after successfully uploading a dataset."""

    id: uuid.UUID
    name: str
    target_column: str
    description: str | None = None
    row_count: int
    column_count: int
    status: Literal["uploaded", "profiling", "profiled", "error"]
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class DatasetSummary(BaseModel):
    """Compact dataset summary for list responses."""

    id: uuid.UUID
    name: str
    target_column: str
    row_count: int
    column_count: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    """Paginated list of datasets."""

    items: list[DatasetSummary]
    total: int
    page: int
    page_size: int


class ColumnProfile(BaseModel):
    """Statistical profile of a single column."""

    name: str
    dtype: str
    missing_pct: float
    unique_count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    top_values: list[str] = Field(default_factory=list)


class DatasetDetailResponse(DatasetCreateResponse):
    """Full dataset details including column profiles."""

    file_path: str | None = None
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    correlation_issues: list[str] = Field(default_factory=list)
