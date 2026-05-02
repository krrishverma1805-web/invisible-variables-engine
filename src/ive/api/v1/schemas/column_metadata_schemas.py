"""Pydantic schemas for the dataset column metadata endpoints.

Per plan §142 / §174 / §203.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SensitivityLiteral = Literal["public", "non_public"]


class ColumnMetadataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    column_name: str
    sensitivity: SensitivityLiteral
    created_at: datetime
    updated_at: datetime


class ColumnMetadataListResponse(BaseModel):
    items: list[ColumnMetadataResponse]
    total: int
    public_count: int = Field(description="How many columns are marked public.")


class ColumnSensitivityUpdate(BaseModel):
    """Single column update entry."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(min_length=1, max_length=255)
    sensitivity: SensitivityLiteral


class BulkSensitivityUpdate(BaseModel):
    """Body of ``PUT /datasets/{id}/columns``.

    Each entry overrides the current sensitivity for one column. Columns
    not listed are left unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    updates: list[ColumnSensitivityUpdate] = Field(min_length=1, max_length=1000)
