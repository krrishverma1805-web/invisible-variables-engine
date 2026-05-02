"""Pydantic schemas for the LV annotation endpoints (Phase C2.1)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_body(value: str) -> str:
    """Strip leading/trailing whitespace and reject empty/whitespace-only bodies.

    Audit fix: ``min_length=1`` alone accepts ``"   "`` which is semantically
    empty. Strip first, then re-check length.
    """
    if not isinstance(value, str):
        raise ValueError("body must be a string")
    stripped = value.strip()
    if not stripped:
        raise ValueError("body must not be empty or whitespace-only")
    if len(stripped) > 10_000:
        raise ValueError("body exceeds 10,000 characters after trimming")
    return stripped


class AnnotationCreate(BaseModel):
    """Request body for ``POST /latent-variables/{lv_id}/annotations``."""

    model_config = ConfigDict(extra="forbid")

    # No min_length / max_length on Field — `_normalize_body` trims
    # first then validates length. A pre-trim max_length=10_000 would
    # reject ``"  " + "x"*10000 + "  "`` even though the trimmed body
    # fits (audit fix). A defensive upper bound (1MB) blocks pathological
    # input from tying up the validator.
    body: str = Field(..., max_length=1_000_000)

    @field_validator("body", mode="after")
    @classmethod
    def _trim(cls, v: str) -> str:
        return _normalize_body(v)


class AnnotationUpdate(BaseModel):
    """Request body for ``PUT /latent-variables/{lv_id}/annotations/{id}``."""

    model_config = ConfigDict(extra="forbid")

    # No min_length / max_length on Field — `_normalize_body` trims
    # first then validates length. A pre-trim max_length=10_000 would
    # reject ``"  " + "x"*10000 + "  "`` even though the trimmed body
    # fits (audit fix). A defensive upper bound (1MB) blocks pathological
    # input from tying up the validator.
    body: str = Field(..., max_length=1_000_000)

    @field_validator("body", mode="after")
    @classmethod
    def _trim(cls, v: str) -> str:
        return _normalize_body(v)


class AnnotationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    latent_variable_id: uuid.UUID
    body: str
    api_key_name: str | None
    created_at: datetime
    updated_at: datetime


class AnnotationListResponse(BaseModel):
    items: list[AnnotationResponse]
    total: int
