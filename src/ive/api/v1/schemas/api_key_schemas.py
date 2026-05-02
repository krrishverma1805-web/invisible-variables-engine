"""Pydantic request/response schemas for the api-key admin endpoints.

The raw key value is **only** returned in the create / rotate responses —
never re-shown afterwards (the DB stores only the SHA-256 hash).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ScopeLiteral = Literal["read", "write", "admin"]


class APIKeyCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=3, max_length=64, description="Human-readable label.")
    scopes: list[ScopeLiteral] = Field(
        default_factory=lambda: ["read", "write"],
        description="One or more of read / write / admin.",
    )
    rate_limit: int = Field(default=100, ge=1, le=100_000)
    expires_at: datetime | None = Field(default=None, description="UTC expiry; null = never.")


class APIKeyResponse(BaseModel):
    """API-key metadata. Does NOT include the raw key."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    scopes: list[str]
    rate_limit: int
    is_active: bool
    created_at: datetime
    created_by: str | None = None
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    last_rotated_at: datetime | None = None


class APIKeyCreatedResponse(APIKeyResponse):
    """Returned **only** at creation/rotation time — includes the raw key."""

    raw_key: str = Field(description="Raw key — show once; not retrievable later.")


class APIKeyListResponse(BaseModel):
    items: list[APIKeyResponse]
    total: int
