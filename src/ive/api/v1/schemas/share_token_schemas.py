"""Pydantic schemas for share-token endpoints (Phase C2.2)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ShareTokenCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expires_in_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description="Lifetime in days from issuance. Hard cap 365.",
    )
    passphrase: str | None = Field(
        default=None,
        min_length=4,
        max_length=200,
        description=(
            "Optional passphrase challenge. When set, the share-read "
            "endpoint requires it as a header before serving the report."
        ),
    )


class ShareTokenIssuedResponse(BaseModel):
    """Response on token issuance — the raw ``token`` is shown ONCE."""

    id: uuid.UUID
    token: str
    expires_at: datetime
    has_passphrase: bool


class ShareTokenSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    expires_at: datetime
    revoked_at: datetime | None
    has_passphrase: bool
    created_by_name: str | None
    created_at: datetime


class ShareTokenListResponse(BaseModel):
    items: list[ShareTokenSummary]
    total: int
