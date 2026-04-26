"""
Latent Variable API Schemas — Invisible Variables Engine.

Pydantic v2 schemas for latent variable request/response models.
All ORM-backed schemas use ``model_config = ConfigDict(from_attributes=True)``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class LatentVariableResponse(BaseModel):  # type: ignore[misc]
    """Single latent variable — maps directly from ``LatentVariable`` ORM model."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_id: UUID
    name: str
    description: str
    construction_rule: dict[str, Any]
    importance_score: float
    stability_score: float
    bootstrap_presence_rate: float
    explanation_text: str
    status: str
    created_at: datetime


class LatentVariableListResponse(BaseModel):  # type: ignore[misc]
    """Paginated list of latent variables."""

    variables: list[LatentVariableResponse]
    total: int
    skip: int
    limit: int


# Legacy aliases used by the existing latent_variables.py stub
LatentVariableDetail = LatentVariableResponse
