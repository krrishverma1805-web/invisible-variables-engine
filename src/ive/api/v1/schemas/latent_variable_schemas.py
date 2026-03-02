"""
Latent Variable API Schemas — Invisible Variables Engine.

STUB — Phase 2.  Defines the shape of latent variable response models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class LatentVariableSummary(BaseModel):
    """Compact latent variable summary for list responses."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_id: UUID
    name: str
    status: str          # "candidate" | "validated" | "rejected"
    importance_score: float
    stability_score: float
    bootstrap_presence_rate: float
    created_at: datetime


class LatentVariableDetail(BaseModel):
    """Full latent variable detail response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_id: UUID
    name: str
    description: str
    construction_rule: dict[str, Any]
    source_pattern_ids: list[UUID]
    importance_score: float
    stability_score: float
    bootstrap_presence_rate: float
    model_improvement_pct: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    explanation_text: str
    status: str
    created_at: datetime


class LatentVariableListResponse(BaseModel):
    """Paginated list of latent variables for an experiment."""

    experiment_id: UUID
    latent_variables: list[LatentVariableSummary]
    total: int
    validated_count: int
    skip: int
    limit: int
