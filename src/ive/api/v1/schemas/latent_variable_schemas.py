"""
Pydantic schemas for Latent Variable API responses.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class ValidationResults(BaseModel):
    """Statistical validation results for a latent variable."""

    bootstrap_stability: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    holdout_improvement: float | None = None
    n_bootstrap_iterations: int = 1000
    ci_lower: float | None = None
    ci_upper: float | None = None


class LatentVariableSummary(BaseModel):
    """Compact latent variable for list responses."""

    id: uuid.UUID
    experiment_id: uuid.UUID
    rank: int
    name: str | None = None
    confidence_score: float
    effect_size: float
    coverage_pct: float
    created_at: datetime

    model_config = {"from_attributes": True}


class LatentVariableDetailResponse(LatentVariableSummary):
    """Full latent variable details."""

    description: str | None = None
    explanation: str | None = None
    candidate_features: list[str] = Field(default_factory=list)
    validation: ValidationResults | None = None
    feature_importance: dict[str, float] = Field(
        default_factory=dict,
        description="SHAP-based importance scores for candidate features.",
    )
    cluster_stats: dict[str, float] = Field(
        default_factory=dict,
        description="Cluster-level statistical summary.",
    )


class LatentVariableListResponse(BaseModel):
    """List of latent variables for an experiment."""

    experiment_id: uuid.UUID
    items: list[LatentVariableSummary]


class LatentVariableExplanationResponse(BaseModel):
    """Natural language explanation for a latent variable."""

    id: uuid.UUID
    name: str | None = None
    explanation: str
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Bullet-point evidence supporting this explanation.",
    )
    confidence_score: float
