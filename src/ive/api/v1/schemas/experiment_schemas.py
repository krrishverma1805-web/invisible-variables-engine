"""
Pydantic schemas for Experiment API requests and responses.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Configuration for an IVE experiment run."""

    model_types: list[Literal["linear", "xgboost"]] = Field(
        default=["linear", "xgboost"],
        description="ML model types to use for residual computation.",
    )
    cv_folds: int = Field(default=5, ge=2, le=20, description="Cross-validation folds.")
    random_seed: int = Field(default=42)
    min_cluster_size: int = Field(default=10, ge=2, description="HDBSCAN minimum cluster size.")
    shap_sample_size: int = Field(default=500, ge=50, description="Max samples for SHAP.")
    max_latent_variables: int = Field(default=5, ge=1, le=20)
    feature_selection_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Minimum SHAP importance to include a feature.",
    )


class ExperimentCreateRequest(BaseModel):
    """Request body for creating a new experiment."""

    dataset_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=255)
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)


class ExperimentCreateResponse(BaseModel):
    """Response after successfully queueing an experiment."""

    id: uuid.UUID
    dataset_id: uuid.UUID
    name: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    task_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"from_attributes": True}


class PhaseStatus(BaseModel):
    """Status of a single pipeline phase."""

    status: Literal["pending", "running", "completed", "failed"]
    duration_s: float | None = None
    error_msg: str | None = None


class ExperimentDetailResponse(ExperimentCreateResponse):
    """Full experiment details including per-phase progress."""

    current_phase: str | None = None
    progress_pct: int = 0
    phases: dict[str, PhaseStatus] = Field(default_factory=dict)
    error_msg: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExperimentSummary(BaseModel):
    """Compact experiment summary for list responses."""

    id: uuid.UUID
    dataset_id: uuid.UUID
    name: str
    status: str
    current_phase: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    """Paginated list of experiments."""

    items: list[ExperimentSummary]
    total: int
    page: int
    page_size: int
