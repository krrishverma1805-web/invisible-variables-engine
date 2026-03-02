"""
Experiment API Schemas — Invisible Variables Engine.

STUB — Phase 2.  Defines the shape of experiment request/response models.
Endpoints are not yet implemented.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ExperimentConfig(BaseModel):
    """Configuration for a single IVE experiment run.

    TODO: implement full validation in Phase 2.
    """

    # Model config
    model_types: list[str] = Field(
        default=["linear", "xgboost"],
        description="Model types to train (linear, xgboost, lightgbm).",
    )
    n_cv_folds: int = Field(default=5, ge=2, le=20)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42)

    # Detection thresholds
    min_subgroup_size: int = Field(default=30)
    significance_level: float = Field(default=0.05)
    effect_size_threshold: float = Field(default=0.3)

    # Bootstrap validation
    n_bootstrap_iterations: int = Field(default=100)
    bootstrap_presence_threshold: float = Field(default=0.7)

    # Feature limits
    max_features: int = Field(default=100)
    shap_sample_size: int = Field(default=500)


class ExperimentCreateRequest(BaseModel):
    """Request body for creating a new experiment."""

    dataset_id: UUID
    config: ExperimentConfig = Field(default_factory=ExperimentConfig)


class ExperimentResponse(BaseModel):
    """Full experiment detail response.

    TODO: implement in Phase 2.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_id: UUID
    status: str
    progress_pct: int
    current_stage: str | None
    error_message: str | None
    celery_task_id: str | None
    config_json: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ExperimentListResponse(BaseModel):
    """Paginated experiment list."""

    experiments: list[ExperimentResponse]
    total: int
    skip: int
    limit: int


class ExperimentCreateResponse(BaseModel):
    """Minimal response on experiment creation (202 Accepted)."""

    id: UUID
    status: str
    message: str


class ExperimentProgressResponse(BaseModel):
    """Lightweight progress-poll response."""

    id: UUID
    status: str
    progress_pct: int
    current_stage: str | None
