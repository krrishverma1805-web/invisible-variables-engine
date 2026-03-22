"""
Experiment API Schemas — Invisible Variables Engine.

Pydantic v2 schemas for all experiment-related request/response models.
All ORM-backed schemas use ``model_config = ConfigDict(from_attributes=True)``
so they can be constructed directly from SQLAlchemy model instances.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Allowed analysis modes.
AnalysisMode = Literal["demo", "production"]

# ---------------------------------------------------------------------------
# Config (embedded in create request and stored as config_json)
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """Configuration for a single IVE experiment run."""

    model_types: list[str] = Field(
        default=["linear", "xgboost"],
        description="Model types to train (linear, xgboost).",
    )
    cv_folds: int = Field(default=5, ge=2, le=20, alias="n_cv_folds")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = Field(default=42)

    min_subgroup_size: int = Field(default=30)
    significance_level: float = Field(default=0.05)
    effect_size_threshold: float = Field(default=0.3)

    bootstrap_iterations: int = Field(default=50, ge=10, le=500, alias="n_bootstrap_iterations")
    bootstrap_presence_threshold: float = Field(default=0.7)

    max_features: int = Field(default=100)
    shap_sample_size: int = Field(default=500)

    analysis_mode: AnalysisMode = Field(
        default="demo",
        description=(
            "Analysis mode: 'demo' applies permissive thresholds for exploration; "
            "'production' applies stricter thresholds to reduce false positives."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ExperimentCreate(BaseModel):
    """Request body for POST /experiments/."""

    dataset_id: UUID
    config: dict[str, Any] = Field(
        default_factory=lambda: {
            "model_types": ["linear", "xgboost"],
            "cv_folds": 5,
            "bootstrap_iterations": 50,
            "analysis_mode": "demo",
        }
    )


# Keep legacy name so existing import in experiments.py stub still works
ExperimentCreateRequest = ExperimentCreate


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class ExperimentResponse(BaseModel):
    """Full experiment detail — maps directly from the Experiment ORM model."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    dataset_id: UUID
    status: str
    progress_pct: int
    current_stage: str | None = None
    error_message: str | None = None
    celery_task_id: str | None = None
    config_json: dict[str, Any]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ExperimentCreateResponse(BaseModel):
    """Minimal 201 Created response body."""

    id: UUID
    status: str
    celery_task_id: str | None = None
    message: str


class ExperimentListResponse(BaseModel):
    """Paginated experiment list."""

    experiments: list[ExperimentResponse]
    total: int
    skip: int
    limit: int


class ExperimentProgressResponse(BaseModel):
    """Lightweight progress-poll response for WebSocket/polling clients."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    progress_pct: int
    current_stage: str | None = None


# ---------------------------------------------------------------------------
# Error pattern response
# ---------------------------------------------------------------------------


class ErrorPatternResponse(BaseModel):
    """A statistically significant pattern discovered in the residuals."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_id: UUID
    pattern_type: str
    subgroup_definition: dict[str, Any]
    effect_size: float
    p_value: float
    adjusted_p_value: float
    sample_count: int
    mean_residual: float
    std_residual: float
    created_at: datetime


class ExperimentSummaryResponse(BaseModel):
    """Compact experiment summary — headline, counts, and recommendations."""

    headline: str
    patterns_found: int
    validated_variables: int
    rejected_variables: int
    summary_text: str
    top_findings: list[str]
    recommendations: list[str]
    analysis_mode: str = Field(
        default="demo", description="Analysis mode used for this experiment."
    )
    threshold_profile: str = Field(
        default="Permissive (Demo)",
        description="Human-readable description of the threshold profile applied.",
    )


class ExperimentFullReportResponse(BaseModel):
    """Full experiment report bundling all result data."""

    experiment: dict[str, Any]
    dataset: dict[str, Any]
    patterns: list[dict[str, Any]]
    latent_variables: list[dict[str, Any]]
    summary: ExperimentSummaryResponse


# ---------------------------------------------------------------------------
# Experiment event log
# ---------------------------------------------------------------------------


class ExperimentEventResponse(BaseModel):
    """A single entry in the experiment audit / execution log.

    Mirrors the ``ExperimentEvent`` ORM model.  The ``payload`` field
    contains the human-readable ``message`` and any supplementary metadata
    recorded at the time the event occurred.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    experiment_id: UUID
    phase: str | None = None
    event_type: str
    payload: dict[str, Any] | None = None
    created_at: datetime


class ExperimentEventsListResponse(BaseModel):
    """Chronological list of audit events for an experiment."""

    experiment_id: UUID
    events: list[ExperimentEventResponse]
    total: int
