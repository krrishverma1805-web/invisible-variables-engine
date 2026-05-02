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


class ExperimentConfig(BaseModel):  # type: ignore[misc]
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

    # ── Phase B (B2 / B5) optional overrides ───────────────────────────────
    problem_type: Literal["regression", "binary", "multiclass"] | None = Field(
        default=None,
        description=(
            "User override for the auto-detected problem type. When None, the "
            "pipeline runs `detect_problem_type(y)` and logs the inferred value. "
            "Multiclass is supported for prediction + uplift only — residual-based "
            "detection is regression/binary only (per docs/RESPONSE_CONTRACT.md §8.2)."
        ),
    )
    cv_strategy: Literal["auto", "kfold", "stratified", "timeseries", "group"] | None = Field(
        default=None,
        description=(
            "User override for the CV splitter strategy. None → use the deployment "
            "default (`MLSettings.cv_strategy`, defaults to 'auto')."
        ),
    )
    cv_gap_size: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Purged-CV gap (samples) for TimeSeriesSplit. Set ≥ max_lag when the "
            "dataset has lagged autoregressive features."
        ),
    )

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class ExperimentCreate(BaseModel):  # type: ignore[misc]
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


class ExperimentResponse(BaseModel):  # type: ignore[misc]
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


class ExperimentCreateResponse(BaseModel):  # type: ignore[misc]
    """Minimal 201 Created response body."""

    id: UUID
    status: str
    celery_task_id: str | None = None
    message: str


class ExperimentListResponse(BaseModel):  # type: ignore[misc]
    """Paginated experiment list."""

    experiments: list[ExperimentResponse]
    total: int
    skip: int
    limit: int


class ExperimentProgressResponse(BaseModel):  # type: ignore[misc]
    """Lightweight progress-poll response for WebSocket/polling clients."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    progress_pct: int
    current_stage: str | None = None


# ---------------------------------------------------------------------------
# Error pattern response
# ---------------------------------------------------------------------------


class ErrorPatternResponse(BaseModel):  # type: ignore[misc]
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
    # Phase B4 + plan §96 — CIs and selective-inference bookkeeping.
    effect_size_ci_lower: float | None = None
    effect_size_ci_upper: float | None = None
    effect_size_ci_method: str | None = Field(
        default=None,
        description="'bca' / 'percentile' / 'degenerate'. None when CI unavailable.",
    )
    cross_fit_splits_supporting: int | None = Field(
        default=None,
        description="Splits (out of K) in which this pattern was discovered.",
    )
    selection_corrected: bool = Field(
        default=False,
        description="True when CI was computed via cross-fit (selection-aware).",
    )
    created_at: datetime


class ExperimentSummaryResponse(BaseModel):  # type: ignore[misc]
    """Compact experiment summary — headline, counts, and recommendations.

    ``headline`` and ``summary_text`` surface the LLM-generated prose
    when ``llm_explanation_status='ready'``; otherwise they fall back to
    the rule-based generator. The ``explanation_source`` field tells the
    UI which one is being shown so the AI-assisted badge can be rendered.
    """

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

    # ── LLM enrichment surface (per plan §A1) ──────────────────────────────
    explanation_source: Literal["llm", "rule_based"] = Field(
        default="rule_based",
        description="Which generator produced headline + summary_text.",
    )
    llm_explanation_pending: bool = Field(
        default=False,
        description=(
            "True when llm_explanation_status='pending' — UI should poll. "
            "Always false in flag-off / disabled / failed states."
        ),
    )
    llm_explanation_status: str = Field(
        default="pending",
        description="Lifecycle: pending | ready | failed | disabled.",
    )


class ExperimentFullReportResponse(BaseModel):  # type: ignore[misc]
    """Full experiment report bundling all result data."""

    experiment: dict[str, Any]
    dataset: dict[str, Any]
    patterns: list[dict[str, Any]]
    latent_variables: list[dict[str, Any]]
    summary: ExperimentSummaryResponse


# ---------------------------------------------------------------------------
# Experiment event log
# ---------------------------------------------------------------------------


class ExperimentEventResponse(BaseModel):  # type: ignore[misc]
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


class ExperimentEventsListResponse(BaseModel):  # type: ignore[misc]
    """Chronological list of audit events for an experiment."""

    experiment_id: UUID
    events: list[ExperimentEventResponse]
    total: int
