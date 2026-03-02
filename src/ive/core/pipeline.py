"""
Pipeline Data Structures.

Defines the shared context (PipelineContext) and result types that
flow through the four-phase IVE pipeline. Using a context object
avoids tight coupling between phases — each phase reads what it needs
from context and writes its outputs back.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from ive.api.v1.schemas.experiment_schemas import ExperimentConfig


@dataclass
class LatentVariableCandidate:
    """
    A candidate latent variable discovered by Phase 3 and enriched by Phase 4.

    Attributes:
        rank: Ranking by confidence (1 = highest)
        name: Auto-generated or user-provided name
        description: Prose description of what the variable represents
        explanation: Human-readable NL explanation
        confidence_score: Weighted composite score (0–1)
        effect_size: Cohen's d between high/low error clusters
        coverage_pct: Fraction of dataset covered by this pattern
        candidate_features: Existing features correlated with this LV
        validation: Bootstrap + permutation test results
    """

    rank: int = 0
    name: str | None = None
    description: str | None = None
    explanation: str | None = None
    confidence_score: float = 0.0
    effect_size: float = 0.0
    coverage_pct: float = 0.0
    candidate_features: list[str] = field(default_factory=list)
    validation: dict[str, float] = field(default_factory=dict)
    cluster_labels: np.ndarray | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Generic container for a single phase's outputs."""

    phase_name: str
    success: bool
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error_msg: str | None = None


@dataclass
class PipelineContext:
    """
    Shared mutable context passed through all pipeline phases.

    Phase 1 (Understand) populates: df, column_types, profile
    Phase 2 (Model)      populates: residuals, model_artifacts
    Phase 3 (Detect)     populates: patterns, cluster_labels, shap_values
    Phase 4 (Construct)  populates: latent_variables

    Design rationale: A single mutable context avoids returning complex
    multi-value tuples from each phase and keeps the engine orchestration
    simple.
    """

    experiment_id: uuid.UUID
    config: ExperimentConfig
    data_path: str

    # Populated by Phase 1 — Understand
    df: pd.DataFrame | None = None
    column_types: dict[str, str] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    target_series: pd.Series | None = None
    feature_columns: list[str] = field(default_factory=list)

    # Populated by Phase 2 — Model
    residuals: np.ndarray | None = None
    predictions: np.ndarray | None = None
    model_artifacts: dict[str, Any] = field(default_factory=dict)
    feature_matrix: np.ndarray | None = None

    # Populated by Phase 3 — Detect
    patterns: list[dict[str, Any]] = field(default_factory=list)
    cluster_labels: np.ndarray | None = None
    shap_values: np.ndarray | None = None
    shap_interaction_values: np.ndarray | None = None

    # Populated by Phase 4 — Construct
    latent_variables: list[LatentVariableCandidate] = field(default_factory=list)

    # Phase result metadata
    phase_results: dict[str, PhaseResult] = field(default_factory=dict)


@dataclass
class EngineResult:
    """Final output of a completed IVE engine run."""

    experiment_id: uuid.UUID
    latent_variables: list[LatentVariableCandidate]
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)
