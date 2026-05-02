"""
SQLAlchemy 2.0 ORM Models — Invisible Variables Engine.

Defines the complete database schema using mapped columns and modern
``Mapped[T]`` type annotations.  Every model inherits from
:class:`ive.db.database.Base`.

Tables
------
=================  ============================================================
Table              Purpose
=================  ============================================================
``datasets``       Uploaded dataset metadata and schema snapshot
``experiments``    Analysis run configuration, status, and lifecycle timestamps
``models``         Per-fold trained model metrics and artefacts
``residuals``      Individual sample residuals (row-level prediction errors)
``error_patterns`` Statistical patterns discovered in the residual space
``latent_variables`` Constructed and validated latent variable definitions
``api_keys``       Hashed API keys with permissions and rate limits
``experiment_events`` Append-only audit/event log for experiment phases
=================  ============================================================
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ive.db.database import Base

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp (used as column defaults)."""
    return datetime.now(UTC)


def _uuid() -> uuid.UUID:
    """Return a new UUID4 (used as the default for all ``id`` primary keys)."""
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# 1. datasets
# ---------------------------------------------------------------------------


class Dataset(Base):
    """Metadata for an uploaded dataset file.

    The raw CSV/Parquet is stored on disk at ``file_path``; this row keeps a
    SHA-256 ``checksum`` so the pipeline can verify file integrity before
    running an experiment.
    """

    __tablename__ = "datasets"
    __table_args__ = (
        Index("idx_datasets_created_at", "created_at"),
        CheckConstraint("row_count >= 0", name="positive_row_count"),
        CheckConstraint("col_count >= 0", name="positive_col_count"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    col_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    target_column: Mapped[str] = mapped_column(String(255), nullable=False)
    time_column: Mapped[str | None] = mapped_column(String(255), nullable=True)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    # -- Relationships --------------------------------------------------------
    experiments: Mapped[list[Experiment]] = relationship(
        "Experiment",
        back_populates="dataset",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Dataset id={self.id} name={self.name!r} rows={self.row_count}>"


# ---------------------------------------------------------------------------
# 2. experiments
# ---------------------------------------------------------------------------


class Experiment(Base):
    """An IVE analysis run bound to a dataset.

    Lifecycle: ``queued → running → completed | failed | cancelled``.
    Progress is tracked by ``progress_pct`` (0–100) and ``current_stage``
    (understand / model / detect / construct).
    """

    __tablename__ = "experiments"
    __table_args__ = (
        Index("idx_experiments_dataset_id", "dataset_id"),
        Index("idx_experiments_status", "status"),
        Index("idx_experiments_created_at", "created_at"),
        CheckConstraint(
            "status IN ('queued','running','completed','failed','cancelled')",
            name="valid_experiment_status",
        ),
        CheckConstraint("progress_pct BETWEEN 0 AND 100", name="valid_progress_pct"),
        CheckConstraint(
            "llm_explanation_status IN ('pending','ready','failed','disabled')",
            name="valid_exp_llm_status",
        ),
        CheckConstraint(
            "problem_type IS NULL OR problem_type IN ('regression','binary','multiclass')",
            name="valid_problem_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    progress_pct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Phase B5: classification support
    problem_type: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # -- LLM enrichment columns (added in migration b2c3d4e5f6a7) ------------
    llm_headline: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_recommendations: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    llm_explanation_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    llm_explanation_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    llm_explanation_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
    )
    llm_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # -- Relationships --------------------------------------------------------
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="experiments")
    trained_models: Mapped[list[TrainedModel]] = relationship(
        "TrainedModel",
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    residuals: Mapped[list[Residual]] = relationship(
        "Residual",
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    error_patterns: Mapped[list[ErrorPattern]] = relationship(
        "ErrorPattern",
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    latent_variables: Mapped[list[LatentVariable]] = relationship(
        "LatentVariable",
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    events: Mapped[list[ExperimentEvent]] = relationship(
        "ExperimentEvent",
        back_populates="experiment",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Experiment id={self.id} status={self.status} stage={self.current_stage}>"


# ---------------------------------------------------------------------------
# 3. models (trained ML models)
# ---------------------------------------------------------------------------


class TrainedModel(Base):
    """A single trained model for one fold of one model type in an experiment."""

    __tablename__ = "models"
    __table_args__ = (
        Index("idx_models_experiment_id", "experiment_id"),
        UniqueConstraint(
            "experiment_id",
            "model_type",
            "fold_number",
            name="uq_models_exp_type_fold",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)
    fold_number: Mapped[int] = mapped_column(Integer, nullable=False)
    train_metric: Mapped[float] = mapped_column(Float, nullable=False)
    val_metric: Mapped[float] = mapped_column(Float, nullable=False)
    metric_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="rmse",
    )
    artifact_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    hyperparams: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # Phase B1 — Optuna HPO outcomes (NULL when HPO disabled or skipped).
    hpo_search_results: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    hpo_best_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    feature_importances: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    training_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # -- Relationships --------------------------------------------------------
    experiment: Mapped[Experiment] = relationship(
        "Experiment",
        back_populates="trained_models",
    )

    def __repr__(self) -> str:
        return (
            f"<TrainedModel id={self.id} type={self.model_type} "
            f"fold={self.fold_number} val={self.val_metric:.4f}>"
        )


# ---------------------------------------------------------------------------
# 4. residuals
# ---------------------------------------------------------------------------


class Residual(Base):
    """Per-sample, per-fold out-of-fold residual for an experiment.

    This table can grow large (rows × folds × model_types).  For datasets
    above ~100 K rows the pipeline stores the data as NumPy artefacts and
    writes summary rows here instead.
    """

    __tablename__ = "residuals"
    __table_args__ = (
        Index("idx_residuals_experiment_model", "experiment_id", "model_type"),
        Index("idx_residuals_experiment_fold", "experiment_id", "fold_number"),
        CheckConstraint(
            "residual_kind IN ('raw','deviance')",
            name="valid_residual_kind",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_type: Mapped[str] = mapped_column(String(20), nullable=False)
    sample_index: Mapped[int] = mapped_column(Integer, nullable=False)
    fold_number: Mapped[int] = mapped_column(Integer, nullable=False)
    actual_value: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    residual_value: Mapped[float] = mapped_column(Float, nullable=False)
    abs_residual: Mapped[float] = mapped_column(Float, nullable=False)
    # Phase B5: 'raw' for regression OOF residuals, 'deviance' for classification.
    residual_kind: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="raw",
    )
    feature_vector: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # -- Relationships --------------------------------------------------------
    experiment: Mapped[Experiment] = relationship(
        "Experiment",
        back_populates="residuals",
    )

    def __repr__(self) -> str:
        return (
            f"<Residual exp={self.experiment_id} model={self.model_type} "
            f"idx={self.sample_index} r={self.residual_value:.3f}>"
        )


# ---------------------------------------------------------------------------
# 5. error_patterns
# ---------------------------------------------------------------------------


class ErrorPattern(Base):
    """A statistically significant pattern discovered in the residuals.

    Pattern types: ``subgroup``, ``cluster``, ``interaction``, ``temporal``.
    Each row captures the statistical evidence (effect size, p-value,
    Bonferroni-adjusted p-value) and the subgroup definition as JSONB.
    """

    __tablename__ = "error_patterns"
    __table_args__ = (
        Index("idx_patterns_experiment_id", "experiment_id"),
        Index("idx_patterns_type", "pattern_type"),
        CheckConstraint(
            "pattern_type IN ('subgroup','cluster','interaction','temporal','variance_regime')",
            name="valid_pattern_type",
        ),
        CheckConstraint(
            "effect_size_ci_method IS NULL OR "
            "effect_size_ci_method IN ('bca','percentile','degenerate')",
            name="valid_effect_size_ci_method",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    pattern_type: Mapped[str] = mapped_column(String(20), nullable=False)
    subgroup_definition: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    effect_size: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_p_value: Mapped[float] = mapped_column(Float, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    stability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_residual: Mapped[float] = mapped_column(Float, nullable=False)
    std_residual: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # Phase B4 — bootstrap-bourne CIs on the effect size.
    effect_size_ci_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    effect_size_ci_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    effect_size_ci_method: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Plan §96 + §172 — cross-fit selective-inference bookkeeping.
    cross_fit_splits_supporting: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    selection_corrected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # -- Relationships --------------------------------------------------------
    experiment: Mapped[Experiment] = relationship(
        "Experiment",
        back_populates="error_patterns",
    )

    def __repr__(self) -> str:
        return (
            f"<ErrorPattern id={self.id} type={self.pattern_type} "
            f"d={self.effect_size:.2f} p={self.p_value:.4f}>"
        )


# ---------------------------------------------------------------------------
# 6. latent_variables
# ---------------------------------------------------------------------------


class LatentVariable(Base):
    """A constructed and (optionally validated) latent variable.

    Each row references the ``error_patterns`` that informed its construction
    via ``source_pattern_ids`` (UUID array).  The ``construction_rule`` JSONB
    stores enough information to recompute the variable on new data.

    Lifecycle: ``candidate → validated | rejected``.
    """

    __tablename__ = "latent_variables"
    __table_args__ = (
        Index("idx_lv_experiment_id", "experiment_id"),
        Index("idx_lv_status", "status"),
        Index(
            "idx_lv_llm_status",
            "llm_explanation_status",
            postgresql_where="llm_explanation_status != 'ready'",
        ),
        CheckConstraint(
            "status IN ('candidate','validated','rejected')",
            name="valid_lv_status",
        ),
        CheckConstraint(
            "bootstrap_presence_rate BETWEEN 0.0 AND 1.0",
            name="valid_bootstrap_rate",
        ),
        CheckConstraint(
            "llm_explanation_status IN ('pending','ready','failed','disabled')",
            name="valid_lv_llm_status",
        ),
        CheckConstraint(
            "apply_compatibility IN ('ok','requires_review','incompatible')",
            name="valid_apply_compatibility",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    construction_rule: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source_pattern_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PG_UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    importance_score: Mapped[float] = mapped_column(Float, nullable=False)
    stability_score: Mapped[float] = mapped_column(Float, nullable=False)
    bootstrap_presence_rate: Mapped[float] = mapped_column(Float, nullable=False)
    model_improvement_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_interval_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_interval_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="candidate",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # -- LLM enrichment columns (added in migration a1b2c3d4e5f6) ------------
    llm_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_explanation_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    llm_explanation_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    llm_explanation_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="pending",
    )

    # -- Apply-compatibility flag (added in migration a3b4c5d6e7f8) ---------
    # 'ok' (default), 'requires_review' (a column the LV references changed),
    # 'incompatible' (a column the LV references was dropped or retyped).
    apply_compatibility: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="ok",
        server_default="ok",
    )

    # -- Selective-inference bookkeeping (added in migration b4c5d6e7f8a9) --
    cross_fit_splits_supporting: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    selection_corrected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )

    # -- Relationships --------------------------------------------------------
    experiment: Mapped[Experiment] = relationship(
        "Experiment",
        back_populates="latent_variables",
    )

    def __repr__(self) -> str:
        return f"<LatentVariable id={self.id} name={self.name!r} status={self.status}>"


# ---------------------------------------------------------------------------
# 7. api_keys
# ---------------------------------------------------------------------------


class APIKey(Base):
    """Hashed API key used for request authentication.

    The raw key is **never stored**.  On creation, ``key_hash`` is computed
    via ``hashlib.sha256`` and only the hash is persisted.  Authentication
    hashes the incoming header value and compares against ``key_hash``.
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        Index("idx_api_keys_active", "is_active"),
        CheckConstraint(
            "scopes <@ ARRAY['read','write','admin']::varchar[]",
            name="ck_api_keys_scopes_valid",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    permissions: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {"read": True, "write": True, "admin": False},
    )
    # Structured scopes (added in migration e5f6a7b8c9d0). Replaces the
    # free-form ``permissions`` JSONB above for new code; ``permissions``
    # is retained for backwards compatibility.
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String(32)),
        nullable=False,
        default=lambda: ["read", "write"],
    )
    rate_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    created_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_rotated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<APIKey id={self.id} name={self.name!r} active={self.is_active} "
            f"scopes={self.scopes}>"
        )

    def has_scope(self, scope: str) -> bool:
        """Return True when this key has ``scope`` (admin implies all)."""
        return scope in self.scopes or "admin" in self.scopes

    def is_expired(self, now: datetime | None = None) -> bool:
        """Return True when ``expires_at`` is set and has passed."""
        if self.expires_at is None:
            return False
        return (now or _utcnow()) >= self.expires_at


# ---------------------------------------------------------------------------
# 8. experiment_events (audit log)
# ---------------------------------------------------------------------------


class ExperimentEvent(Base):
    """Append-only audit log for experiment lifecycle events.

    Each event records a ``phase`` (understand / model / detect / construct),
    an ``event_type`` (e.g. ``phase_started``, ``phase_completed``,
    ``error_occurred``) and an optional JSONB ``payload`` with details.
    """

    __tablename__ = "experiment_events"
    __table_args__ = (
        Index("idx_events_experiment_id", "experiment_id"),
        Index("idx_events_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    phase: Mapped[str | None] = mapped_column(String(50), nullable=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    # -- Relationships --------------------------------------------------------
    experiment: Mapped[Experiment] = relationship(
        "Experiment",
        back_populates="events",
    )

    def __repr__(self) -> str:
        return f"<ExperimentEvent type={self.event_type} phase={self.phase}>"


# ---------------------------------------------------------------------------
# 9. explanation_feedback (per plan §78 / §158 / §192)
# ---------------------------------------------------------------------------


class ExplanationFeedback(Base):
    """Thumbs-up/down feedback on generated explanations.

    Carries ``prompt_version`` and ``model_version`` so feedback can be
    sliced by what was actually generated. Used to compare rule-based vs
    LLM-enriched outputs in production (per §92 axis 2).
    """

    __tablename__ = "explanation_feedback"
    __table_args__ = (
        Index("idx_feedback_entity", "entity_type", "entity_id"),
        Index("idx_feedback_created_at", "created_at"),
        CheckConstraint(
            "entity_type IN ('experiment','latent_variable','pattern')",
            name="ck_explanation_feedback_entity_type",
        ),
        CheckConstraint(
            "explanation_source IN ('llm','rule_based')",
            name="ck_explanation_feedback_source",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    explanation_source: Mapped[str] = mapped_column(String(16), nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String(16), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    helpful: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    api_key_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<ExplanationFeedback id={self.id} entity={self.entity_type}:{self.entity_id} "
            f"helpful={self.helpful} source={self.explanation_source}>"
        )


# ---------------------------------------------------------------------------
# 10. dataset_column_metadata (per plan §142 / §174 / §203)
# ---------------------------------------------------------------------------


class DatasetColumnMetadata(Base):
    """Per-column sensitivity metadata.

    Default ``non_public`` (safe by default). Only columns marked ``public``
    may appear in LLM payloads. LVs whose segments reference any non-public
    column have ``llm_explanation_status='disabled'`` with reason
    ``pii_protection_per_column``.
    """

    __tablename__ = "dataset_column_metadata"
    __table_args__ = (
        Index("idx_dataset_column_metadata_dataset", "dataset_id"),
        UniqueConstraint(
            "dataset_id",
            "column_name",
            name="uq_dataset_column_metadata_dataset_column",
        ),
        CheckConstraint(
            "sensitivity IN ('public','non_public')",
            name="ck_dataset_column_metadata_sensitivity",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    sensitivity: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="non_public",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<DatasetColumnMetadata dataset={self.dataset_id} "
            f"column={self.column_name!r} sensitivity={self.sensitivity}>"
        )


# ---------------------------------------------------------------------------
# 11. auth_audit_log (per plan §113 / §155)
# ---------------------------------------------------------------------------


class AuthAuditLog(Base):
    """Append-only audit log of authentication events.

    One row per authenticated request (success or failure). 30-day retention
    is the recommended default; ops may rotate via a beat task.
    """

    __tablename__ = "auth_audit_log"
    __table_args__ = (
        Index("idx_auth_audit_log_api_key_created", "api_key_id", "created_at"),
        Index(
            "idx_auth_audit_log_failures",
            "created_at",
            postgresql_where="event_type != 'auth_success'",
        ),
        Index("idx_auth_audit_log_created_at", "created_at"),
        CheckConstraint(
            "event_type IN ('auth_success','auth_failure','auth_missing','auth_expired')",
            name="ck_auth_audit_log_event_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
    )
    api_key_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    method: Mapped[str] = mapped_column(String(8), nullable=False)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<AuthAuditLog event={self.event_type} key={self.api_key_name!r} "
            f"path={self.path!r} status={self.status_code}>"
        )


# ---------------------------------------------------------------------------
# 13. latent_variable_annotations  (Phase C2.1)
# ---------------------------------------------------------------------------


class LatentVariableAnnotation(Base):
    """Free-text annotations on latent variables.

    Power users + reviewers leave context, hypotheses, or follow-up
    notes attached to a specific LV. Each annotation carries the API
    key that authored it so the PR-2 audit log ties cleanly to LV
    history. Cascading delete with the LV.
    """

    __tablename__ = "latent_variable_annotations"
    __table_args__ = (
        Index(
            "idx_lv_annotations_lv",
            "latent_variable_id",
            "created_at",
        ),
        CheckConstraint(
            "char_length(body) BETWEEN 1 AND 10000",
            name="ck_lv_annotations_body_length",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    latent_variable_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("latent_variables.id", ondelete="CASCADE"),
        nullable=False,
    )
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
    )
    api_key_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<LatentVariableAnnotation lv={self.latent_variable_id} "
            f"author={self.api_key_name!r} len={len(self.body)}>"
        )


# ---------------------------------------------------------------------------
# 14. share_tokens + share_access_log  (Phase C2.2)
# ---------------------------------------------------------------------------


class ShareToken(Base):
    """A read-only share token for an experiment report.

    Per plan §C2.2, the raw token is **never** stored — only its
    sha256 hash. The optional ``passphrase_hash`` (bcrypt) gates an
    extra interactive challenge before the report is rendered.
    Tokens have a default 7-day expiry; revocation is soft (sets
    ``revoked_at``) so the audit log can reference the token row.
    """

    __tablename__ = "share_tokens"
    __table_args__ = (
        Index("idx_share_tokens_experiment", "experiment_id"),
        Index("idx_share_tokens_active", "revoked_at", "expires_at"),
        UniqueConstraint("token_hash", name="uq_share_tokens_token_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
    )
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    passphrase_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_by_api_key_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_by_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<ShareToken experiment={self.experiment_id} "
            f"expires={self.expires_at} revoked={self.revoked_at}>"
        )


class ShareAccessLog(Base):
    """One row per successful share-token access. Append-only audit log."""

    __tablename__ = "share_access_log"
    __table_args__ = (
        Index("idx_share_access_token_time", "share_token_id", "accessed_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    share_token_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("share_tokens.id", ondelete="CASCADE"),
        nullable=False,
    )
    accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ShareAccessLog token={self.share_token_id} "
            f"at={self.accessed_at} ip={self.client_ip}>"
        )


# ---------------------------------------------------------------------------
# 13. dataset_column_versions (per plan §157 + §197 / RC §19)
# ---------------------------------------------------------------------------


class DatasetColumnVersion(Base):
    """Per-column lineage snapshot.

    One row per (dataset_id, column_name, version). ``value_hash`` is
    sha256 of the canonical-bytes representation of the column at upload
    time. The lineage detector compares consecutive versions to classify
    each column as ``ok / retype / value_change / rename_candidate /
    drop / add`` — see :mod:`ive.data.lineage`.
    """

    __tablename__ = "dataset_column_versions"
    __table_args__ = (
        Index(
            "idx_dataset_column_versions_dataset",
            "dataset_id",
            "column_name",
        ),
        UniqueConstraint(
            "dataset_id",
            "column_name",
            "version",
            name="uq_dataset_column_versions_id_col_ver",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
    )
    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dtype: Mapped[str] = mapped_column(String(64), nullable=False)
    value_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<DatasetColumnVersion dataset={self.dataset_id} "
            f"column={self.column_name!r} v{self.version}>"
        )
