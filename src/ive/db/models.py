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
from typing import Any
from datetime import UTC, datetime

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
            "pattern_type IN ('subgroup','cluster','interaction','temporal')",
            name="valid_pattern_type",
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
        CheckConstraint(
            "status IN ('candidate','validated','rejected')",
            name="valid_lv_status",
        ),
        CheckConstraint(
            "bootstrap_presence_rate BETWEEN 0.0 AND 1.0",
            name="valid_bootstrap_rate",
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
    __table_args__ = (Index("idx_api_keys_active", "is_active"),)

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=_uuid,
    )
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    permissions: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {"read": True, "write": True, "admin": False},
    )
    rate_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<APIKey id={self.id} name={self.name!r} active={self.is_active}>"


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
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
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
