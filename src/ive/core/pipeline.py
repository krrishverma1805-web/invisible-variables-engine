"""
Pipeline Orchestrator — Invisible Variables Engine.

``IVEPipeline`` bridges all four ML phases into a single async run:

1. **Understand** — load data, apply schema, split X/y.
2. **Model**     — run K-fold cross-validation with Linear + XGBoost,
                   collect OOF residuals, persist per-fold metrics.
3. **Detect**    — subgroup KS-scan + HDBSCAN clustering on worst errors,
                   persist patterns.
4. **Construct** — synthesise latent-variable candidates, bootstrap-validate,
                   persist validated variables.

The pipeline is async so it integrates naturally with SQLAlchemy's
async session.  Callers running in a synchronous Celery worker bridge
using ``asyncio.run(_run_pipeline_async(...))``.

This module also preserves the legacy data-structure types
(``LatentVariableCandidate``, ``PipelineContext``, ``PhaseResult``,
``EngineResult``) that are imported by other modules.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

from sklearn.model_selection import train_test_split

from ive.construction.bootstrap_validator import BootstrapValidator
from ive.construction.explanation_generator import ExplanationGenerator
from ive.construction.variable_synthesizer import VariableSynthesizer, apply_construction_rule
from ive.data.preprocessor import DataPreprocessor
from ive.data.validator import DataValidator
from ive.db.models import Dataset, Experiment, LatentVariable
from ive.db.repositories.dataset_repo import DatasetRepository
from ive.db.repositories.experiment_repo import ExperimentRepository
from ive.db.repositories.latent_variable_repo import LatentVariableRepository
from ive.detection.clustering import HDBSCANClustering
from ive.detection.pattern_scorer import PatternScorer
from ive.detection.subgroup_discovery import SubgroupDiscovery
from ive.detection.temporal_analysis import TemporalAnalyzer
from ive.models.cross_validator import CrossValidator
from ive.models.linear_model import LinearIVEModel
from ive.models.xgboost_model import XGBoostIVEModel
from ive.utils.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from ive.api.v1.schemas.experiment_schemas import ExperimentConfig

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Experiment event helper — independent of the ORM session
# ---------------------------------------------------------------------------
# Each event is written in its own psycopg2 transaction so that:
#  a) events are immediately visible (not deferred to the end of the
#     SQLAlchemy session commit);
#  b) failure of the main pipeline does NOT roll back the events —
#     the audit log survives even for failed experiments.
# ---------------------------------------------------------------------------


def _record_event(
    experiment_id: uuid.UUID,
    event_type: str,
    message: str,
    phase: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write a single lifecycle event to ``experiment_events`` via psycopg2.

    Uses a dedicated connection so the write is committed immediately,
    independent of any active SQLAlchemy session.  Failures are logged but
    never re-raised — event persistence must never abort the pipeline.

    Args:
        experiment_id: UUID of the parent experiment.
        event_type:    Machine-readable event identifier
                       (e.g. ``"dataset_loaded"``).
        message:       Human-readable description of the event.
        phase:         Optional pipeline phase label
                       (``understand`` / ``model`` / ``detect`` / ``construct``).
        metadata:      Optional supplementary data stored as JSONB payload.
    """
    import psycopg2

    from ive.config import get_settings

    payload: dict[str, Any] = {"message": message, **(metadata or {})}
    event_id = uuid.uuid4()

    try:
        settings = get_settings()
        dsn = settings.sync_database_url.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(dsn)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO experiment_events
                    (id, experiment_id, phase, event_type, payload, created_at)
                VALUES (%s::uuid, %s::uuid, %s, %s, %s, NOW())
                """,
                (
                    str(event_id),
                    str(experiment_id),
                    phase,
                    event_type,
                    json.dumps(payload),
                ),
            )
            conn.commit()
            log.debug(
                "pipeline.event_recorded",
                experiment_id=str(experiment_id),
                event_type=event_type,
                phase=phase,
            )
        finally:
            conn.close()
    except Exception as exc:
        # Never let event-writing abort the pipeline.
        log.warning(
            "pipeline.event_record_failed",
            experiment_id=str(experiment_id),
            event_type=event_type,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Legacy data-structure types (preserved for backward compatibility)
# ---------------------------------------------------------------------------


@dataclass
class LatentVariableCandidate:
    """A candidate latent variable discovered by Phase 3 and enriched by Phase 4."""

    rank: int = 0
    name: str | None = None
    description: str | None = None
    explanation: str | None = None
    confidence_score: float = 0.0
    effect_size: float = 0.0
    coverage_pct: float = 0.0
    candidate_features: list[str] = field(default_factory=list)
    validation: dict[str, float] = field(default_factory=dict)
    cluster_labels: np.ndarray[Any, Any] | None = None
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
    """Shared mutable context passed through all pipeline phases."""

    experiment_id: uuid.UUID
    config: ExperimentConfig
    data_path: str

    df: pd.DataFrame | None = None
    column_types: dict[str, str] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    target_series: pd.Series | None = None
    feature_columns: list[str] = field(default_factory=list)

    residuals: np.ndarray[Any, Any] | None = None
    predictions: np.ndarray[Any, Any] | None = None
    model_artifacts: dict[str, Any] = field(default_factory=dict)
    feature_matrix: np.ndarray[Any, Any] | None = None

    patterns: list[dict[str, Any]] = field(default_factory=list)
    cluster_labels: np.ndarray[Any, Any] | None = None
    shap_values: np.ndarray[Any, Any] | None = None
    shap_interaction_values: np.ndarray[Any, Any] | None = None

    latent_variables: list[LatentVariableCandidate] = field(default_factory=list)
    phase_results: dict[str, PhaseResult] = field(default_factory=dict)


@dataclass
class EngineResult:
    """Final output of a completed IVE engine run."""

    experiment_id: uuid.UUID
    latent_variables: list[LatentVariableCandidate]
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Columns of this dtype are dropped from features before modelling.
_NON_FEATURE_TYPES = frozenset({"text", "id", "datetime"})

# Map from config model_type string to IVEModel class + whether to use
# stratified cross-validation (classification).
_MODEL_MAP = {
    "linear": (LinearIVEModel, False),
    "xgboost": (XGBoostIVEModel, False),
}


def _parse_model_types(config: dict[str, Any]) -> list[str]:
    """Return the list of model type strings from the experiment config.

    Falls back to ``["linear", "xgboost"]`` when the key is absent or empty.

    Args:
        config: Raw ``experiment.config_json`` dict.

    Returns:
        List of model type strings.
    """
    types = config.get("model_types", ["linear", "xgboost"])
    return types if types else ["linear", "xgboost"]


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON/JSONB serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _drop_non_feature_columns(
    X: pd.DataFrame,
    schema: dict[str, Any],
    time_column: str | None,
) -> pd.DataFrame:
    """Drop schema-defined non-feature columns and the time column from X.

    Args:
        X:           Feature DataFrame (target already removed).
        schema:      ``dataset.schema_json`` dict containing a ``'columns'`` list.
        time_column: Optional name of the temporal column to exclude.

    Returns:
        Filtered DataFrame with non-feature and time columns removed.
    """
    drop_cols = [
        c["name"]
        for c in schema.get("columns", [])
        if c.get("type") in _NON_FEATURE_TYPES and c["name"] in X.columns
    ]
    if time_column and time_column in X.columns:
        drop_cols.append(time_column)

    if drop_cols:
        log.debug("pipeline.drop_columns", dropped=drop_cols)
        X = X.drop(columns=drop_cols, errors="ignore")

    return X


def _build_residual_rows(
    model_type: str,
    fold_assignments: np.ndarray[Any, Any],
    y_values: np.ndarray[Any, Any],
    oof_predictions: np.ndarray[Any, Any],
    oof_residuals: np.ndarray[Any, Any],
) -> list[dict[str, Any]]:
    """Build a list of residual-row dicts suitable for ``add_residuals_batch``.

    Caps the batch at 10 000 rows to avoid excessive memory + DB writes for
    very large datasets; a uniform subsample is taken if needed.

    Args:
        model_type:      String model identifier.
        fold_assignments: Per-sample fold index array.
        y_values:        Actual target values.
        oof_predictions: Out-of-fold predicted values.
        oof_residuals:   Pre-computed ``y - oof_predictions``.

    Returns:
        List of row dicts compatible with the ``Residual`` ORM model.
    """
    n = len(y_values)
    indices = np.arange(n)

    # Subsample if dataset is very large
    max_rows = 10_000
    if n > max_rows:
        rng = np.random.default_rng(42)
        indices = rng.choice(n, max_rows, replace=False)
        indices.sort()

    rows: list[dict[str, Any]] = []
    for i in indices:
        rows.append(
            {
                "model_type": model_type,
                "sample_index": int(i),
                "fold_number": int(fold_assignments[i]),
                "actual_value": float(y_values[i]),
                "predicted_value": float(oof_predictions[i]),
                "residual_value": float(oof_residuals[i]),
                "abs_residual": float(abs(oof_residuals[i])),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


class IVEPipeline:
    """Full 4-phase IVE pipeline orchestrator.

    Accepts an async SQLAlchemy session and an artifact store, then runs all
    pipeline phases in sequence.  Progress and status are persisted to the
    database after each major step so that the WebSocket polling endpoint
    reflects live state.

    Args:
        db_session:     Async SQLAlchemy ``AsyncSession``.
        artifact_store: File-store client for loading raw dataset bytes.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        artifact_store: Any,
    ) -> None:
        self.session = db_session
        self.store = artifact_store
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_experiment(self, experiment_id: uuid.UUID) -> dict[str, Any]:
        """Run all four IVE phases for a single experiment.

        Fetches the experiment and dataset from the database, loads data from
        the artifact store, executes each pipeline phase, and persists all
        intermediate and final results.

        Args:
            experiment_id: UUID of the experiment to run.

        Returns:
            Summary dict with keys:

            * ``status``      — always ``"completed"``
            * ``n_patterns``  — total patterns discovered in Phase 3
            * ``n_validated`` — latent variables accepted in Phase 4

        Raises:
            ValueError: If the experiment does not exist in the database.
            Exception:  Any unhandled error is logged, the experiment is
                        marked ``"failed"`` in the DB, and then re-raised.
        """
        t_start = time.perf_counter()
        exp_repo = ExperimentRepository(self.session, Experiment)

        # Outer variable declared here so the except block can reference it.
        validated: list[dict[str, Any]] = []

        try:
            # Bind experiment_id to logger context for all subsequent log lines
            self.logger = self.logger.bind(experiment_id=str(experiment_id))

            # ── Fetch experiment ───────────────────────────────────────
            experiment = await exp_repo.get_by_id(experiment_id)
            if experiment is None:
                raise ValueError(f"Experiment {experiment_id} not found in DB.")

            await exp_repo.mark_started(experiment_id)
            await exp_repo.update_progress(experiment_id, 5, "understand")
            _record_event(
                experiment_id,
                event_type="experiment_started",
                message="Experiment accepted by the worker and pipeline execution has begun.",
                phase="understand",
            )

            self.logger.info(
                "pipeline.start",
                experiment_id=str(experiment_id),
                dataset_id=str(experiment.dataset_id),
            )

            # ── Phase 1: Understand — load data ───────────────────────
            ds_repo = DatasetRepository(self.session, Dataset)
            dataset = await ds_repo.get_by_id(experiment.dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset {experiment.dataset_id} not found.")

            file_bytes = await self.store.load_file(dataset.file_path)
            df = pd.read_csv(io.BytesIO(file_bytes))

            target_col: str = dataset.target_column
            if target_col not in df.columns:
                raise ValueError(
                    f"Target column '{target_col}' not found in dataset. "
                    f"Available: {df.columns.tolist()}"
                )

            X = df.drop(columns=[target_col])
            y: pd.Series = df[target_col]

            # ── Validate data before proceeding ──────────────────────
            validator = DataValidator()
            validation_result = validator.validate(df, target_col)
            for warn_msg in validation_result.warnings:
                self.logger.warning("pipeline.validation_warning", msg=warn_msg)

            schema: dict[str, Any] = dataset.schema_json or {}
            X = _drop_non_feature_columns(X, schema, getattr(dataset, "time_column", None))

            # Drop zero-variance columns flagged by validator
            if validation_result.dropped_columns:
                cols_to_drop = [c for c in validation_result.dropped_columns if c in X.columns]
                if cols_to_drop:
                    X = X.drop(columns=cols_to_drop)
                    self.logger.info("pipeline.dropped_columns", columns=cols_to_drop)

            # Keep raw X for detection (construction rules must reference
            # original column names so they work in holdout, bootstrap, and
            # the /apply API). Preprocessing is only for model training.
            X_raw = X.copy()

            y_values = y.to_numpy(dtype=np.float64)

            # ── Holdout split (BEFORE preprocessing to avoid leakage) ──
            config: dict[str, Any] = getattr(experiment, "config_json", None) or {}
            holdout_ratio = float(config.get("test_size", 0.2))
            min_holdout_rows = 20
            skip_holdout = (
                holdout_ratio <= 0.0
                or len(X_raw) < min_holdout_rows * 2
            )

            if skip_holdout:
                X_train_raw, X_holdout_raw = X_raw, X_raw.iloc[0:0]
                y_train, y_holdout = y_values, y_values[0:0]
                if holdout_ratio > 0.0:
                    self.logger.warning(
                        "pipeline.holdout_skipped",
                        reason="dataset_too_small",
                        n_rows=len(X_raw),
                        min_required=min_holdout_rows * 2,
                    )
            else:
                time_column_name = getattr(dataset, "time_column", None)

                if time_column_name and time_column_name in df.columns:
                    time_vals = df[time_column_name]
                    sorted_indices = time_vals.argsort()
                    split_point = int(len(sorted_indices) * (1 - holdout_ratio))
                    train_indices = sorted_indices[:split_point]
                    holdout_indices = sorted_indices[split_point:]
                    X_train_raw = X_raw.iloc[train_indices]
                    X_holdout_raw = X_raw.iloc[holdout_indices]
                    y_train = y_values[train_indices]
                    y_holdout = y_values[holdout_indices]
                else:
                    X_train_raw, X_holdout_raw, y_train, y_holdout = train_test_split(
                        X_raw, y_values, test_size=holdout_ratio, random_state=42
                    )

                self.logger.info(
                    "pipeline.holdout_split",
                    holdout_ratio=holdout_ratio,
                    train_size=len(X_train_raw),
                    holdout_size=len(X_holdout_raw),
                    temporal_split=bool(time_column_name and time_column_name in df.columns),
                )

            # X_train = raw features for detection/construction (original column names)
            X_train = X_train_raw
            X_holdout = X_holdout_raw

            # ── Preprocess for model training only ───────────────────
            col_types: dict[str, str] | None = None
            if schema.get("columns"):
                col_types = {
                    c["name"]: c.get("detected_type", "numeric")
                    for c in schema["columns"]
                    if c["name"] in X_train_raw.columns
                }

            feature_columns = list(X_train_raw.columns)
            preprocessor = DataPreprocessor()
            try:
                X_processed, _feature_names_out = preprocessor.fit_transform(
                    X_train_raw, feature_columns, column_types=col_types
                )
                X_holdout_processed = preprocessor.transform(
                    X_holdout_raw, feature_columns
                ) if len(X_holdout_raw) > 0 else np.empty((0, X_processed.shape[1]))
            except ValueError as prep_err:
                self.logger.warning("pipeline.preprocessor_fallback", error=str(prep_err))
                X_processed = X_train_raw.select_dtypes(include=[np.number]).fillna(
                    X_train_raw.median(numeric_only=True)
                ).to_numpy(dtype=np.float64)
                X_holdout_processed = X_holdout_raw.select_dtypes(include=[np.number]).fillna(
                    X_train_raw.median(numeric_only=True)
                ).to_numpy(dtype=np.float64) if len(X_holdout_raw) > 0 else np.empty((0, X_processed.shape[1]))

            self.logger.info(
                "pipeline.data_loaded",
                n_rows=len(df),
                n_features=len(X.columns),
            )
            _record_event(
                experiment_id,
                event_type="dataset_loaded",
                message=(
                    f"Dataset loaded successfully: {len(df):,} rows, "
                    f"{len(X.columns)} feature columns, target: '{target_col}'."
                ),
                phase="understand",
                metadata={
                    "n_rows": len(df),
                    "n_features": len(X.columns),
                    "target_column": target_col,
                },
            )
            await exp_repo.update_progress(experiment_id, 20, "model")

            # ── Phase 2: Model — cross-validation ─────────────────────
            model_types = _parse_model_types(config)
            cv_folds: int = int(config.get("cv_folds", 5))

            # Use PREPROCESSED training set for model fitting
            # X_train remains raw (original columns) for detection/construction
            X_values = X_processed  # numpy array from preprocessor

            all_residuals: list[dict[str, Any]] = []

            cv_results_by_model: dict[str, Any] = {}  # Fix 3: track per-model cv_result

            _record_event(
                experiment_id,
                event_type="modeling_started",
                message=(
                    f"Cross-validation started with models: {', '.join(model_types)}, "
                    f"{cv_folds} folds."
                ),
                phase="model",
                metadata={"model_types": model_types, "cv_folds": cv_folds},
            )

            for model_type in model_types:
                model_cls, stratified = _MODEL_MAP.get(model_type, (LinearIVEModel, False))
                model_instance = model_cls()

                cv = CrossValidator(model_instance, n_splits=cv_folds, stratified=stratified)
                cv_result = cv.fit(X_values, y_train)
                cv_results_by_model[model_type] = cv_result

                oof_preds = cv_result.oof_predictions
                oof_resid = cv_result.oof_residuals  # y - oof_preds

                all_residuals.append(
                    {
                        "model_type": model_type,
                        "residuals": oof_resid,
                        "abs_residuals": np.abs(oof_resid),
                        "oof_predictions": oof_preds,
                    }
                )

                # Persist per-fold metrics
                for fold_idx, fold_score in enumerate(cv_result.fold_scores):
                    val_rmse = (
                        float(
                            np.sqrt(np.mean(oof_resid[cv_result.fold_assignments == fold_idx] ** 2))
                        )
                        if np.any(cv_result.fold_assignments == fold_idx)
                        else 0.0
                    )

                    await exp_repo.add_trained_model(
                        experiment_id=experiment_id,
                        model_type=model_type,
                        fold_number=fold_idx,
                        train_metric=float(fold_score),
                        val_metric=val_rmse,
                        metric_name="r2_and_rmse",
                        hyperparams={},
                    )

                # Bulk-insert residuals for this model
                residual_rows = _build_residual_rows(
                    model_type,
                    cv_result.fold_assignments,
                    y_train,
                    oof_preds,
                    oof_resid,
                )
                await exp_repo.add_residuals_batch(experiment_id, residual_rows)

                self.logger.info(
                    "pipeline.model_complete",
                    model_type=model_type,
                    mean_r2=round(cv_result.mean_score, 4),
                    residual_std=round(float(np.std(oof_resid)), 4),
                )
                self.logger.info(
                    "pipeline.model_residual_summary",
                    model_type=model_type,
                    mean_score=round(cv_result.mean_score, 4),
                    std_score=round(cv_result.std_score, 4),
                    residual_std=round(float(np.std(oof_resid)), 4),
                    n_unstable_features=sum(
                        1 for v in getattr(cv_result, "importance_stability", {}).values()
                        if v > 0.5
                    ),
                )
                _record_event(
                    experiment_id,
                    event_type="modeling_completed",
                    message=(
                        f"{model_type.capitalize()} cross-validation complete — "
                        f"mean score: {cv_result.mean_score:.4f}, "
                        f"residual std: {float(np.std(oof_resid)):.4f}."
                    ),
                    phase="model",
                    metadata={
                        "model_type": model_type,
                        "mean_score": round(cv_result.mean_score, 4),
                        "std_score": round(cv_result.std_score, 4),
                        "residual_std": round(float(np.std(oof_resid)), 4),
                    },
                )

            # ── SHAP Interaction Analysis ────────────────────────────
            from ive.detection.shap_interactions import SHAPInteractionAnalyzer

            shap_interaction_pairs: list[tuple[str, str, float]] = []

            # Prefer XGBoost for SHAP (TreeExplainer is exact); fall back to linear
            shap_cv = cv_results_by_model.get("xgboost") or cv_results_by_model.get("linear")
            if shap_cv and shap_cv.fitted_models:
                try:
                    shap_analyzer = SHAPInteractionAnalyzer(sample_size=500)
                    feature_names = list(X_train.columns)
                    shap_result = shap_analyzer.compute(
                        shap_cv.fitted_models[-1],  # last fold's model
                        X_values,
                        feature_names,
                        compute_interactions=("xgboost" in cv_results_by_model),
                    )
                    shap_interaction_pairs = shap_result.top_interaction_pairs
                    self.logger.info(
                        "pipeline.shap_done",
                        model_used=("xgboost" if "xgboost" in cv_results_by_model else "linear"),
                        n_interaction_pairs=len(shap_interaction_pairs),
                        top_pair=shap_interaction_pairs[0] if shap_interaction_pairs else None,
                    )
                except Exception as shap_err:
                    self.logger.warning("pipeline.shap_failed", error=str(shap_err))
            else:
                self.logger.info("pipeline.shap_skipped", reason="no_fitted_models")

            await exp_repo.update_progress(experiment_id, 55, "detect")

            # ── Phase 3: Detect — pattern analysis ────────────────────
            # Run detection on ALL models' residuals, then score by agreement
            primary = all_residuals[-1]

            # Read analysis mode from experiment config (default to 'demo' for safety)
            analysis_mode: str = str(config.get("analysis_mode", "demo")).lower()
            if analysis_mode not in ("demo", "production"):
                analysis_mode = "demo"

            # Mode-specific subgroup discovery thresholds
            if analysis_mode == "demo":
                sg_effect_size = 0.15
                sg_min_subgroup = 20
            else:  # production
                sg_effect_size = 0.20
                sg_min_subgroup = 30

            all_patterns: list[dict[str, Any]] = []

            _record_event(
                experiment_id,
                event_type="detection_started",
                message=(
                    f"Pattern detection started in {analysis_mode!r} mode using "
                    f"{primary['model_type'].capitalize()} residuals."
                ),
                phase="detect",
                metadata={"analysis_mode": analysis_mode, "residual_source": primary["model_type"]},
            )

            n_models = len(all_residuals)

            if n_models <= 1:
                # Single model — use the simple/fast path (no agreement scoring)
                residuals = primary["residuals"]
                abs_residuals = primary["abs_residuals"]

                sg = SubgroupDiscovery(min_effect_size=sg_effect_size, min_bin_samples=sg_min_subgroup)
                sg_patterns = sg.detect(X_train, residuals)
                all_patterns.extend(sg_patterns)

                hdb = HDBSCANClustering()
                cl_patterns = hdb.detect(X_train, abs_residuals)
                all_patterns.extend(cl_patterns)
            else:
                # Multi-model ensemble: detect per model, then score by agreement
                per_model_patterns: dict[str, list[dict[str, Any]]] = {}

                for resid_data in all_residuals:
                    mtype = resid_data["model_type"]
                    m_residuals = resid_data["residuals"]
                    m_abs_residuals = resid_data["abs_residuals"]

                    model_patterns: list[dict[str, Any]] = []

                    sg = SubgroupDiscovery(min_effect_size=sg_effect_size, min_bin_samples=sg_min_subgroup)
                    sg_pats = sg.detect(X_train, m_residuals)
                    for p in sg_pats:
                        p["source_model"] = mtype
                    model_patterns.extend(sg_pats)

                    hdb = HDBSCANClustering()
                    cl_pats = hdb.detect(X_train, m_abs_residuals)
                    for p in cl_pats:
                        p["source_model"] = mtype
                    model_patterns.extend(cl_pats)

                    per_model_patterns[mtype] = model_patterns

                # Merge patterns and compute ensemble agreement
                primary_model = primary["model_type"]  # XGBoost if available
                primary_patterns = per_model_patterns.get(primary_model, [])
                other_models = [m for m in per_model_patterns if m != primary_model]

                for pat in primary_patterns:
                    pat_col = pat.get("column_name", "")
                    pat_bin = pat.get("bin_value", "")
                    pat_type = pat.get("pattern_type", "")

                    agreement_count = 0
                    agreement_models = [primary_model]

                    for other_model in other_models:
                        for other_pat in per_model_patterns[other_model]:
                            if (other_pat.get("column_name") == pat_col
                                    and other_pat.get("pattern_type") == pat_type
                                    and other_pat.get("bin_value") == pat_bin):
                                agreement_count += 1
                                agreement_models.append(other_model)
                                break

                    agreement_ratio = (agreement_count + 1) / n_models  # +1 for primary

                    # Asymmetric scoring:
                    # - Found in both: confidence boost (x1.5)
                    # - Found only in Linear: XGBoost captured the nonlinearity (annotate)
                    # - Found only in XGBoost: nonlinear latent interaction
                    if agreement_ratio >= 1.0:
                        pat["ensemble_agreement"] = "consensus"
                        pat["ensemble_confidence"] = 1.5
                    elif primary_model == "xgboost":
                        pat["ensemble_agreement"] = "xgboost_only"
                        pat["ensemble_confidence"] = 1.0  # neutral
                    else:
                        pat["ensemble_agreement"] = "linear_only"
                        pat["ensemble_confidence"] = 0.8  # slight penalty

                    pat["agreement_models"] = agreement_models
                    pat["agreement_ratio"] = agreement_ratio
                    all_patterns.append(pat)

                # Also add patterns unique to other models (not in primary)
                for other_model in other_models:
                    for other_pat in per_model_patterns[other_model]:
                        other_col = other_pat.get("column_name", "")
                        other_bin = other_pat.get("bin_value", "")
                        other_type = other_pat.get("pattern_type", "")
                        already_found = any(
                            p.get("column_name") == other_col
                            and p.get("bin_value") == other_bin
                            and p.get("pattern_type") == other_type
                            for p in primary_patterns
                        )
                        if not already_found:
                            if other_model == "linear":
                                other_pat["ensemble_agreement"] = "linear_only"
                                other_pat["ensemble_confidence"] = 0.8
                            else:
                                other_pat["ensemble_agreement"] = f"{other_model}_only"
                                other_pat["ensemble_confidence"] = 1.0
                            other_pat["agreement_models"] = [other_model]
                            other_pat["agreement_ratio"] = 1 / n_models
                            all_patterns.append(other_pat)

                # Use primary model's residuals for downstream scoring
                residuals = primary["residuals"]
                abs_residuals = primary["abs_residuals"]

                # Track subgroup/cluster counts for logging (aggregate across models)
                sg_patterns = [p for p in all_patterns if p.get("pattern_type") != "cluster"]
                cl_patterns = [p for p in all_patterns if p.get("pattern_type") == "cluster"]

                self.logger.info(
                    "pipeline.ensemble_agreement",
                    n_consensus=sum(1 for p in all_patterns if p.get("ensemble_agreement") == "consensus"),
                    n_primary_only=sum(1 for p in all_patterns if "only" in p.get("ensemble_agreement", "")),
                    n_total=len(all_patterns),
                )

            # ── Temporal analysis (if time column available) ──────────
            time_col_name = getattr(dataset, "time_column", None)
            if time_col_name and time_col_name in df.columns:
                try:
                    temporal = TemporalAnalyzer(n_bins=5)
                    # Pass training-subset of df (must align with residuals length)
                    df_train = df.loc[X_train.index] if hasattr(X_train, "index") else df.iloc[:len(residuals)]
                    temporal_patterns = temporal.analyze(df_train, residuals, [time_col_name])
                    for tp in temporal_patterns:
                        all_patterns.append({
                            "pattern_type": "temporal",
                            "column_name": tp.column,
                            "pattern_subtype": tp.pattern_type,
                            "effect_size": tp.effect_size,
                            "p_value": tp.p_value,
                            "sample_count": len(residuals),
                            "mean_residual": 0.0,
                            "std_residual": 0.0,
                            "description": tp.description,
                        })
                    self.logger.info(
                        "pipeline.temporal_done",
                        n_temporal_patterns=len(temporal_patterns),
                    )
                except Exception as te:
                    self.logger.warning("pipeline.temporal_failed", error=str(te))

            # ── SHAP interaction-based patterns ──────────────────────
            if shap_interaction_pairs:
                for feat_a, feat_b, strength in shap_interaction_pairs[:10]:
                    if feat_a in X_train.columns and feat_b in X_train.columns:
                        # Create interaction feature and test it
                        interaction_values = X_train[feat_a].values * X_train[feat_b].values
                        # Check if this interaction explains residuals
                        from scipy.stats import pearsonr

                        try:
                            corr, p_val = pearsonr(interaction_values, residuals)
                            if abs(corr) > 0.1 and p_val < 0.05:
                                all_patterns.append({
                                    "pattern_type": "interaction",
                                    "column_name": f"{feat_a}_x_{feat_b}",
                                    "feature_a": feat_a,
                                    "feature_b": feat_b,
                                    "interaction_strength": float(strength),
                                    "correlation_with_residuals": float(corr),
                                    "p_value": float(p_val),
                                    "effect_size": float(abs(corr)),
                                    "sample_count": len(residuals),
                                    "mean_residual": 0.0,
                                    "std_residual": float(np.std(residuals)),
                                })
                        except Exception:
                            pass
                self.logger.info(
                    "pipeline.interaction_patterns",
                    n_interaction_patterns=sum(
                        1 for p in all_patterns if p.get("pattern_type") == "interaction"
                    ),
                )

            # ── Score and rank patterns ───────────────────────────────
            if all_patterns:
                # PatternScorer re-computes effect sizes against PRIMARY model's
                # residuals. For ensemble patterns from other models, this tests
                # whether the pattern's signal persists in the primary model —
                # which is the cross-model validation signal.
                scorer = PatternScorer()
                scored = scorer.score_and_rank(all_patterns, residuals, top_k=20)
                if scored:
                    # Convert ScoredPattern objects back to dicts for synthesis
                    scored_dicts: list[dict[str, Any]] = []
                    for sp in scored:
                        raw = sp.raw_pattern if isinstance(sp.raw_pattern, dict) else {}
                        raw["_composite_score"] = sp.composite_score
                        scored_dicts.append(raw)
                    all_patterns = scored_dicts
                    self.logger.info(
                        "pipeline.scoring_done",
                        n_before=len(sg_patterns) + len(cl_patterns),
                        n_after=len(scored_dicts),
                    )

            # Log top patterns for operational tracing
            for i, p in enumerate(all_patterns[:10]):  # log top 10 only
                self.logger.debug(
                    "pipeline.pattern_retained",
                    pattern_idx=i,
                    pattern_type=p.get("pattern_type"),
                    column=p.get("column_name", "N/A"),
                    effect_size=round(float(p.get("effect_size", 0)), 4),
                    p_value=round(float(p.get("p_value", 0)), 6),
                    sample_count=p.get("sample_count", 0),
                )

            # Bulk-insert patterns (more efficient than individual inserts)
            if all_patterns:
                pattern_rows = [
                    {
                        "pattern_type": p.get("pattern_type", "unknown"),
                        "subgroup_definition": {k: v for k, v in p.items() if k != "pattern_type"},
                        "effect_size": float(p.get("effect_size", 0.0)),
                        "p_value": float(p.get("p_value", 0.0)),
                        "adjusted_p_value": float(p.get("adjusted_alpha", 0.0)),
                        "sample_count": int(p.get("sample_count", 0)),
                        "mean_residual": float(p.get("mean_residual", p.get("mean_error", 0.0))),
                        "std_residual": float(p.get("std_residual", p.get("std_error", 0.0))),
                    }
                    for p in all_patterns
                ]
                await exp_repo.add_error_patterns_batch(experiment_id, pattern_rows)

            self.logger.info(
                "pipeline.detection_complete",
                n_subgroup_patterns=len(sg_patterns),
                n_cluster_patterns=len(cl_patterns),
            )
            _record_event(
                experiment_id,
                event_type="detection_completed",
                message=(
                    f"Pattern detection complete: {len(sg_patterns)} subgroup pattern(s) "
                    f"and {len(cl_patterns)} cluster pattern(s) identified "
                    f"({len(all_patterns)} total)."
                ),
                phase="detect",
                metadata={
                    "n_subgroup_patterns": len(sg_patterns),
                    "n_cluster_patterns": len(cl_patterns),
                    "n_total_patterns": len(all_patterns),
                },
            )
            await exp_repo.update_progress(experiment_id, 75, "construct")

            # ── Phase 4: Construct — synthesis + validation ───────────
            explainer = ExplanationGenerator()

            if all_patterns:
                _record_event(
                    experiment_id,
                    event_type="construction_started",
                    message=(
                        f"Latent variable synthesis and bootstrap validation started "
                        f"for {len(all_patterns)} candidate pattern(s)."
                    ),
                    phase="construct",
                    metadata={"n_input_patterns": len(all_patterns)},
                )
                synth = VariableSynthesizer()
                candidates = synth.synthesize(all_patterns, X_train)

                # ── Causal plausibility check (annotate, don't discard) ──
                from ive.construction.causal_checker import CausalChecker

                causal_checker = CausalChecker()
                candidates = causal_checker.filter(candidates, X_train, target_column=target_col)

                validator = BootstrapValidator(mode=analysis_mode)  # type: ignore[arg-type]
                validated = validator.validate(
                    X_train, candidates, n_iterations=int(config.get("bootstrap_iterations", 50))
                )

                # Log per-candidate outcomes for traceability
                for var in validated:
                    self.logger.info(
                        "pipeline.candidate_outcome",
                        name=var.get("name", "unknown"),
                        status=var.get("status"),
                        presence_rate=round(float(var.get("bootstrap_presence_rate", 0)), 4),
                        holdout_validated=var.get("holdout_validated"),
                        rejection_reason=var.get("rejection_reason"),
                        causal_warnings=var.get("causal_warnings", []),
                    )

                # ── Holdout validation ─────────────────────────────────
                if len(X_holdout) > 0 and validated:
                    synth_holdout = VariableSynthesizer()
                    holdout_candidates = synth_holdout.synthesize(all_patterns, X_holdout)

                    for var in validated:
                        if var.get("status") != "validated":
                            continue

                        # Find matching holdout candidate by name
                        matching = [
                            c for c in holdout_candidates if c.get("name") == var.get("name")
                        ]
                        if not matching:
                            var["holdout_validated"] = False
                            var["holdout_note"] = "Could not reconstruct on holdout set"
                            continue

                        holdout_scores = matching[0].get("scores")
                        if holdout_scores is None or len(holdout_scores) == 0:
                            var["holdout_validated"] = False
                            var["holdout_note"] = "No scores on holdout set"
                            continue

                        # Check if the variable separates holdout data meaningfully
                        mask = holdout_scores > 0
                        if mask.sum() < 5 or (~mask).sum() < 5:
                            var["holdout_validated"] = False
                            var["holdout_note"] = "Insufficient holdout samples in subgroups"
                            continue

                        # Check if the construction rule produces meaningful variance
                        holdout_variance = float(np.var(holdout_scores))
                        holdout_range = float(
                            np.max(holdout_scores) - np.min(holdout_scores)
                        )

                        var["holdout_validated"] = bool(
                            holdout_variance > 1e-6 and holdout_range > 0.01
                        )
                        var["holdout_variance"] = holdout_variance
                        var["holdout_range"] = holdout_range

                    n_holdout_passed = sum(
                        1 for v in validated if v.get("holdout_validated") is True
                    )
                    n_validated = sum(
                        1 for v in validated if v.get("status") == "validated"
                    )
                    self.logger.info(
                        "pipeline.holdout_validation",
                        n_holdout_passed=n_holdout_passed,
                        n_validated=n_validated,
                        holdout_size=len(X_holdout),
                    )
                    self.logger.info(
                        "pipeline.holdout_summary",
                        n_validated=n_validated,
                        n_holdout_passed=n_holdout_passed,
                        n_holdout_failed=sum(
                            1 for v in validated if v.get("holdout_validated") is False
                        ),
                        n_causal_penalized=sum(
                            1 for v in validated
                            if v.get("causal_confidence_penalty", 1.0) < 1.0
                        ),
                    )

                # Build rows for bulk_create
                lv_rows: list[dict[str, Any]] = []
                for var in validated:
                    presence = float(var.get("bootstrap_presence_rate", 0.0))
                    explanation = explainer.generate_latent_variable_explanation(var)
                    recommendation = explainer.generate_business_recommendation(var)
                    row: dict[str, Any] = {
                        "name": var.get("name", "Unknown"),
                        "description": recommendation,
                        "construction_rule": var.get("construction_rule", {}),
                        "source_pattern_ids": [],
                        "importance_score": float(var.get("stability_score", 0.0)),
                        "stability_score": float(var.get("stability_score", 0.0)),
                        "bootstrap_presence_rate": presence,
                        "explanation_text": explanation,
                        "status": var.get("status", "rejected"),
                    }
                    # Persist rejection reason for rejected candidates
                    if var.get("rejection_reason"):
                        row["rejection_reason"] = var["rejection_reason"]

                    # Embed holdout validation results
                    if "holdout_validated" in var:
                        holdout_info = {
                            "holdout_validated": var["holdout_validated"],
                        }
                        if "holdout_note" in var:
                            holdout_info["holdout_note"] = var["holdout_note"]
                        if "holdout_variance" in var:
                            holdout_info["holdout_variance"] = var["holdout_variance"]
                        if "holdout_range" in var:
                            holdout_info["holdout_range"] = var["holdout_range"]

                        # Merge into construction_rule dict
                        cr = row.get("construction_rule", {})
                        if isinstance(cr, dict):
                            cr["holdout_validation"] = holdout_info
                            row["construction_rule"] = cr

                        # Also append to explanation_text for visibility
                        hv_label = "PASSED" if var["holdout_validated"] else "FAILED"
                        note = var.get("holdout_note", "")
                        hv_text = f"\n[Holdout validation: {hv_label}]"
                        if note:
                            hv_text += f" {note}"
                        row["explanation_text"] = (row.get("explanation_text") or "") + hv_text

                    # Append causal warnings to explanation text
                    causal_warnings = var.get("causal_warnings", [])
                    causal_penalty = var.get("causal_confidence_penalty", 1.0)
                    if causal_warnings:
                        warnings_text = "; ".join(causal_warnings)
                        row["explanation_text"] = (
                            (row.get("explanation_text") or "")
                            + f"\n[Causal check: {warnings_text} (confidence x{causal_penalty})]"
                        )

                    # Include model improvement metrics if available
                    if var.get("model_improvement_pct"):
                        row["model_improvement_pct"] = var["model_improvement_pct"]

                    # Sanitize numpy types for JSONB serialization
                    row["construction_rule"] = _sanitize_for_json(row.get("construction_rule", {}))

                    lv_rows.append(row)

                if lv_rows:
                    lv_repo = LatentVariableRepository(self.session, LatentVariable)
                    await lv_repo.bulk_create(experiment_id, lv_rows)

                n_validated = sum(1 for v in validated if v.get("status") == "validated")
                n_rejected = sum(1 for v in validated if v.get("status") == "rejected")
                self.logger.info(
                    "pipeline.construction_complete",
                    n_candidates=len(candidates),
                    n_validated=n_validated,
                    n_rejected=n_rejected,
                    analysis_mode=analysis_mode,
                )
                _record_event(
                    experiment_id,
                    event_type="construction_completed",
                    message=(
                        f"Bootstrap validation complete: {n_validated} latent variable(s) validated, "
                        f"{n_rejected} rejected out of {len(candidates)} candidate(s)."
                    ),
                    phase="construct",
                    metadata={
                        "n_candidates": len(candidates),
                        "n_validated": n_validated,
                        "n_rejected": n_rejected,
                        "analysis_mode": analysis_mode,
                    },
                )
            else:
                validated = []
                self.logger.info("pipeline.no_patterns_found")
                _record_event(
                    experiment_id,
                    event_type="construction_completed",
                    message="No patterns were detected; latent variable construction was skipped.",
                    phase="construct",
                    metadata={"n_candidates": 0, "n_validated": 0},
                )

            # ── Phase 5: Evaluate — retrain with latent variables ────
            validated_lvs = [v for v in validated if v.get("status") == "validated"]

            if validated_lvs and len(X_holdout) >= 20:
                self.logger.info("pipeline.retrain_start", n_lvs=len(validated_lvs))

                from sklearn.metrics import r2_score
                try:
                    from sklearn.metrics import roc_auc_score
                except ImportError:
                    roc_auc_score = None  # type: ignore[assignment]

                # Determine task type (regression vs classification)
                n_unique_target = len(np.unique(y_train))
                is_classification = n_unique_target <= 20 and np.issubdtype(
                    y_train.dtype, np.integer
                )

                # Baseline: train on X_train, evaluate on X_holdout
                baseline_model = XGBoostIVEModel()
                try:
                    baseline_model.fit(X_values, y_train)
                    baseline_preds = baseline_model.predict(X_holdout_processed)

                    if is_classification and roc_auc_score:
                        baseline_metric = float(roc_auc_score(y_holdout, baseline_preds))
                        metric_name = "auc"
                    else:
                        baseline_metric = float(r2_score(y_holdout, baseline_preds))
                        metric_name = "r2"

                    self.logger.info(
                        "pipeline.baseline_metric",
                        metric=metric_name,
                        value=round(baseline_metric, 4),
                    )
                except Exception as base_err:
                    self.logger.warning("pipeline.baseline_failed", error=str(base_err))
                    baseline_metric = None
                    metric_name = "r2"

                if baseline_metric is not None:
                    # Greedy forward selection of LVs
                    selected_lvs: list[dict[str, Any]] = []
                    remaining_lvs = list(validated_lvs)
                    current_metric = baseline_metric

                    while remaining_lvs:
                        best_improvement = -float("inf")
                        best_lv: dict[str, Any] | None = None
                        best_lv_metric = current_metric

                        for lv in remaining_lvs:
                            # Build augmented features: X + selected LVs + this candidate
                            try:
                                aug_cols_train: list[np.ndarray[Any, Any]] = []
                                aug_cols_holdout: list[np.ndarray[Any, Any]] = []
                                for sel_lv in selected_lvs + [lv]:
                                    rule = sel_lv.get("construction_rule", {})
                                    ptype = sel_lv.get("pattern_type", "subgroup")
                                    train_scores = apply_construction_rule(rule, ptype, X_train)
                                    holdout_scores = apply_construction_rule(rule, ptype, X_holdout)
                                    aug_cols_train.append(train_scores)
                                    aug_cols_holdout.append(holdout_scores)

                                # Augment feature matrices
                                X_aug_train = np.column_stack(
                                    [X_values] + [s.reshape(-1, 1) for s in aug_cols_train]
                                )
                                X_aug_holdout = np.column_stack(
                                    [X_holdout_processed]
                                    + [s.reshape(-1, 1) for s in aug_cols_holdout]
                                )

                                # Retrain and evaluate
                                aug_model = XGBoostIVEModel()
                                aug_model.fit(X_aug_train, y_train)
                                aug_preds = aug_model.predict(X_aug_holdout)

                                if is_classification and roc_auc_score:
                                    aug_metric = float(roc_auc_score(y_holdout, aug_preds))
                                else:
                                    aug_metric = float(r2_score(y_holdout, aug_preds))

                                improvement = aug_metric - current_metric
                                if improvement > best_improvement:
                                    best_improvement = improvement
                                    best_lv = lv
                                    best_lv_metric = aug_metric
                            except Exception:
                                continue

                        if best_lv is None or best_improvement <= 0:
                            break  # No more beneficial LVs

                        selected_lvs.append(best_lv)
                        remaining_lvs.remove(best_lv)

                        # Record marginal improvement on the LV
                        improvement_pct = (
                            (best_improvement / max(abs(current_metric), 1e-10)) * 100
                        )
                        best_lv["model_improvement_pct"] = {
                            "metric": metric_name,
                            "baseline": round(current_metric, 6),
                            "augmented": round(best_lv_metric, 6),
                            "marginal_improvement": round(best_improvement, 6),
                            "improvement_pct": round(improvement_pct, 2),
                            "selection_order": len(selected_lvs),
                        }

                        current_metric = best_lv_metric
                        self.logger.info(
                            "pipeline.retrain_step",
                            lv_name=best_lv.get("name"),
                            order=len(selected_lvs),
                            improvement_pct=round(improvement_pct, 2),
                            new_metric=round(best_lv_metric, 4),
                        )

                    # Combined improvement (all selected LVs vs baseline)
                    if selected_lvs:
                        combined_improvement_pct = (
                            (current_metric - baseline_metric)
                            / max(abs(baseline_metric), 1e-10)
                        ) * 100
                        self.logger.info(
                            "pipeline.retrain_complete",
                            n_selected=len(selected_lvs),
                            baseline=round(baseline_metric, 4),
                            final=round(current_metric, 4),
                            combined_improvement_pct=round(combined_improvement_pct, 2),
                        )
                    else:
                        self.logger.info("pipeline.retrain_no_improvement")

            # ── Generate experiment summary ───────────────────────────
            dataset_name = getattr(dataset, "name", "dataset")
            experiment_summary = explainer.generate_experiment_summary(
                patterns=all_patterns,
                candidates=validated,
                dataset_name=dataset_name,
                target_column=target_col,
                analysis_mode=analysis_mode,
            )

            # ── Complete ──────────────────────────────────────────────
            await exp_repo.update_progress(experiment_id, 100, "complete")

            elapsed = round(time.perf_counter() - t_start, 2)

            n_validated_final = sum(1 for v in validated if v.get("status") == "validated")
            _record_event(
                experiment_id,
                event_type="experiment_completed",
                message=(
                    f"Experiment completed successfully in {elapsed}s — "
                    f"{len(all_patterns)} pattern(s) detected, "
                    f"{n_validated_final} latent variable(s) validated."
                ),
                phase="construct",
                metadata={
                    "elapsed_seconds": elapsed,
                    "n_patterns": len(all_patterns),
                    "n_validated": n_validated_final,
                },
            )
            await exp_repo.mark_completed(experiment_id)

            self.logger.info(
                "pipeline.complete",
                experiment_id=str(experiment_id),
                elapsed_seconds=elapsed,
                n_patterns=len(all_patterns),
            )

            return {
                "status": "completed",
                "n_patterns": len(all_patterns),
                "n_validated": sum(1 for v in validated if v.get("status") == "validated"),
                "elapsed_seconds": elapsed,
                "analysis_mode": analysis_mode,
                "experiment_summary": experiment_summary,
            }

        except Exception as exc:
            self.logger.error(
                "pipeline.failed",
                experiment_id=str(experiment_id),
                error=str(exc),
                exc_info=True,
            )
            try:
                _record_event(
                    experiment_id,
                    event_type="experiment_failed",
                    message=f"Pipeline execution failed: {exc}",
                    phase=None,
                    metadata={"error": str(exc)[:1000]},
                )
                await exp_repo.mark_failed(experiment_id, str(exc))
            except Exception:
                pass
            raise
