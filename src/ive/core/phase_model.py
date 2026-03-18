"""
Phase 2 — Model.

Trains baseline ML models on pre-processed features and extracts cross-validated
Out-Of-Fold (OOF) residuals.  The OOF residuals are the mathematical heart of the
IVE system: they represent *exactly* what the model does not know, which Phase 3
mines for hidden structure.

Workflow (per requested model_type)
------------------------------------
  2.1  Load dataset metadata → ArtifactStore → parse CSV → pandas DataFrame
  2.2  Preprocess features (numeric scaling, categorical encoding, drop nulls)
  2.3  Run K-fold cross-validation via CrossValidator
  2.4  Analyse residual structure via ResidualAnalyzer
  2.5  Persist each fold's trained model to ArtifactStore (pickle)
  2.6  Record TrainedModel rows in the DB
  2.7  Bulk-insert per-sample Residual rows (batches of 5 000)
  2.8  Return summary statistics dict

Outputs written to PipelineContext::

    ctx.residuals         — np.ndarray of OOF residuals (averaged across models)
    ctx.predictions       — np.ndarray of OOF predictions (last model wins)
    ctx.feature_matrix    — preprocessed numpy feature array
    ctx.model_artifacts   — {model_type: CVResult}

Raises
------
PhaseExecutionError
    Wraps any exception raised during any orchestration step.
"""

from __future__ import annotations

import io
import math
import time
import traceback
import uuid
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from ive.core.pipeline import PhaseBase, PhaseResult, PipelineContext
from ive.db.models import Dataset
from ive.db.repositories.dataset_repo import DatasetRepository
from ive.db.repositories.experiment_repo import ExperimentRepository
from ive.models.cross_validator import CrossValidator, CVResult
from ive.models.linear_model import LinearIVEModel
from ive.models.residual_analyzer import ResidualAnalyzer
from ive.models.xgboost_model import XGBoostIVEModel
from ive.storage.artifact_store import ArtifactStore

log = structlog.get_logger(__name__)

# Maximum rows sent to ExperimentRepository.add_residuals_batch in one call.
# The repo itself chunks further (500 rows) to stay within PG param limits.
_RESIDUAL_BATCH_SIZE = 5_000

# Progress milestones reported after each model type completes.
_PROGRESS_START = 25  # entering phase 2
_PROGRESS_END = 60  # leaving phase 2 (detect starts at 60)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class PhaseExecutionError(Exception):
    """Raised when a step in the Phase 2 orchestration pipeline fails.

    Wraps the original ``cause`` for full traceback preservation.

    Args:
        step:    Human-readable label for the failing step.
        message: Concise error description.
        cause:   Original exception.
    """

    def __init__(self, step: str, message: str, cause: Exception | None = None) -> None:
        self.step = step
        self.cause = cause
        super().__init__(f"[phase_model:{step}] {message}")
        if cause:
            self.__cause__ = cause


# ---------------------------------------------------------------------------
# Phase 2 orchestrator
# ---------------------------------------------------------------------------


class PhaseModel(PhaseBase):
    """Phase 2: Train baseline models and extract cross-validated residuals.

    Can be used in two modes:

    **Pipeline mode** (standard): Instantiate with ``PhaseModel()``; the engine
    calls ``execute(ctx)`` and the phase pulls config from ``ctx``.

    **Service mode** (Celery task injection): Instantiate with dependencies
    injected via ``__init__``.  Call ``run(model_types, task_type, cv_folds, …)``.

    Both modes share the same internal implementation.
    """

    def __init__(
        self,
        db_session: AsyncSession | None = None,
        artifact_store: ArtifactStore | None = None,
        experiment_id: uuid.UUID | None = None,
        dataset_id: uuid.UUID | None = None,
    ) -> None:
        """Initialise the Phase 2 orchestrator.

        When invoked from the engine, all arguments should be omitted (``None``);
        they are resolved from the :class:`~ive.core.pipeline.PipelineContext`
        inside :meth:`execute`.

        When invoked from a Celery task, pass all four arguments so that
        :meth:`run` can be called directly without a ``PipelineContext``.

        Args:
            db_session:    Active async SQLAlchemy session.
            artifact_store: Storage backend for dataset CSV and model pickles.
            experiment_id:  UUID of the experiment being run.
            dataset_id:     UUID of the parent dataset record.
        """
        self._db_session = db_session
        self._artifact_store = artifact_store
        self._experiment_id = experiment_id
        self._dataset_id = dataset_id

    # ------------------------------------------------------------------
    # PhaseBase interface
    # ------------------------------------------------------------------

    def get_phase_name(self) -> str:
        """Return the canonical phase identifier."""
        return "model"

    async def execute(self, ctx: PipelineContext) -> PhaseResult:
        """Run Phase 2 within a pipeline context.

        Reads configuration from *ctx*, delegates to private implementation,
        and writes OOF residuals / predictions / model artefacts back to *ctx*.

        Args:
            ctx: Shared pipeline context.  Must have ``ctx.df``,
                 ``ctx.feature_columns``, ``ctx.target_series``,
                 ``ctx.config``, and ``ctx.experiment_id`` populated by Phase 1.

        Returns:
            :class:`~ive.core.pipeline.PhaseResult` with summary metadata.

        Raises:
            PhaseExecutionError: If any internal step fails.
        """
        start = time.monotonic()
        log.info("ive.phase.model.start", experiment_id=str(ctx.experiment_id))

        if ctx.df is None or ctx.target_series is None:
            raise PhaseExecutionError(
                step="validate_context",
                message="ctx.df and ctx.target_series must be populated by Phase 1 before Phase 2.",
            )

        config = ctx.config  # type: ignore[attr-defined]
        model_types: list[str] = getattr(config, "model_types", ["linear", "xgboost"])
        cv_folds: int = getattr(config, "cv_folds", 5)
        task_type: str = getattr(config, "task_type", "regression")
        random_seed: int = getattr(config, "random_seed", 42)

        try:
            summary, X, all_residuals = await self._run_model_loop(
                df=ctx.df,
                feature_columns=ctx.feature_columns,
                y=ctx.target_series.values,
                X_df_original=ctx.df[ctx.feature_columns] if ctx.feature_columns else ctx.df,
                model_types=model_types,
                task_type=task_type,
                cv_folds=cv_folds,
                random_seed=random_seed,
                experiment_id=ctx.experiment_id,
                db_session=self._db_session,
                artifact_store=self._artifact_store,
            )
        except PhaseExecutionError:
            raise
        except Exception as exc:
            raise PhaseExecutionError(
                "model_loop",
                f"Unexpected failure: {exc}",
                cause=exc,
            ) from exc

        # ── Write back to context ──────────────────────────────────────
        ctx.feature_matrix = X
        if all_residuals:
            ctx.residuals = np.mean(all_residuals, axis=0)

        elapsed = time.monotonic() - start
        log.info(
            "ive.phase.model.complete",
            elapsed_s=round(elapsed, 2),
            model_types=model_types,
        )

        return PhaseResult(
            phase_name="model",
            success=True,
            duration_seconds=elapsed,
            metadata=summary,
        )

    # ------------------------------------------------------------------
    # Service / Celery entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        model_types: list[str] | None = None,
        task_type: str = "regression",
        cv_folds: int = 5,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Execute Phase 2 in service mode (Celery task or direct call).

        Loads the dataset CSV from the artifact store, preprocesses features,
        runs cross-validated training for each requested model type, persists
        model artefacts and residuals to the database, and returns a summary
        statistics dict.

        Args:
            model_types:  List of model identifiers to train
                          (``"linear"``, ``"xgboost"``).  Defaults to both.
            task_type:    ``"regression"`` or ``"classification"``.
            cv_folds:     Number of cross-validation folds (default 5).
            random_seed:  Global random seed for reproducibility.

        Returns:
            Dictionary of summary statistics, e.g.::

                {
                    "linear_rmse": 0.423,
                    "linear_r2": 0.781,
                    "xgboost_rmse": 0.318,
                    "xgboost_r2": 0.854,
                    "n_samples": 5000,
                    "n_features": 12,
                    "model_types": ["linear", "xgboost"],
                }

        Raises:
            PhaseExecutionError: If any required dependency is missing or a
                step fails during orchestration.
        """
        if self._db_session is None or self._artifact_store is None:
            raise PhaseExecutionError(
                "init",
                "db_session and artifact_store must be provided when using service mode run().",
            )
        if self._experiment_id is None or self._dataset_id is None:
            raise PhaseExecutionError(
                "init",
                "experiment_id and dataset_id must be provided when using service mode run().",
            )

        if model_types is None:
            model_types = ["linear", "xgboost"]

        try:
            # ── Step 1: Load dataset metadata ─────────────────────────
            dataset = await self._load_dataset_metadata()

            # ── Step 2: Download CSV → DataFrame ──────────────────────
            df = await self._load_dataframe(dataset)

            # ── Step 3: Extract X / y ─────────────────────────────────
            target_col = dataset.target_column
            time_col = dataset.time_column

            feature_columns = [c for c in df.columns if c != target_col and c != time_col]

            # ── Step 4: Drop all-null columns; drop time column ───────
            df = _drop_degenerate_columns(df, feature_columns)
            feature_columns = [c for c in feature_columns if c in df.columns]

            y = df[target_col].values.astype(float)
            X_df = df[feature_columns].copy()

            # ── Step 5: Update experiment progress ────────────────────
            await self._report_progress(
                _PROGRESS_START,
                "Phase 2: Training Base Models",
            )

            # ── Steps 6–7: Train, persist, insert residuals ───────────
            summary, _, _ = await self._run_model_loop(
                df=df,
                feature_columns=feature_columns,
                y=y,
                X_df_original=X_df,
                model_types=model_types,
                task_type=task_type,
                cv_folds=cv_folds,
                random_seed=random_seed,
                experiment_id=self._experiment_id,
                db_session=self._db_session,
                artifact_store=self._artifact_store,
            )

            await self._report_progress(_PROGRESS_END, "Phase 2: Complete")
            return summary

        except PhaseExecutionError:
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            log.error(
                "ive.phase.model.fatal_error",
                experiment_id=str(self._experiment_id),
                error=str(exc),
                traceback=tb,
            )
            raise PhaseExecutionError(
                step="run",
                message=str(exc),
                cause=exc,
            ) from exc

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------

    async def _run_model_loop(
        self,
        *,
        df: pd.DataFrame,
        feature_columns: list[str],
        y: np.ndarray,
        X_df_original: pd.DataFrame,
        model_types: list[str],
        task_type: str,
        cv_folds: int,
        random_seed: int,
        experiment_id: uuid.UUID,
        db_session: AsyncSession | None,
        artifact_store: ArtifactStore | None,
    ) -> tuple[dict[str, Any], np.ndarray, list[np.ndarray]]:
        """Core loop: preprocess → CV → analyse → persist.

        Args:
            df:             Full cleaned DataFrame.
            feature_columns: Columns to use as features.
            y:              Target values array.
            X_df_original:  Feature DataFrame (original index preserved for
                            residual record generation).
            model_types:    List of model type strings.
            task_type:      ``"regression"`` or ``"classification"``.
            cv_folds:       Number of folds.
            random_seed:    Random seed.
            experiment_id:  Experiment UUID.
            db_session:     Optional async DB session.
            artifact_store: Optional artifact store.

        Returns:
            Tuple of (summary_dict, X_array, list_of_residual_arrays).
        """
        # ── Preprocess features ────────────────────────────────────────
        try:
            X, feature_names = _preprocess_features(X_df_original, feature_columns)
        except Exception as exc:
            raise PhaseExecutionError(
                "preprocess",
                f"Feature preprocessing failed: {exc}",
                cause=exc,
            ) from exc

        n_samples, n_features = X.shape
        log.info(
            "ive.phase.model.preprocessed",
            n_samples=n_samples,
            n_features=n_features,
            experiment_id=str(experiment_id),
        )

        summary: dict[str, Any] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "model_types": model_types,
        }
        all_residuals: list[np.ndarray] = []

        # Progress increment per model type
        n_models = max(1, len(model_types))
        progress_step = (_PROGRESS_END - _PROGRESS_START) // n_models

        for model_idx, model_type in enumerate(model_types):
            loop_log = log.bind(
                model_type=model_type,
                experiment_id=str(experiment_id),
            )
            loop_log.info("ive.phase.model.training_start")
            model_start = time.monotonic()

            # ── 6a: Instantiate model ──────────────────────────────────
            try:
                model = _build_model(model_type)
            except ValueError as exc:
                raise PhaseExecutionError(
                    "build_model",
                    str(exc),
                    cause=exc,
                ) from exc

            # ── 6b: Cross-validate ─────────────────────────────────────
            stratified = task_type == "classification"
            cv = CrossValidator(
                model=model,
                n_splits=cv_folds,
                seed=random_seed,
                stratified=stratified,
            )

            try:
                cv_result: CVResult = cv.fit(X, y)
            except Exception as exc:
                raise PhaseExecutionError(
                    f"cross_validate[{model_type}]",
                    f"CV failed: {exc}",
                    cause=exc,
                ) from exc

            all_residuals.append(cv_result.oof_residuals)

            # ── Compute summary metrics ────────────────────────────────
            rmse = float(np.sqrt(np.mean(cv_result.oof_residuals**2)))
            r2 = float(cv_result.mean_score) if not stratified else float(np.nan)
            training_elapsed = time.monotonic() - model_start

            summary[f"{model_type}_rmse"] = round(rmse, 6)
            summary[f"{model_type}_r2"] = round(r2, 6)
            summary[f"{model_type}_mean_cv_score"] = round(float(cv_result.mean_score), 6)
            summary[f"{model_type}_std_cv_score"] = round(float(cv_result.std_score), 6)

            # ── 6c: Residual analysis ──────────────────────────────────
            analyzer = ResidualAnalyzer()
            residual_analysis = analyzer.analyze(cv_result.oof_residuals, X)

            residual_records = analyzer.build_residual_records(
                X_df=X_df_original.reset_index(drop=True),
                y_true=y,
                oof_predictions=cv_result.oof_predictions,
                fold_assignments=cv_result.fold_assignments,
                task_type=task_type,
            )

            # Inject model_type into each record for the Residual table
            for rec in residual_records:
                rec["model_type"] = model_type

            loop_log.info(
                "ive.phase.model.cv_done",
                rmse=round(rmse, 4),
                r2=round(r2, 4) if not math.isnan(r2) else "n/a",
                n_warnings=len(residual_analysis.warnings),
                elapsed_s=round(training_elapsed, 2),
            )

            # ── 6d/e: Persist fold model artefacts to DB ───────────────
            if db_session is not None and artifact_store is not None:
                exp_repo = ExperimentRepository(
                    db_session, type(db_session).__mro__[0]
                )  # session already typed

                for fold_idx, fold_model in enumerate(cv_result.fitted_models):
                    fold_start = time.monotonic()

                    # Serialise the fitted sklearn/XGBoost model to bytes
                    artifact_path: str | None = None
                    try:
                        artifact_path = await artifact_store.save_pickle(
                            obj=fold_model,
                            category="models",
                            filename=f"{model_type}_fold{fold_idx}.pkl",
                            experiment_id=str(experiment_id),
                        )
                    except Exception as exc:
                        loop_log.warning(
                            "ive.phase.model.artifact_save_failed",
                            fold=fold_idx,
                            error=str(exc),
                        )

                    fold_elapsed = time.monotonic() - fold_start

                    # Fold score from CVResult (R² or AUC)
                    fold_score = (
                        cv_result.fold_scores[fold_idx]
                        if fold_idx < len(cv_result.fold_scores)
                        else 0.0
                    )
                    feat_importances = (
                        cv_result.feature_importances[fold_idx]
                        if fold_idx < len(cv_result.feature_importances)
                        else {}
                    )

                    metric_name = "r2" if not stratified else "roc_auc"

                    try:
                        await _add_trained_model(
                            db_session=db_session,
                            experiment_id=experiment_id,
                            model_type=model_type,
                            fold_number=fold_idx,
                            val_metric=fold_score,
                            metric_name=metric_name,
                            artifact_path=artifact_path,
                            hyperparams=fold_model.get_params(),
                            feature_importances=feat_importances,
                            training_time_seconds=fold_elapsed,
                        )
                    except Exception as exc:
                        loop_log.warning(
                            "ive.phase.model.db_model_insert_failed",
                            fold=fold_idx,
                            error=str(exc),
                        )

                # ── 6f: Bulk-insert residuals ──────────────────────────
                try:
                    total_inserted = await _bulk_insert_residuals(
                        db_session=db_session,
                        experiment_id=experiment_id,
                        records=residual_records,
                        batch_size=_RESIDUAL_BATCH_SIZE,
                    )
                    loop_log.info(
                        "ive.phase.model.residuals_inserted",
                        total=total_inserted,
                    )
                except Exception as exc:
                    raise PhaseExecutionError(
                        f"residual_bulk_insert[{model_type}]",
                        f"Residual bulk insert failed: {exc}",
                        cause=exc,
                    ) from exc

                # Update progress after each model completes
                progress = _PROGRESS_START + (model_idx + 1) * progress_step
                await self._report_progress(
                    min(progress, _PROGRESS_END - 1),
                    f"Phase 2: {model_type} complete",
                )

            loop_log.info(
                "ive.phase.model.training_done",
                rmse=round(rmse, 4),
            )

        return summary, X, all_residuals

    # ------------------------------------------------------------------
    # Helper: dataset loading
    # ------------------------------------------------------------------

    async def _load_dataset_metadata(self) -> Dataset:
        """Fetch the ``Dataset`` row from the database.

        Returns:
            ``Dataset`` ORM instance.

        Raises:
            PhaseExecutionError: If the dataset is not found.
        """
        assert self._db_session is not None  # noqa: S101
        repo = DatasetRepository(self._db_session, Dataset)
        dataset = await repo.get_by_id(self._dataset_id)  # type: ignore[arg-type]
        if dataset is None:
            raise PhaseExecutionError(
                "load_dataset_metadata",
                f"Dataset {self._dataset_id} not found in database.",
            )
        log.debug(
            "ive.phase.model.dataset_loaded",
            dataset_id=str(self._dataset_id),
            file_path=dataset.file_path,
        )
        return dataset

    async def _load_dataframe(self, dataset: Dataset) -> pd.DataFrame:
        """Download CSV from the artifact store and parse it into a DataFrame.

        Args:
            dataset: ``Dataset`` ORM instance with ``file_path`` and
                     ``target_column`` fields.

        Returns:
            Parsed ``pd.DataFrame``.

        Raises:
            PhaseExecutionError: If the file is missing or cannot be parsed.
        """
        assert self._artifact_store is not None  # noqa: S101
        try:
            raw_bytes = await self._artifact_store.load_file(dataset.file_path)
        except FileNotFoundError as exc:
            raise PhaseExecutionError(
                "load_csv",
                f"Dataset file not found at {dataset.file_path!r}.",
                cause=exc,
            ) from exc

        try:
            df = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise PhaseExecutionError(
                "parse_csv",
                f"Failed to parse CSV: {exc}",
                cause=exc,
            ) from exc

        log.debug(
            "ive.phase.model.csv_parsed",
            rows=len(df),
            cols=len(df.columns),
        )
        return df

    async def _report_progress(self, progress_pct: int, stage: str) -> None:
        """Update experiment progress in the database (best-effort).

        Failures are logged as warnings and do not abort the pipeline.

        Args:
            progress_pct: Integer 0–100.
            stage:        Human-readable stage label.
        """
        if self._db_session is None or self._experiment_id is None:
            return
        try:
            repo = ExperimentRepository(self._db_session, type(None))  # type: ignore[arg-type]
            await repo.update_progress(self._experiment_id, progress_pct, stage)
        except Exception as exc:
            log.warning(
                "ive.phase.model.progress_update_failed",
                progress_pct=progress_pct,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions — no state, fully testable)
# ---------------------------------------------------------------------------


def _build_model(model_type: str) -> LinearIVEModel | XGBoostIVEModel:
    """Instantiate the correct IVEModel for the given type string.

    Args:
        model_type: ``"linear"`` or ``"xgboost"``.

    Returns:
        Unfitted model instance.

    Raises:
        ValueError: For unrecognised type strings.
    """
    if model_type == "linear":
        return LinearIVEModel()
    if model_type == "xgboost":
        return XGBoostIVEModel()
    raise ValueError(f"Unknown model_type {model_type!r}. Supported: 'linear', 'xgboost'.")


def _drop_degenerate_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Drop feature columns that are entirely null.

    Completely-null columns provide zero information and would cause
    errors in downstream imputation or scaling steps.

    Args:
        df:              Input DataFrame.
        feature_columns: Column names to inspect.

    Returns:
        DataFrame with all-null feature columns removed.
    """
    all_null = [c for c in feature_columns if c in df.columns and df[c].isna().all()]
    if all_null:
        log.info("ive.phase.model.dropping_null_columns", columns=all_null)
        df = df.drop(columns=all_null)
    return df


def _preprocess_features(
    X_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Transform raw feature DataFrame into a numeric numpy array.

    Processing steps (all leak-free — performed on the full dataset before
    the CV split so that the CV loop operates on already-encoded arrays):

    1. Retain only ``feature_columns`` that still exist in ``X_df``.
    2. Impute missing numeric values with column median.
    3. Impute missing categorical values with the string ``"__missing__"``.
    4. One-hot encode object/category columns (``drop_first=True`` to avoid
       perfect multicollinearity).
    5. Z-score normalise numeric columns (mean=0, std=1).

    Note: Imputation and scaling are fit on the full dataset here.  For a
    production-grade pipeline, these should be refitted inside each CV fold.
    The ``DataPreprocessor`` class is the intended home for that logic once
    its preprocessing stub is replaced with a real ColumnTransformer pipeline.

    Args:
        X_df:            Raw feature DataFrame (rows = samples, cols = features).
        feature_columns: Ordered list of column names to include.

    Returns:
        Tuple of (X: np.ndarray shape (n_samples, n_features), feature_names_out).
    """
    from sklearn.preprocessing import StandardScaler

    # Work on a copy to avoid mutating caller's DataFrame
    df = X_df[[c for c in feature_columns if c in X_df.columns]].copy()

    if df.empty:
        raise ValueError("No valid feature columns remain after filtering.")

    # ── Identify column groups ─────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # ── Impute numeric ─────────────────────────────────────────────────
    for col in numeric_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)

    # ── Impute categorical ─────────────────────────────────────────────
    for col in cat_cols:
        df[col] = df[col].fillna("__missing__").astype(str)

    # ── One-hot encode categoricals ────────────────────────────────────
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    # ── Scale numerics ─────────────────────────────────────────────────
    numeric_in_encoded = [c for c in numeric_cols if c in df.columns]
    if numeric_in_encoded:
        scaler = StandardScaler()
        df[numeric_in_encoded] = scaler.fit_transform(df[numeric_in_encoded])

    feature_names_out = df.columns.tolist()
    X = df.values.astype(np.float64)

    return X, feature_names_out


async def _add_trained_model(
    *,
    db_session: AsyncSession,
    experiment_id: uuid.UUID,
    model_type: str,
    fold_number: int,
    val_metric: float,
    metric_name: str,
    artifact_path: str | None,
    hyperparams: dict[str, Any],
    feature_importances: dict[str, float],
    training_time_seconds: float,
) -> None:
    """Insert a single ``TrainedModel`` row.

    Uses ``ExperimentRepository`` for consistency with the rest of the DB layer.
    Passes ``train_metric=val_metric`` as a placeholder — computing a true
    training-set score would require re-predicting on every train fold.

    Args:
        db_session:           Active async session.
        experiment_id:        Parent experiment UUID.
        model_type:           Model type string (``"linear"``, ``"xgboost"``).
        fold_number:          Zero-indexed fold number.
        val_metric:           Validation-set metric value for this fold.
        metric_name:          Name of the metric (e.g. ``"r2"``, ``"roc_auc"``).
        artifact_path:        Storage path of the serialised model pickle.
        hyperparams:          Model hyperparameter dict for the DB record.
        feature_importances:  Feature importance dict for the DB record.
        training_time_seconds: Wall-clock training time for this fold.
    """
    # ExperimentRepository needs the model class for BaseRepository generic;
    # we import lazily to avoid circular imports at module level.
    from ive.db.models import TrainedModel as TM

    repo = ExperimentRepository(db_session, TM)  # type: ignore[arg-type]
    await repo.add_trained_model(
        experiment_id=experiment_id,
        model_type=model_type,
        fold_number=fold_number,
        # Use val_metric as a proxy for train_metric (no train re-prediction)
        train_metric=val_metric,
        val_metric=val_metric,
        metric_name=metric_name,
        artifact_path=artifact_path,
        hyperparams=hyperparams,
        feature_importances=feature_importances,
        training_time_seconds=training_time_seconds,
    )


async def _bulk_insert_residuals(
    *,
    db_session: AsyncSession,
    experiment_id: uuid.UUID,
    records: list[dict[str, Any]],
    batch_size: int = _RESIDUAL_BATCH_SIZE,
) -> int:
    """Bulk-insert residuals in caller-side batches to prevent memory spikes.

    Splits ``records`` into chunks of ``batch_size`` and calls
    :meth:`ExperimentRepository.add_residuals_batch` for each chunk.
    The repo itself further sub-chunks to respect PG parameter limits.

    Args:
        db_session:     Active async session.
        experiment_id:  Experiment UUID (injected into every row).
        records:        List of residual-record dicts from
                        :meth:`ResidualAnalyzer.build_residual_records` plus
                        an injected ``model_type`` key.
        batch_size:     Max records per call (default 5 000).

    Returns:
        Total number of rows successfully inserted.
    """
    from ive.db.models import Residual

    repo = ExperimentRepository(db_session, Residual)  # type: ignore[arg-type]
    total = 0

    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        inserted = await repo.add_residuals_batch(experiment_id, chunk)
        total += inserted
        log.debug(
            "ive.phase.model.residual_chunk_inserted",
            chunk_start=i,
            chunk_size=len(chunk),
            total_so_far=total,
        )

    return total
