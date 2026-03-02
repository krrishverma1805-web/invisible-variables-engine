"""
Phase 2 — Model.

Trains baseline ML models and extracts cross-validated residuals.

Steps:
    1. Preprocess features (encode categoricals, scale numerics)
    2. Train Linear (Ridge) and/or XGBoost models using K-fold CV
    3. Compute out-of-fold predictions and residuals
    4. Analyze residuals for heteroscedasticity and structure

Outputs written to PipelineContext:
    ctx.residuals         — numpy array of OOF residuals
    ctx.predictions       — numpy array of OOF predictions
    ctx.feature_matrix    — preprocessed feature array
    ctx.model_artifacts   — dict of trained model objects and metadata
"""

from __future__ import annotations

import time

import structlog

from ive.core.pipeline import PhaseBase, PhaseResult, PipelineContext

log = structlog.get_logger(__name__)


class PhaseModel(PhaseBase):
    """
    Phase 2: Train models and compute residuals.

    The key output is a residual array that captures *what the model
    doesn't know* — which Phase 3 will then analyse for structure.
    """

    def get_phase_name(self) -> str:
        return "model"

    async def execute(self, ctx: PipelineContext) -> PhaseResult:
        """
        Run cross-validated training and residual extraction.

        TODO:
            - Call DataPreprocessor.fit_transform(ctx.df, ctx.feature_columns)
              → store result as ctx.feature_matrix
            - For each model_type in ctx.config.model_types:
                  model = LinearIVEModel() or XGBoostIVEModel()
                  cv_result = CrossValidator(model, ctx.config.cv_folds).fit(X, y)
                  ctx.model_artifacts[model_type] = cv_result
            - Compute context-level OOF residuals by averaging or combining
              across model types
            - Call ResidualAnalyzer.analyze(ctx.residuals) → store metadata
        """
        start = time.monotonic()
        log.info("ive.phase.model.start", experiment_id=str(ctx.experiment_id))

        # TODO: Preprocess
        # from ive.data.preprocessor import DataPreprocessor
        # preprocessor = DataPreprocessor()
        # ctx.feature_matrix, _ = preprocessor.fit_transform(ctx.df, ctx.feature_columns)

        # TODO: Cross-validate each requested model type
        # from ive.models.cross_validator import CrossValidator
        # from ive.models.linear_model import LinearIVEModel
        # from ive.models.xgboost_model import XGBoostIVEModel
        # all_residuals = []
        # for model_type in ctx.config.model_types:
        #     model = LinearIVEModel() if model_type == "linear" else XGBoostIVEModel()
        #     cv = CrossValidator(model, n_splits=ctx.config.cv_folds, seed=ctx.config.random_seed)
        #     result = cv.fit(ctx.feature_matrix, ctx.target_series.values)
        #     all_residuals.append(result.residuals)
        #     ctx.model_artifacts[model_type] = result
        # ctx.residuals = np.mean(all_residuals, axis=0)

        # TODO: Analyze residuals
        # from ive.models.residual_analyzer import ResidualAnalyzer
        # ctx.profile["residual_analysis"] = ResidualAnalyzer().analyze(ctx.residuals)

        elapsed = time.monotonic() - start
        return PhaseResult(
            phase_name="model",
            success=True,
            duration_seconds=elapsed,
            metadata={"model_types": ctx.config.model_types, "cv_folds": ctx.config.cv_folds},
        )
