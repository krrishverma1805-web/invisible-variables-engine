"""
Cross-Validator — Invisible Variables Engine.

Performs K-fold cross-validation on IVEModel instances to produce
out-of-fold (OOF) predictions and residuals.

Using OOF predictions rather than in-sample predictions is critical:
it ensures residuals reflect generalisation error, not overfitting,
which makes pattern detection in Phase 3 meaningful.

Architecture
------------
    1. Accept *any* ``IVEModel`` implementation (Linear, XGBoost, …).
    2. Split data with ``KFold`` (regression) or ``StratifiedKFold``
       (classification) — never leak validation rows into training.
    3. Clone the model freshly for each fold via ``copy.deepcopy``.
    4. Collect OOF predictions, per-fold scores, fitted models,
       feature importances, and fold assignments for downstream
       residual analysis.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from sklearn.metrics import (
    accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from ive.models.base_model import IVEModel

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CVResult:
    """Results from a cross-validation run.

    Attributes:
        model_name:          Canonical name of the model type.
        n_splits:            Number of CV folds.
        oof_predictions:     One prediction per sample, generated when that
                             sample was in the held-out fold.
        oof_residuals:       ``y - oof_predictions`` (regression).
        fold_assignments:    ``fold_assignments[i]`` = the fold index where
                             sample ``i`` was in the validation set.
        fold_scores:         Performance score per fold (R² or AUC).
        mean_score:          Mean of ``fold_scores``.
        std_score:           Std of ``fold_scores``.
        fitted_models:       One fitted model per fold.
        feature_importances: One importance dict per fold.
    """

    model_name: str
    n_splits: int
    oof_predictions: np.ndarray[Any, Any]
    oof_residuals: np.ndarray[Any, Any]
    fold_assignments: np.ndarray[Any, Any]  # int array, length = n_samples
    fold_scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    fitted_models: list[Any] = field(default_factory=list)
    feature_importances: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cross-validator
# ---------------------------------------------------------------------------


class CrossValidator:
    """K-Fold cross-validator for IVEModel instances.

    Produces out-of-fold predictions that are unbiased estimates of
    generalisation performance — critical for reliable residual analysis.

    Usage::

        cv = CrossValidator(LinearIVEModel(), n_splits=5, stratified=False)
        result = cv.fit(X, y)
        print(result.mean_score, result.oof_residuals.std())
    """

    def __init__(
        self,
        model: IVEModel,
        n_splits: int = 5,
        seed: int = 42,
        stratified: bool = False,
    ) -> None:
        """Initialise the cross-validator.

        Args:
            model:     An unfitted ``IVEModel`` instance.  A deep-copy is
                       created for each fold so the original stays untouched.
            n_splits:  Number of CV folds (default 5).
            seed:      Random seed for fold assignment reproducibility.
            stratified: ``True`` → ``StratifiedKFold`` (classification);
                        ``False`` → ``KFold`` (regression).
        """
        self.model = model
        self.n_splits = n_splits
        self.seed = seed
        self.stratified = stratified

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> CVResult:
        """Run K-fold cross-validation and return OOF predictions + metrics.

        For each fold this method:

        1. Splits ``X`` / ``y`` into train and validation sets.
        2. Deep-copies the template model for a clean instance.
        3. Fits the fold model on the training split.
        4. Generates predictions on the held-out validation split.
        5. Computes a per-fold performance score.
        6. Stores the fitted model and its feature importances.

        After all folds, the OOF predictions cover every sample exactly
        once.  Residuals are computed as ``y − oof_predictions``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
               May be a NumPy ndarray or a 2-D array-like.
            y: Target array of shape ``(n_samples,)``.

        Returns:
            A fully-populated :class:`CVResult`.
        """
        n_samples = X.shape[0]

        log.info(
            "ive.cross_validator.start",
            model=self.model.model_name,
            n_splits=self.n_splits,
            n_samples=n_samples,
            n_features=X.shape[1],
            stratified=self.stratified,
        )

        # ── Choose splitter ───────────────────────────────────────────
        if self.stratified:
            splitter: KFold | StratifiedKFold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            )
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            )

        # ── Accumulators ──────────────────────────────────────────────
        oof_predictions = np.full(n_samples, np.nan)
        fold_assignments = np.full(n_samples, -1, dtype=int)
        fold_scores: list[float] = []
        fitted_models: list[Any] = []
        feature_importances: list[dict[str, float]] = []

        # ── Fold loop ─────────────────────────────────────────────────
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fresh model clone per fold to prevent data leakage
            fold_model = copy.deepcopy(self.model)
            fold_model.fit(X_train, y_train)

            preds = fold_model.predict(X_val)

            # Guard against NaN predictions
            if np.any(np.isnan(preds)):
                nan_count = int(np.isnan(preds).sum())
                log.warning(
                    "ive.cross_validator.nan_predictions",
                    fold=fold_idx,
                    nan_count=nan_count,
                )
                preds = np.nan_to_num(preds, nan=float(np.nanmean(preds)))

            oof_predictions[val_idx] = preds
            fold_assignments[val_idx] = fold_idx

            # ── Per-fold scoring ──────────────────────────────────────
            score = self._compute_fold_score(y_val, preds)
            fold_scores.append(score)

            fitted_models.append(fold_model)
            feature_importances.append(fold_model.get_feature_importance())

            log.debug(
                "ive.cross_validator.fold_done",
                fold=fold_idx,
                val_size=len(val_idx),
                score=round(score, 4),
            )

        # ── Aggregate ─────────────────────────────────────────────────
        residuals = y - oof_predictions
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))

        log.info(
            "ive.cross_validator.complete",
            model=self.model.model_name,
            mean_score=round(mean_score, 4),
            std_score=round(std_score, 4),
            residual_std=round(float(np.std(residuals)), 4),
        )

        return CVResult(
            model_name=self.model.model_name,
            n_splits=self.n_splits,
            oof_predictions=oof_predictions,
            oof_residuals=residuals,
            fold_assignments=fold_assignments,
            fold_scores=fold_scores,
            mean_score=mean_score,
            std_score=std_score,
            fitted_models=fitted_models,
            feature_importances=feature_importances,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_fold_score(
        self,
        y_val: np.ndarray,
        preds: np.ndarray,
    ) -> float:
        """Compute a performance metric for a single fold.

        For classification (``self.stratified=True``):
            - Binary targets → ROC-AUC.
            - Multi-class targets → accuracy (ROC-AUC requires
              probability outputs not available from all models).

        For regression (``self.stratified=False``):
            - R² (coefficient of determination).

        Args:
            y_val: True targets for the validation fold.
            preds: Model predictions for the validation fold.

        Returns:
            A single float score.  Higher is always better.
        """
        try:
            if self.stratified:
                # Classification
                n_classes = len(np.unique(y_val))
                if n_classes == 2:
                    return float(roc_auc_score(y_val, preds))
                return float(accuracy_score(y_val, np.round(preds)))
            else:
                # Regression
                return float(r2_score(y_val, preds))
        except (ValueError, TypeError) as exc:
            log.warning(
                "ive.cross_validator.scoring_error",
                error=str(exc),
            )
            return 0.0
