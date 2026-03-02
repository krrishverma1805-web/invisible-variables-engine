"""
Cross-Validator.

Performs K-fold cross-validation on IVEModel instances to produce
out-of-fold (OOF) predictions and residuals.

Using OOF predictions rather than in-sample predictions is critical:
it ensures residuals reflect generalisation error, not overfitting,
which makes pattern detection in Phase 3 meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from sklearn.model_selection import KFold, StratifiedKFold

from ive.models.base_model import IVEModel

log = structlog.get_logger(__name__)


@dataclass
class CVResult:
    """Results from a cross-validation run."""

    model_name: str
    n_splits: int
    oof_predictions: np.ndarray
    oof_residuals: np.ndarray
    fold_scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    fitted_models: list[Any] = field(default_factory=list)
    feature_importances: list[dict[str, float]] = field(default_factory=list)


class CrossValidator:
    """
    K-Fold cross-validator for IVEModel instances.

    Produces out-of-fold predictions that are unbiased estimates of
    generalisation performance — critical for reliable residual analysis.
    """

    def __init__(
        self,
        model: IVEModel,
        n_splits: int = 5,
        seed: int = 42,
        stratified: bool = False,
    ) -> None:
        """
        Args:
            model: A fitted or unfitted IVEModel instance.
            n_splits: Number of CV folds.
            seed: Random seed for fold assignment reproducibility.
            stratified: Use StratifiedKFold (for classification tasks).
        """
        self.model = model
        self.n_splits = n_splits
        self.seed = seed
        self.stratified = stratified

    def fit(self, X: np.ndarray, y: np.ndarray) -> CVResult:
        """
        Run K-fold cross-validation and return OOF predictions.

        For each fold:
            1. Split X, y into train/val
            2. Clone and fit the model on train
            3. Predict on val
            4. Accumulate OOF predictions

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target array (n_samples,).

        Returns:
            CVResult with oof_predictions, oof_residuals, and per-fold scores.

        TODO:
            - Use sklearn.base.clone(self.model._model) for clean fold models
            - Compute per-fold R² (regression) or ROC-AUC (classification)
            - Aggregate fold-level feature importances
            - Handle the case where a fold produces NaN predictions
        """
        import copy

        log.info(
            "ive.cross_validator.start",
            model=self.model.model_name,
            n_splits=self.n_splits,
            n_samples=X.shape[0],
        )

        if self.stratified:
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        oof_predictions = np.zeros(len(y))
        fold_scores: list[float] = []
        fitted_models: list[Any] = []
        feature_importances: list[dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Deep copy the model to get a fresh instance per fold
            fold_model = copy.deepcopy(self.model)
            fold_model.fit(X_train, y_train)

            preds = fold_model.predict(X_val)
            oof_predictions[val_idx] = preds

            # TODO: Compute fold score (R² for regression)
            # from sklearn.metrics import r2_score
            # score = r2_score(y_val, preds)
            score = 0.0
            fold_scores.append(score)
            fitted_models.append(fold_model)
            feature_importances.append(fold_model.get_feature_importance())

            log.debug("ive.cross_validator.fold_done", fold=fold_idx, score=score)

        residuals = y - oof_predictions

        return CVResult(
            model_name=self.model.model_name,
            n_splits=self.n_splits,
            oof_predictions=oof_predictions,
            oof_residuals=residuals,
            fold_scores=fold_scores,
            mean_score=float(np.mean(fold_scores)),
            std_score=float(np.std(fold_scores)),
            fitted_models=fitted_models,
            feature_importances=feature_importances,
        )
