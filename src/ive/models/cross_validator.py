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
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

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
    importance_stability: dict[str, float] = field(default_factory=dict)


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
        cv_strategy: str = "auto",
        time_index: np.ndarray[Any, Any] | None = None,
        groups: np.ndarray[Any, Any] | None = None,
        gap: int = 0,
    ) -> None:
        """Initialise the cross-validator.

        Args:
            model:     An unfitted ``IVEModel`` instance.  A deep-copy is
                       created for each fold so the original stays untouched.
            n_splits:  Number of CV folds (default 5).
            seed:      Random seed for fold assignment reproducibility.
            stratified: **Deprecated** — use ``cv_strategy='stratified'`` instead.
                        Kept for one release for backward compat. When
                        ``cv_strategy='auto'`` and ``stratified=True``, the
                        constructor coerces to ``cv_strategy='stratified'``.
            cv_strategy: One of ``'auto' | 'kfold' | 'stratified' | 'timeseries' | 'group'``.
                         ``'auto'`` defers to ``_resolve_strategy()`` at fit time:
                         classification target → ``stratified``, ``time_index``
                         provided → ``timeseries``, ``groups`` provided →
                         ``group``, otherwise ``kfold``.  See plan §B2.
            time_index:  Per-row timestamp ranks (or any monotonic index) used
                         by ``TimeSeriesSplit``.  Must be passed when
                         ``cv_strategy='timeseries'`` or when auto-detection
                         should select that strategy.
            groups:      Per-row group labels used by ``GroupKFold``.  Required
                         when ``cv_strategy='group'``.
            gap:         Number of samples to drop between train and validation
                         in ``TimeSeriesSplit`` (purged-CV gap for forecasting
                         with autoregressive features). Default 0 — bump to at
                         least the max lag length when the dataset has lagged
                         features (per plan §27).
        """
        self.model = model
        self.n_splits = n_splits
        self.seed = seed
        # Backward-compat shim: stratified=True with auto strategy → stratified.
        if cv_strategy == "auto" and stratified:
            cv_strategy = "stratified"
        self.stratified = stratified  # Retained for legacy logging + scoring branches.
        self.cv_strategy = cv_strategy
        self.time_index = time_index
        self.groups = groups
        self.gap = gap

    def _resolve_strategy(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
    ) -> str:
        """Pick a CV strategy when ``cv_strategy='auto'``.

        Priority (per plan §B2):
            1. ``time_index`` is provided → ``'timeseries'``.
            2. ``groups`` is provided → ``'group'``.
            3. Target looks like binary/multiclass classification (per
               cardinality + dtype heuristic) → ``'stratified'``.
            4. Fallback → ``'kfold'``.
        """
        if self.cv_strategy != "auto":
            return self.cv_strategy
        if self.time_index is not None:
            return "timeseries"
        if self.groups is not None:
            return "group"
        # Classification heuristic — conservative per plan §B5: only when
        # all of (integer/bool dtype, nunique<=10, n/nunique>=30) hold.
        if y.dtype.kind in {"b", "i", "u"} and len(y) > 0:
            nunique = int(np.unique(y).shape[0])
            if 1 < nunique <= 10 and len(y) / nunique >= 30:
                return "stratified"
        return "kfold"

    def _build_splitter(
        self,
        X: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
    ) -> Any:
        """Construct the sklearn splitter for the resolved strategy."""
        strategy = self._resolve_strategy(X, y)
        if strategy == "stratified":
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.seed,
            )
        if strategy == "timeseries":
            if self.time_index is None:
                # Auto-resolved to timeseries without an index — fall back
                # to row order, which is the same thing modulo type.
                pass
            return TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        if strategy == "group":
            if self.groups is None:
                raise ValueError(
                    "cv_strategy='group' requires `groups` to be passed to CrossValidator()."
                )
            return GroupKFold(n_splits=self.n_splits)
        # kfold
        return KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.seed,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> CVResult:
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
        splitter = self._build_splitter(X, y)

        # ── Accumulators ──────────────────────────────────────────────
        oof_predictions = np.full(n_samples, np.nan)
        fold_assignments = np.full(n_samples, -1, dtype=int)
        fold_scores: list[float] = []
        fitted_models: list[Any] = []
        feature_importances: list[dict[str, float]] = []

        # ── Fold loop ─────────────────────────────────────────────────
        # For TimeSeriesSplit, splitter expects rows in chronological order;
        # we honor `time_index` by sorting once and remapping back at end.
        strategy = self._resolve_strategy(X, y)
        if strategy == "timeseries" and self.time_index is not None:
            order = np.argsort(self.time_index, kind="stable")
            X_sorted = X[order]
            y_sorted = y[order]
            split_iter = splitter.split(X_sorted, y_sorted)
            split_pairs = [(order[tr], order[va]) for tr, va in split_iter]
        elif strategy == "group" and self.groups is not None:
            split_iter = splitter.split(X, y, groups=self.groups)
            split_pairs = list(split_iter)
        else:
            split_iter = splitter.split(X, y)
            split_pairs = list(split_iter)

        for fold_idx, (train_idx, val_idx) in enumerate(split_pairs):
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
                fill_value = float(np.nanmean(preds))
                if np.isnan(fill_value):
                    raise RuntimeError(
                        f"Fold {fold_idx} produced all-NaN predictions. "
                        "The model failed to generate any valid outputs."
                    )
                preds = np.nan_to_num(preds, nan=fill_value)

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

        # ── Feature importance stability across folds ─────────────────
        importance_stability: dict[str, float] = {}
        if feature_importances and len(feature_importances) > 1:
            # Collect all feature names across all folds
            all_features: set[str] = set()
            for fi in feature_importances:
                all_features.update(fi.keys())

            for feat in all_features:
                # Get importance of this feature across all folds (0.0 if missing in a fold)
                values = [fi.get(feat, 0.0) for fi in feature_importances]
                arr = np.array(values)
                mean_imp = float(np.mean(arr))
                std_imp = float(np.std(arr))

                # Coefficient of variation (CV) — lower is more stable
                # Use 0.0 for features with zero mean importance (consistently unimportant)
                if mean_imp > 1e-10:
                    cv = std_imp / mean_imp
                else:
                    cv = 0.0
                importance_stability[feat] = round(cv, 4)

            # Log features with high instability (CV > 0.5)
            unstable = {k: v for k, v in importance_stability.items() if v > 0.5}
            if unstable:
                log.warning(
                    "ive.cross_validator.unstable_importances",
                    n_unstable=len(unstable),
                    top_unstable=dict(
                        sorted(unstable.items(), key=lambda x: x[1], reverse=True)[:5]
                    ),
                )

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
            importance_stability=importance_stability,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_fold_score(
        self,
        y_val: np.ndarray[Any, Any],
        preds: np.ndarray[Any, Any],
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
