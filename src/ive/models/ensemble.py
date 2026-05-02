"""Stacked ensemble (Phase B3).

Plan reference: §B3 + §95 + §96.

**Scope.** The ensemble exists for Phase 5 holdout-uplift measurement and
nothing else. Detection (Phase 3) continues to run against canonical
XGBoost residuals + canonical XGBoost SHAP — the residuals/attributions
must come from the same model, otherwise the "errors I see" and "features
I attribute them to" are misaligned.

**Design.**
- Inputs: a list of base-model OOF prediction arrays (one per model_type).
- Meta-learner: ``Ridge(alpha=1.0)`` for regression, ``LogisticRegression``
  for binary classification. Multiclass ensembling is out of scope for B3.
- Cross-fitted meta predictions: K-fold splits over the OOF columns
  prevent meta-leakage (a meta-fold's training never sees the row it
  later predicts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import structlog
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

log = structlog.get_logger(__name__)


@dataclass
class StackedEnsembleResult:
    """Outcome of fitting a stacked ensemble."""

    oof_predictions: np.ndarray[Any, Any]
    oof_residuals: np.ndarray[Any, Any]
    base_model_oof: dict[str, np.ndarray[Any, Any]]
    meta_learner_coefs: dict[str, float]
    blend_weights: dict[str, float]
    meta_kind: Literal["ridge", "logistic"]
    # The fitted meta-learner trained on the full finite OOF stack.
    # Used by :meth:`predict` so Phase 5 can score X_holdout via the
    # ensemble. Optional for the legacy API and tests that only need
    # OOF outputs.
    meta_learner: Any | None = None
    model_type_order: tuple[str, ...] = ()

    def predict(
        self,
        base_predictions: dict[str, np.ndarray[Any, Any]],
    ) -> np.ndarray[Any, Any]:
        """Predict on a held-out set given fresh base-model predictions.

        Args:
            base_predictions: Dict ``{model_type: predict(X_holdout)}``
                with one entry per base model used at fit time.

        Returns:
            1-D ensemble predictions. For regression, raw Ridge output;
            for binary classification, the positive-class probability.

        Raises:
            ValueError: When the input dict's keys don't match the
                base models the ensemble was fit against, or when the
                meta-learner wasn't persisted on this result.
        """
        if self.meta_learner is None:
            raise ValueError(
                "StackedEnsembleResult.predict: meta_learner is None. "
                "This result was constructed without a fitted meta-learner."
            )
        expected = set(self.base_model_oof.keys())
        provided = set(base_predictions.keys())
        if provided != expected:
            raise ValueError(
                f"StackedEnsembleResult.predict: base-model mismatch. "
                f"Expected {sorted(expected)}, got {sorted(provided)}."
            )

        order = self.model_type_order or tuple(sorted(expected))
        feature_matrix = np.column_stack([base_predictions[mt] for mt in order])
        if self.meta_kind == "ridge":
            return cast(
                np.ndarray[Any, Any],
                self.meta_learner.predict(feature_matrix),
            )
        # Logistic — positive-class probability.
        return cast(
            np.ndarray[Any, Any],
            self.meta_learner.predict_proba(feature_matrix)[:, 1],
        )


class StackedEnsemble:
    """Cross-fitted stacked ensemble over per-model OOF predictions.

    Construct with the per-model OOF arrays (already produced by
    ``CrossValidator``). The :meth:`fit` method computes meta-OOF
    predictions and aggregates them into a single ensemble residual
    array suitable for Phase 5 baseline comparison.
    """

    def __init__(
        self,
        base_oof_predictions: dict[str, np.ndarray[Any, Any]],
        problem_type: str,
        seed: int = 42,
        n_meta_splits: int = 5,
    ) -> None:
        if not base_oof_predictions:
            raise ValueError("StackedEnsemble requires at least one base model.")
        if problem_type == "multiclass":
            raise ValueError(
                "StackedEnsemble does not support multiclass; "
                "skip ensembling for multiclass experiments per plan §B3."
            )
        if problem_type not in ("regression", "binary"):
            raise ValueError(
                f"Unknown problem_type {problem_type!r}; expected regression or binary."
            )

        self._base_oof = base_oof_predictions
        self._model_types: list[str] = sorted(base_oof_predictions.keys())
        self._problem_type = problem_type
        self._seed = seed
        self._n_meta_splits = n_meta_splits

    def fit(self, y: np.ndarray[Any, Any]) -> StackedEnsembleResult:
        """Compute cross-fitted meta predictions and aggregate per-model OOF.

        For each split:
            1. Train the meta-learner on rows OUT of the split using the
               base OOF predictions as features.
            2. Predict on the held-out rows.
            3. Aggregate predictions back into a single OOF array.

        Returns:
            :class:`StackedEnsembleResult` with the ensemble OOF
            predictions, residuals (``y - oof_predictions``), per-base
            OOF arrays preserved, and meta-learner coefficients.
        """
        y_arr = np.asarray(y, dtype=float)
        n = len(y_arr)

        # Stack per-model OOF predictions into a (n, k) feature matrix.
        # The order of model_types is locked in __init__ for reproducibility.
        meta_features = np.column_stack(
            [self._base_oof[mt] for mt in self._model_types]
        )

        # Some rows may be NaN (TimeSeriesSplit head-chunk per Wave 1
        # audit). Mask them — meta predictions for those rows stay NaN.
        finite_mask = np.all(np.isfinite(meta_features), axis=1)
        if not finite_mask.any():
            raise ValueError(
                "StackedEnsemble.fit: no finite OOF predictions available."
            )

        oof_predictions = np.full(n, np.nan)

        if self._problem_type == "regression":
            splitter: Any = KFold(
                n_splits=min(self._n_meta_splits, max(2, finite_mask.sum() // 2)),
                shuffle=True,
                random_state=self._seed,
            )
            meta_kind: Literal["ridge", "logistic"] = "ridge"
        else:
            # Binary — stratify over the finite rows.
            splitter = StratifiedKFold(
                n_splits=min(self._n_meta_splits, max(2, finite_mask.sum() // 2)),
                shuffle=True,
                random_state=self._seed,
            )
            meta_kind = "logistic"

        finite_idx = np.where(finite_mask)[0]
        X_finite = meta_features[finite_idx]
        y_finite = y_arr[finite_idx]

        # Track coefficients from the final fit on the full finite set
        # (used for blend-weight reporting). The cross-fit loop below
        # produces the leak-free OOF predictions.
        for tr_local, va_local in splitter.split(X_finite, y_finite):
            tr_global = finite_idx[tr_local]
            va_global = finite_idx[va_local]
            X_tr, y_tr = meta_features[tr_global], y_arr[tr_global]
            X_va = meta_features[va_global]
            if meta_kind == "ridge":
                meta = Ridge(alpha=1.0, random_state=self._seed)
                meta.fit(X_tr, y_tr)
                oof_predictions[va_global] = meta.predict(X_va)
            else:
                meta = LogisticRegression(
                    C=1.0,
                    random_state=self._seed,
                    max_iter=1000,
                )
                meta.fit(X_tr, y_tr)
                oof_predictions[va_global] = meta.predict_proba(X_va)[:, 1]

        # Final fit on all finite rows for reportable coefficients.
        if meta_kind == "ridge":
            final = Ridge(alpha=1.0, random_state=self._seed)
            final.fit(X_finite, y_finite)
            coefs = final.coef_
        else:
            final = LogisticRegression(C=1.0, random_state=self._seed, max_iter=1000)
            final.fit(X_finite, y_finite)
            coefs = final.coef_.ravel()

        meta_learner_coefs = {
            mt: float(coefs[i]) for i, mt in enumerate(self._model_types)
        }
        # Normalize the coefficients to a [0, 1] blend weight by absolute
        # magnitude so the operator can read it as "this base model
        # contributed N% of the meta signal." Degenerate cases — single
        # base model, perfectly correlated bases, constant target — make
        # all coefs zero; in that case fall back to uniform weights so
        # the dict still sums to 1.0 (Wave 2 audit fix).
        abs_coefs = np.abs(coefs)
        total = float(abs_coefs.sum())
        n_models = len(self._model_types)
        if total > 0:
            blend_weights = {
                mt: float(abs_coefs[i] / total)
                for i, mt in enumerate(self._model_types)
            }
        else:
            uniform = 1.0 / n_models
            blend_weights = {mt: uniform for mt in self._model_types}

        if self._problem_type == "regression":
            oof_residuals = y_arr - oof_predictions
        else:
            # Binary — use raw probability gap as the residual surrogate.
            # For Phase 5 uplift the metric (log-loss) cares about the
            # probability itself, not a deviance residual.
            oof_residuals = y_arr - oof_predictions

        log.info(
            "ive.ensemble.complete",
            problem_type=self._problem_type,
            meta_kind=meta_kind,
            n_finite=int(finite_mask.sum()),
            n_total=n,
            blend_weights=blend_weights,
        )
        return StackedEnsembleResult(
            meta_learner=final,
            model_type_order=tuple(self._model_types),
            oof_predictions=oof_predictions,
            oof_residuals=oof_residuals,
            base_model_oof=dict(self._base_oof),
            meta_learner_coefs=meta_learner_coefs,
            blend_weights=blend_weights,
            meta_kind=meta_kind,
        )


__all__ = ["StackedEnsemble", "StackedEnsembleResult"]
