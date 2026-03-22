"""
Residual Analyzer — Invisible Variables Engine.

Analyzes the distribution and structure of model residuals to characterise
the nature of model errors.  This informs Phase 3 detection strategies.

Two main responsibilities:

    1. **Statistical characterisation** — mean, std, skew, kurtosis,
       normality (Shapiro-Wilk), heteroscedasticity (Breusch-Pagan),
       and autocorrelation (Durbin-Watson).

    2. **Structured residual records** — convert raw OOF outputs into
       a ``list[dict]`` ready for bulk database insertion, with full
       per-sample metadata (index, fold, actuals, predictions, residuals,
       feature vectors).

All computations are vectorised.  No row-by-row loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHAPIRO_SAMPLE_LIMIT = 5_000  # Shapiro-Wilk is O(n²); sample for speed
_NORMALITY_ALPHA = 0.05
_HETEROSCEDASTICITY_ALPHA = 0.05
_LARGE_RESIDUAL_SIGMA = 2.0  # |residual| > 2σ is "large"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ResidualAnalysis:
    """Container for the outputs of residual analysis.

    Attributes:
        mean:               Mean of residuals (should be ≈ 0 for unbiased model).
        std:                Standard deviation of residuals.
        skewness:           Fisher skewness.
        kurtosis:           Fisher (excess) kurtosis.
        max_abs:            Maximum |residual| observed.
        pct_large:          Percentage of residuals exceeding ``2 × std``.
        heteroscedastic:    ``True`` if Breusch-Pagan p < 0.05.
        breusch_pagan_p:    Breusch-Pagan p-value (``None`` if X not provided).
        normal:             ``True`` if Shapiro-Wilk p > 0.05.
        shapiro_p:          Shapiro-Wilk p-value.
        durbin_watson:      Durbin-Watson statistic (≈2 → no autocorrelation).
        warnings:           Human-readable quality warnings.
    """

    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    max_abs: float = 0.0
    pct_large: float = 0.0
    heteroscedastic: bool = False
    breusch_pagan_p: float | None = None
    normal: bool = False
    shapiro_p: float | None = None
    durbin_watson: float | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ResidualAnalyzer:
    """Characterises model residual distributions.

    Used after cross-validation to understand *what kind* of errors
    remain, which guides which detection strategies to prioritise.

    Usage::

        analyzer = ResidualAnalyzer()
        stats = analyzer.analyze(residuals, X=X_features)
        records = analyzer.build_residual_records(
            X_df=df_features,
            y_true=y,
            oof_predictions=cv_result.oof_predictions,
            fold_assignments=cv_result.fold_assignments,
        )
    """

    def __init__(self, large_residual_threshold: float = _LARGE_RESIDUAL_SIGMA) -> None:
        """Initialise the analyzer.

        Args:
            large_residual_threshold: Number of standard deviations beyond
                which a residual is considered "large".
        """
        self.threshold = large_residual_threshold

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        residuals: np.ndarray[Any, Any],
        X: np.ndarray[Any, Any] | None = None,
    ) -> ResidualAnalysis:
        """Perform full residual analysis with statistical tests.

        Tests applied:

        - **Shapiro-Wilk** (normality) on a random sample of ≤ 5 000
          values.  A *non-normal* residual distribution often signals a
          missing structural variable.
        - **Breusch-Pagan** (heteroscedasticity) when ``X`` is provided.
          Heteroscedastic residuals mean the error variance depends on
          feature values — a strong hint that a latent interaction exists.
        - **Durbin-Watson** (autocorrelation).  Values far from 2 suggest
          temporal or sequential dependencies not captured by the model.

        Args:
            residuals: 1-D array of residual values (``y_true - y_pred``).
            X:         Optional feature matrix for heteroscedasticity tests.

        Returns:
            Fully-populated :class:`ResidualAnalysis`.
        """
        from scipy import stats as sp_stats

        analysis = ResidualAnalysis()

        if len(residuals) == 0:
            log.warning("ive.residual_analyzer.empty_residuals")
            return analysis

        # ── Basic statistics ──────────────────────────────────────────
        analysis.mean = float(np.mean(residuals))
        analysis.std = float(np.std(residuals))
        analysis.skewness = float(sp_stats.skew(residuals))
        analysis.kurtosis = float(sp_stats.kurtosis(residuals))
        analysis.max_abs = float(np.max(np.abs(residuals)))

        if analysis.std > 0:
            analysis.pct_large = float(
                np.mean(np.abs(residuals) > self.threshold * analysis.std) * 100
            )
        else:
            analysis.pct_large = 0.0

        # ── Normality test (Shapiro-Wilk) ─────────────────────────────
        try:
            sample = residuals
            if len(residuals) > _SHAPIRO_SAMPLE_LIMIT:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(residuals), _SHAPIRO_SAMPLE_LIMIT, replace=False)
                sample = residuals[idx]
            _, p_shapiro = sp_stats.shapiro(sample)
            analysis.shapiro_p = float(p_shapiro)
            analysis.normal = p_shapiro > _NORMALITY_ALPHA
        except Exception as exc:
            log.warning("ive.residual_analyzer.shapiro_failed", error=str(exc))

        # ── Heteroscedasticity test (Breusch-Pagan) ───────────────────
        if X is not None and len(residuals) == X.shape[0]:
            try:
                import statsmodels.api as sm
                from statsmodels.stats.diagnostic import het_breuschpagan

                # OLS of residuals² on X (with constant) to test variance
                X_const = sm.add_constant(X, has_constant="add")
                _, p_bp, _, _ = het_breuschpagan(residuals, X_const)
                analysis.breusch_pagan_p = float(p_bp)
                analysis.heteroscedastic = p_bp < _HETEROSCEDASTICITY_ALPHA
            except Exception as exc:
                log.warning("ive.residual_analyzer.breusch_pagan_failed", error=str(exc))

        # ── Autocorrelation (Durbin-Watson) ───────────────────────────
        try:
            from statsmodels.stats.stattools import durbin_watson

            analysis.durbin_watson = float(durbin_watson(residuals))
        except Exception as exc:
            log.warning("ive.residual_analyzer.durbin_watson_failed", error=str(exc))

        # ── Warnings ──────────────────────────────────────────────────
        if analysis.pct_large > 10:
            analysis.warnings.append(
                f"{analysis.pct_large:.1f}% of residuals exceed {self.threshold}σ — "
                "strong evidence of systematic error."
            )

        if analysis.heteroscedastic:
            analysis.warnings.append(
                f"Breusch-Pagan p={analysis.breusch_pagan_p:.4f} — "
                "residual variance depends on features (heteroscedastic). "
                "A latent interaction or regime split is likely."
            )

        if not analysis.normal and analysis.shapiro_p is not None:
            analysis.warnings.append(
                f"Shapiro-Wilk p={analysis.shapiro_p:.4f} — "
                "residuals are non-normal. A missing structural "
                "variable may explain the non-Gaussian error shape."
            )

        if analysis.durbin_watson is not None:
            dw = analysis.durbin_watson
            if dw < 1.5 or dw > 2.5:
                analysis.warnings.append(
                    f"Durbin-Watson={dw:.3f} (expected ≈ 2.0) — "
                    "residuals show autocorrelation. Consider a "
                    "temporal or sequential hidden variable."
                )

        log.info(
            "ive.residual_analyzer.done",
            mean=round(analysis.mean, 4),
            std=round(analysis.std, 4),
            skewness=round(analysis.skewness, 4),
            pct_large=round(analysis.pct_large, 2),
            normal=analysis.normal,
            heteroscedastic=analysis.heteroscedastic,
            n_warnings=len(analysis.warnings),
        )

        return analysis

    # ------------------------------------------------------------------
    # Structured residual records (ready for DB bulk insert)
    # ------------------------------------------------------------------

    def build_residual_records(
        self,
        X_df: pd.DataFrame,
        y_true: np.ndarray | pd.Series,
        oof_predictions: np.ndarray,
        fold_assignments: np.ndarray,
        task_type: str = "regression",
    ) -> list[dict[str, Any]]:
        """Build structured per-sample residual records for database storage.

        Each record contains the sample index, fold number, actual value,
        predicted value, residual, absolute residual, and the original
        feature vector as a dict — everything the pattern-detection
        engine (Phase 3) needs for subgroup discovery.

        Residual calculation:
            - **Regression**: ``residual = y_true − y_pred``
            - **Classification** (binary probability margin):
              ``residual = y_true − y_pred_proba``
              E.g. actual = 1, predicted prob = 0.2 → residual = 0.8
              (the model was very wrong about this sample).

        Args:
            X_df:             Feature DataFrame with the original index.
            y_true:           True targets, aligned with ``X_df``.
            oof_predictions:  Out-of-fold predictions from ``CVResult``.
            fold_assignments: Fold-index array from ``CVResult``.
            task_type:        ``"regression"`` or ``"classification"``.

        Returns:
            A list of dicts, one per sample, ready for bulk DB insertion::

                [
                    {
                        "sample_index": 42,
                        "fold_number": 2,
                        "actual_value": 150_000.0,
                        "predicted_value": 142_320.5,
                        "residual_value": 7_679.5,
                        "abs_residual": 7_679.5,
                        "feature_vector": {"feature_a": 1.2, "category": "B", ...}
                    },
                    ...
                ]
        """
        n_samples = len(X_df)
        y_arr = np.asarray(y_true, dtype=float)
        preds = np.asarray(oof_predictions, dtype=float)
        folds = np.asarray(fold_assignments, dtype=int)

        # Sanity checks — all arrays must be the same length
        if not (n_samples == len(y_arr) == len(preds) == len(folds)):
            raise ValueError(
                f"Length mismatch: X_df={n_samples}, y_true={len(y_arr)}, "
                f"oof_predictions={len(preds)}, fold_assignments={len(folds)}."
            )

        # ── Compute residuals (vectorised) ────────────────────────────
        if task_type == "classification":
            # For binary classification, predictions are probabilities of
            # class 1.  The "margin" residual = actual − predicted_prob.
            residuals = y_arr - preds
        else:
            # Standard regression residual
            residuals = y_arr - preds

        abs_residuals = np.abs(residuals)

        # ── Build records ─────────────────────────────────────────────
        #
        # Convert feature rows to dicts once (efficient: avoids per-row
        # DataFrame operations) and then zip all arrays together.

        # Convert DataFrame rows to list-of-dicts in one vectorised call
        feature_records = X_df.to_dict(orient="records")
        original_indices = X_df.index.tolist()

        records: list[dict[str, Any]] = [
            {
                "sample_index": int(original_indices[i]),
                "fold_number": int(folds[i]),
                "actual_value": float(y_arr[i]),
                "predicted_value": float(preds[i]),
                "residual_value": float(residuals[i]),
                "abs_residual": float(abs_residuals[i]),
                "feature_vector": _sanitise_feature_dict(feature_records[i]),
            }
            for i in range(n_samples)
        ]

        log.info(
            "ive.residual_analyzer.records_built",
            n_records=len(records),
            task_type=task_type,
            mean_abs_residual=round(float(abs_residuals.mean()), 4),
        )

        return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitise_feature_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Ensure all values in a feature dict are JSON-serialisable.

    Converts numpy scalars and NaN/Inf to Python-native types so the
    dict can be stored directly as ``JSONB`` in PostgreSQL.

    Args:
        d: Raw dict from ``DataFrame.to_dict(orient="records")``.

    Returns:
        Cleaned dict with only JSON-safe Python types.
    """
    clean: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            if np.isnan(v) or np.isinf(v):
                clean[k] = None
            else:
                clean[k] = float(v)
        elif isinstance(v, np.bool_):
            clean[k] = bool(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return clean
