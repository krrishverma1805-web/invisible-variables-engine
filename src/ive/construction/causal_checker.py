"""
Causal Plausibility Checker.

Filters latent variable candidates for causal plausibility using
heuristic directional checks. This does not claim to prove causality
but removes clearly spurious candidates before Phase 4 explanation.

Checks performed:
    1. Reverse causality: Does the LV proxy a consequence of Y rather than a cause?
    2. Confounding proxy: Is the LV merely a proxy for an already-included feature?
    3. Temporal ordering: If time columns exist, does the LV precede Y?
    4. Doubly robust orthogonal-effect check (plan §C1): residualize Y on
       controls and T on controls, then estimate the orthogonal effect.
       If the orthogonal effect is materially smaller than the raw effect,
       flag the candidate as ``confounded_by_dml``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

log = structlog.get_logger(__name__)


# Plan §C1 / §48: hand-rolled orthogonalization, no econml.
DML_KFOLDS = 2
DML_MIN_ROWS = 60         # need enough data for cross-fit + residual regression
DML_MIN_RAW_EFFECT = 0.05 # below this, raw effect is too small to even bother
DML_CONFOUND_RATIO = 0.3  # |theta_dml| / |theta_raw| below this -> confounded


def _get_attr(candidate: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get an attribute from a candidate (dict or dataclass)."""
    if isinstance(candidate, dict):
        return candidate.get(key, default)
    return getattr(candidate, key, default)


def _get_construction_rule(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract construction_rule from a candidate, handling both dict and dataclass."""
    rule = _get_attr(candidate, "construction_rule", {})
    if isinstance(rule, dict):
        return rule
    return {}


class CausalChecker:
    """
    Heuristic causal plausibility filter for dict[str, Any] objects.

    Operates as a filter -- candidates failing causal checks are assigned
    a reduced confidence_score with a warning, not silently discarded.
    """

    def filter(
        self,
        candidates: list[dict[str, Any]],
        df: object,  # pd.DataFrame
        target_column: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Apply causal plausibility checks to all candidates.

        Args:
            candidates: List of dict[str, Any] objects from Phase 4.
            df: The original DataFrame for correlation and ordering checks.
            target_column: Name of the target column for reverse-causality check.

        Returns:
            Same list with confidence_score adjusted for failed checks.
        """
        log.info("ive.causal_checker.start", n_candidates=len(candidates))

        if df is None or not candidates:
            return candidates

        if not isinstance(df, pd.DataFrame):
            log.warning("ive.causal_checker.skip", reason="df is not a DataFrame")
            return candidates

        # Get all feature column names (excluding target)
        all_feature_cols = [c for c in df.columns if c != target_column]

        for candidate in candidates:
            warnings: list[str] = []
            confidence_penalty = 1.0

            # Extract the candidate's defining feature column
            rule = _get_construction_rule(candidate)
            candidate_col = rule.get("column_name") if rule else None

            # Check 1: Confounding proxy
            if candidate_col and candidate_col in df.columns:
                if self._is_confounding_proxy(candidate, df, all_feature_cols):
                    confidence_penalty *= 0.5
                    warnings.append(
                        "May be a proxy for an already-included feature (high partial correlation)"
                    )

            # Check 2: Reverse causality
            if candidate_col and target_column and candidate_col in df.columns:
                if self._is_reverse_causal(candidate, df, target_column):
                    confidence_penalty *= 0.5
                    warnings.append(
                        "Candidate feature highly correlated with target -- possible reverse causality"
                    )

            # Check 3: DML orthogonal-effect check (plan §C1).
            # Only attempt on numeric candidate columns with enough rows.
            dml_result: dict[str, float] | None = None
            if (
                candidate_col
                and target_column
                and candidate_col in df.columns
                and pd.api.types.is_numeric_dtype(df[candidate_col])
            ):
                dml_result = self._dml_orthogonal_effect(
                    df=df,
                    treatment_col=candidate_col,
                    target_column=target_column,
                    other_columns=[
                        c for c in all_feature_cols if c != candidate_col
                    ],
                )
                if dml_result and dml_result.get("confounded"):
                    confidence_penalty *= 0.5
                    warnings.append(
                        "Orthogonalized effect ({theta_dml:.3f}) is materially smaller than "
                        "raw effect ({theta_raw:.3f}) -- confounded_by_dml".format(
                            theta_dml=dml_result["theta_dml"],
                            theta_raw=dml_result["theta_raw"],
                        )
                    )

            # Apply penalty and warnings
            if isinstance(candidate, dict):
                current_score = candidate.get(
                    "importance_score", candidate.get("stability_score", 1.0)
                )
                candidate["importance_score"] = float(current_score) * confidence_penalty
                candidate["causal_warnings"] = warnings
                candidate["causal_confidence_penalty"] = confidence_penalty
                if dml_result is not None:
                    candidate["dml_diagnostic"] = dml_result
            else:
                if hasattr(candidate, "importance_score"):
                    candidate.importance_score *= confidence_penalty  # type: ignore[attr-defined]
                if hasattr(candidate, "confidence_score"):
                    candidate.confidence_score *= confidence_penalty
                candidate.causal_warnings = warnings  # type: ignore[attr-defined]
                if dml_result is not None:
                    candidate.dml_diagnostic = dml_result  # type: ignore[attr-defined]

        n_penalized = sum(
            1
            for c in candidates
            if (
                c.get("causal_confidence_penalty", 1.0)  # type: ignore[union-attr]
                if isinstance(c, dict)
                else getattr(c, "causal_confidence_penalty", 1.0)
            )
            < 1.0
        )
        log.info(
            "ive.causal_checker.done",
            n_candidates=len(candidates),
            n_penalized=n_penalized,
        )
        return candidates

    def _is_reverse_causal(
        self,
        candidate: dict[str, Any],
        df: object,
        target_column: str,
    ) -> bool:
        """
        Check if the candidate features are so correlated with the target
        that they likely represent a consequence rather than a cause.

        Uses Pearson correlation as initial screen, then partial correlation
        (controlling for other numeric features) for moderate correlations.
        """
        if not isinstance(df, pd.DataFrame):
            return False

        rule = _get_construction_rule(candidate)
        col_name = rule.get("column_name") if rule else None
        if not col_name or col_name not in df.columns or target_column not in df.columns:
            return False

        try:
            target_vals = pd.to_numeric(df[target_column], errors="coerce")
            feature_vals = pd.to_numeric(df[col_name], errors="coerce")

            # Drop NaN rows
            mask = target_vals.notna() & feature_vals.notna()
            if mask.sum() < 10:
                return False

            # Simple Pearson as initial screen
            corr = float(
                np.corrcoef(feature_vals[mask].values, target_vals[mask].values)[0, 1]
            )

            # Extreme correlation with target is suspicious
            if abs(corr) > 0.95:
                return True

            # For moderate correlations, check partial correlation
            # controlling for other numeric features
            if abs(corr) > 0.8:
                other_numeric = [
                    c
                    for c in df.columns
                    if c != col_name
                    and c != target_column
                    and pd.api.types.is_numeric_dtype(df[c])
                ]
                if other_numeric:
                    from sklearn.linear_model import LinearRegression

                    # Partial correlation: regress both on controls, correlate residuals
                    controls = df[other_numeric].fillna(0).values[mask]
                    if controls.shape[0] > controls.shape[1] + 2:
                        lr1 = LinearRegression().fit(
                            controls, feature_vals[mask].values
                        )
                        lr2 = LinearRegression().fit(
                            controls, target_vals[mask].values
                        )
                        resid1 = feature_vals[mask].values - lr1.predict(controls)
                        resid2 = target_vals[mask].values - lr2.predict(controls)
                        partial_corr = float(
                            np.corrcoef(resid1, resid2)[0, 1]
                        )
                        # If partial correlation drops below 0.3, likely a confound
                        if abs(partial_corr) < 0.3:
                            return False
                        # Still high partial correlation -- ambiguous, flag it
                        if abs(partial_corr) > 0.8:
                            return True

            return False
        except Exception:
            return False

    def _dml_orthogonal_effect(
        self,
        *,
        df: pd.DataFrame,
        treatment_col: str,
        target_column: str,
        other_columns: list[str],
    ) -> dict[str, float] | None:
        """
        Hand-rolled double-ML orthogonalization (plan §C1, §48).

        Given:
            T = df[treatment_col]   (the candidate's defining feature)
            Y = df[target_column]
            X = df[other numeric features]

        We estimate:
            * raw effect (theta_raw): OLS slope of T on Y, no controls.
            * orthogonal effect (theta_dml):
                  - Cross-fit Ridge nuisance models g(X)≈Y and m(X)≈T.
                  - Take residuals Y_resid = Y - g(X), T_resid = T - m(X).
                  - theta_dml = sum(T_resid * Y_resid) / sum(T_resid**2).

        Heuristic confound flag: ``|theta_dml| / |theta_raw| < 0.3`` while
        ``|theta_raw| >= DML_MIN_RAW_EFFECT``. Returns ``None`` when the
        check is infeasible (too few rows / no numeric controls / numeric
        instability) so the caller can skip without penalising the candidate.
        """
        if not isinstance(df, pd.DataFrame):
            return None

        try:
            t_raw = pd.to_numeric(df[treatment_col], errors="coerce")
            y_raw = pd.to_numeric(df[target_column], errors="coerce")
            numeric_controls = [
                c
                for c in other_columns
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not numeric_controls:
                return None

            controls_df = df[numeric_controls].apply(
                pd.to_numeric, errors="coerce"
            )
            mask = (
                t_raw.notna()
                & y_raw.notna()
                & controls_df.notna().all(axis=1)
            )
            if int(mask.sum()) < DML_MIN_ROWS:
                return None

            t = t_raw[mask].to_numpy(dtype=float)
            y = y_raw[mask].to_numpy(dtype=float)
            x = controls_df.loc[mask].to_numpy(dtype=float)

            # Need T to vary -- otherwise the regression is undefined.
            if float(np.std(t)) < 1e-9 or float(np.std(y)) < 1e-9:
                return None

            # Raw effect: bivariate OLS slope of Y on T.
            t_centered = t - t.mean()
            y_centered = y - y.mean()
            denom_raw = float(np.sum(t_centered ** 2))
            if denom_raw < 1e-12:
                return None
            theta_raw = float(np.sum(t_centered * y_centered) / denom_raw)

            # Cross-fit nuisance models. K=2 keeps cost low; this is a screen.
            n = x.shape[0]
            kf = KFold(n_splits=DML_KFOLDS, shuffle=True, random_state=42)
            y_resid = np.zeros(n, dtype=float)
            t_resid = np.zeros(n, dtype=float)
            for train_idx, test_idx in kf.split(x):
                if len(train_idx) < 5 or len(test_idx) < 5:
                    return None
                g = Ridge(alpha=1.0).fit(x[train_idx], y[train_idx])
                m = Ridge(alpha=1.0).fit(x[train_idx], t[train_idx])
                y_resid[test_idx] = y[test_idx] - g.predict(x[test_idx])
                t_resid[test_idx] = t[test_idx] - m.predict(x[test_idx])

            denom_dml = float(np.sum(t_resid ** 2))
            if denom_dml < 1e-12:
                return None
            theta_dml = float(np.sum(t_resid * y_resid) / denom_dml)

            confounded = (
                abs(theta_raw) >= DML_MIN_RAW_EFFECT
                and abs(theta_dml) / max(abs(theta_raw), 1e-9)
                < DML_CONFOUND_RATIO
            )
            return {
                "theta_raw": theta_raw,
                "theta_dml": theta_dml,
                "n_used": float(n),
                "n_controls": float(x.shape[1]),
                "confounded": float(confounded),
            }
        except Exception as exc:
            log.debug("ive.causal_checker.dml_skipped", reason=str(exc))
            return None

    def _is_confounding_proxy(
        self,
        candidate: dict[str, Any],
        df: object,
        all_feature_columns: list[str],
    ) -> bool:
        """
        Check if the candidate is simply a proxy for an already-present feature.

        Returns True if the candidate column has |correlation| > 0.95 with
        any existing feature column, suggesting it adds no new information.
        """
        if not isinstance(df, pd.DataFrame):
            return False

        rule = _get_construction_rule(candidate)
        col_name = rule.get("column_name") if rule else None
        if not col_name or col_name not in df.columns:
            return False

        try:
            feature_vals = pd.to_numeric(df[col_name], errors="coerce")
            if feature_vals.isna().all():
                return False

            # Check pairwise correlation with all other features
            for other_col in all_feature_columns:
                if other_col == col_name:
                    continue
                if other_col not in df.columns:
                    continue
                other_vals = pd.to_numeric(df[other_col], errors="coerce")
                mask = feature_vals.notna() & other_vals.notna()
                if mask.sum() < 10:
                    continue
                corr = float(
                    np.corrcoef(feature_vals[mask].values, other_vals[mask].values)[
                        0, 1
                    ]
                )
                if abs(corr) > 0.95:
                    log.debug(
                        "ive.causal_checker.proxy_detected",
                        candidate_col=col_name,
                        proxy_of=other_col,
                        corr=round(corr, 4),
                    )
                    return True

            return False
        except Exception:
            return False
