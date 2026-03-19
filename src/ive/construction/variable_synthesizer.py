"""
Variable Synthesizer — Phase 4 Construction & Validation.

Converts raw detected patterns (subgroups, clusters) from Phase 3 into
concrete **latent variable candidates**.  Each candidate is a proposed
feature that encodes error-subgroup membership or geometric proximity
to a high-error cluster center.

Synthesis strategies
--------------------
* **Subgroup** pattern → binary indicator (1 if the row matches the
  subgroup rule, 0 otherwise).

  - *Numeric bins* (intervals like ``(10.0, 20.0]``) are stored with
    explicit ``lower`` / ``upper`` bounds and reconstructed via numeric
    range membership:  ``lower < X[col] <= upper``.
  - *Categorical values* continue to use string equality.

* **Cluster** pattern → continuous score ∈ (0, 1] via the inverse-distance
  kernel ``1 / (1 + d)`` where *d* is the Euclidean distance from the row's
  numeric features to the cluster center.

Output
------
A list of candidate dicts, each containing:

* ``name``              — human-readable label
* ``pattern_type``      — ``"subgroup"`` or ``"cluster"``
* ``construction_rule`` — dict describing how the variable was synthesised
* ``scores``            — ``np.ndarray`` of shape ``(n_samples,)``
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Regex for pandas-style interval strings:  "(10.0, 20.0]", "[-inf, 5.0)"
# ---------------------------------------------------------------------------
_INTERVAL_RE = re.compile(
    r"^"
    r"(?P<left_bracket>[\(\[])"  # opening ( or [
    r"\s*(?P<lower>[^,]+?)\s*"  # lower bound
    r","
    r"\s*(?P<upper>[^\)\]]+?)\s*"  # upper bound
    r"(?P<right_bracket>[\)\]])"  # closing ) or ]
    r"$"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VariableSynthesizer:
    """Convert Phase 3 pattern dicts into latent-variable score arrays.

    Each detected pattern is turned into a numeric feature vector over
    every row in *X*.  Subgroup patterns produce binary indicators;
    cluster patterns produce continuous proximity scores.
    """

    # ------------------------------------------------------------------
    # Core entry point
    # ------------------------------------------------------------------

    def synthesize(
        self,
        detected_patterns: list[dict[str, Any]],
        X: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """Synthesise latent-variable candidates from detected patterns.

        Args:
            detected_patterns: Pattern dicts emitted by
                :class:`SubgroupDiscovery` or :class:`HDBSCANClustering`.
            X: Original feature DataFrame (n_samples × n_features).

        Returns:
            List of candidate dicts, one per pattern.  Patterns with an
            unrecognised ``pattern_type`` are silently skipped.
        """
        log.info(
            "ive.synthesizer.start",
            n_patterns=len(detected_patterns),
            n_samples=len(X),
        )

        candidates: list[dict[str, Any]] = []

        for idx, pattern in enumerate(detected_patterns):
            ptype = pattern.get("pattern_type", "")

            if ptype == "subgroup":
                candidate = self._synthesize_subgroup(pattern, X, idx)
            elif ptype == "cluster":
                candidate = self._synthesize_cluster(pattern, X, idx)
            else:
                log.warning(
                    "ive.synthesizer.unknown_pattern_type",
                    pattern_type=ptype,
                    pattern_index=idx,
                )
                continue

            if candidate is not None:
                candidates.append(candidate)

        log.info("ive.synthesizer.done", n_candidates=len(candidates))
        return candidates

    # ------------------------------------------------------------------
    # Subgroup synthesis
    # ------------------------------------------------------------------

    def _synthesize_subgroup(
        self,
        pattern: dict[str, Any],
        X: pd.DataFrame,
        idx: int,
    ) -> dict[str, Any] | None:
        """Create a binary indicator from a subgroup pattern.

        For numeric bins (intervals like ``(10.0, 20.0]``), the rule
        stores explicit ``lower`` / ``upper`` bounds and uses numeric
        range membership so it survives bootstrap resampling.

        For categorical values, the rule uses string comparison.

        Args:
            pattern: Subgroup pattern dict containing ``column_name`` and
                     ``bin_value``.
            X:       Feature DataFrame.
            idx:     Pattern index (used for name generation).

        Returns:
            Candidate dict, or ``None`` if the column is missing from *X*.
        """
        col = pattern.get("column_name", "")
        bin_value = str(pattern.get("bin_value", ""))

        if col not in X.columns:
            log.warning(
                "ive.synthesizer.missing_column",
                column=col,
                pattern_index=idx,
            )
            return None

        # Determine subgroup type and build construction rule
        interval = _parse_interval(bin_value)
        is_numeric_col = pd.api.types.is_numeric_dtype(X[col])

        if interval is not None and is_numeric_col:
            # Numeric interval subgroup
            lower, upper, left_closed, right_closed = interval
            rule: dict[str, Any] = {
                "column": col,
                "value": bin_value,
                "subgroup_type": "numeric_bin",
                "lower": float(lower) if not math.isinf(lower) else lower,
                "upper": float(upper) if not math.isinf(upper) else upper,
                "left_closed": left_closed,
                "right_closed": right_closed,
            }
            scores = _apply_numeric_interval(
                X[col].values,
                lower,
                upper,
                left_closed,
                right_closed,
            )
            log.debug(
                "ive.synthesizer.subgroup_numeric",
                column=col,
                lower=lower,
                upper=upper,
                left_closed=left_closed,
                right_closed=right_closed,
                support=int(np.sum(scores > 0)),
                n_samples=len(scores),
                pattern_index=idx,
            )
        elif is_numeric_col and interval is None:
            # Numeric column but bin_value is not an interval string.
            # Attempt direct numeric comparison to avoid string-mismatch
            # failures during bootstrap resampling.
            parsed_numeric = _try_parse_numeric(bin_value)
            if parsed_numeric is not None:
                rule = {
                    "column": col,
                    "value": bin_value,
                    "subgroup_type": "numeric_exact",
                    "exact_value": parsed_numeric,
                }
                scores = (
                    np.isclose(
                        np.nan_to_num(X[col].values.astype(np.float64), nan=np.inf),
                        parsed_numeric,
                        atol=1e-9,
                    )
                ).astype(np.float64)
            else:
                # Fallback to categorical string comparison
                rule = {
                    "column": col,
                    "value": bin_value,
                    "subgroup_type": "categorical",
                }
                scores = (X[col].astype(str) == bin_value).astype(np.float64).values
            log.debug(
                "ive.synthesizer.subgroup_numeric_noninterval",
                column=col,
                bin_value=bin_value,
                subgroup_type=rule["subgroup_type"],
                support=int(np.sum(scores > 0)),
                pattern_index=idx,
            )
        else:
            # Categorical subgroup
            rule = {
                "column": col,
                "value": bin_value,
                "subgroup_type": "categorical",
            }
            scores = (X[col].astype(str) == bin_value).astype(np.float64).values
            log.debug(
                "ive.synthesizer.subgroup_categorical",
                column=col,
                bin_value=bin_value,
                support=int(np.sum(scores > 0)),
                pattern_index=idx,
            )

        name = f"Latent_Subgroup_{_sanitise_name(col)}_{_sanitise_name(bin_value)}"

        return {
            "name": name,
            "pattern_type": "subgroup",
            "construction_rule": rule,
            "scores": scores,
            # Source metadata for ExplanationGenerator
            "column_name": col,
            "bin_value": bin_value,
            "effect_size": float(pattern.get("effect_size", 0.0)),
            "p_value": float(pattern.get("p_value", 0.0)),
            "sample_count": int(pattern.get("sample_count", 0)),
            # Initial diagnostics for bootstrap hardening
            "initial_support_rate": float(np.mean(scores > 0)) if len(scores) > 0 else 0.0,
            "initial_score_range": float(np.max(scores) - np.min(scores))
            if len(scores) > 0
            else 0.0,
            "initial_variance": float(np.var(scores)) if len(scores) > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Cluster synthesis
    # ------------------------------------------------------------------

    def _synthesize_cluster(
        self,
        pattern: dict[str, Any],
        X: pd.DataFrame,
        idx: int,
    ) -> dict[str, Any] | None:
        """Create a continuous proximity score from a cluster pattern.

        For each row, the score is ``1 / (1 + d)`` where *d* is the
        Euclidean distance from the row's numeric features to
        ``cluster_center``.  Missing values are filled with 0 before the
        distance calculation.

        Args:
            pattern: Cluster pattern dict containing ``cluster_id`` and
                     ``cluster_center`` (dict of column → float mean).
            X:       Feature DataFrame.
            idx:     Pattern index (used for fallback naming).

        Returns:
            Candidate dict, or ``None`` if no usable columns overlap.
        """
        cluster_id = pattern.get("cluster_id", idx)
        center: dict[str, float] = pattern.get("cluster_center", {})

        if not center:
            log.warning(
                "ive.synthesizer.empty_cluster_center",
                cluster_id=cluster_id,
                pattern_index=idx,
            )
            return None

        # Only use columns present in both X and the cluster center
        usable_cols = [c for c in center if c in X.columns]
        if not usable_cols:
            log.warning(
                "ive.synthesizer.no_overlapping_columns",
                cluster_id=cluster_id,
                pattern_index=idx,
            )
            return None

        # Build the center vector and the sample matrix (NaN → 0)
        center_vec = np.array([center[c] for c in usable_cols], dtype=np.float64)
        sample_matrix = X[usable_cols].values.astype(np.float64)
        np.nan_to_num(sample_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Euclidean distance from each row to the center
        diff = sample_matrix - center_vec
        distances = np.sqrt(np.sum(diff**2, axis=1))

        # Inverse-distance kernel → (0, 1]
        scores = 1.0 / (1.0 + distances)

        name = f"Latent_Cluster_{cluster_id}"

        return {
            "name": name,
            "pattern_type": "cluster",
            "construction_rule": {
                "cluster_id": int(cluster_id),
                "center": {c: float(center[c]) for c in usable_cols},
            },
            "scores": scores,
            # Source metadata for ExplanationGenerator
            "cluster_id": int(cluster_id),
            "cluster_center": {c: float(center[c]) for c in usable_cols},
            "effect_size": float(pattern.get("effect_size", 0.0)),
            "sample_count": int(pattern.get("sample_count", 0)),
            # Initial diagnostics for bootstrap hardening
            "initial_support_rate": float(np.mean(scores > 0)) if len(scores) > 0 else 0.0,
            "initial_score_range": float(np.max(scores) - np.min(scores))
            if len(scores) > 0
            else 0.0,
            "initial_variance": float(np.var(scores)) if len(scores) > 0 else 0.0,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitise_name(raw: str) -> str:
    """Replace whitespace and special characters with underscores.

    Used to build filesystem- and Python-friendly variable names from
    column values that may contain spaces, parentheses, slashes, etc.

    Args:
        raw: Original string (column name or bin value).

    Returns:
        Sanitised string safe for use in variable names.
    """
    out: list[str] = []
    for ch in raw:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    # Collapse consecutive underscores and strip leading/trailing
    result = "_".join(part for part in "".join(out).split("_") if part)
    return result or "unknown"


def _parse_interval(bin_value: str) -> tuple[float, float, bool, bool] | None:
    """Parse a pandas-style interval string into numeric bounds.

    Handles formats like ``(10.0, 20.0]``, ``[-inf, 5.0)``, ``[3, 7]``.

    Args:
        bin_value: String representation of the interval.

    Returns:
        ``(lower, upper, left_closed, right_closed)`` tuple, or ``None``
        if the string is not an interval.
    """
    m = _INTERVAL_RE.match(bin_value.strip())
    if m is None:
        return None

    try:
        lower_str = m.group("lower").strip()
        upper_str = m.group("upper").strip()

        lower = _parse_bound(lower_str, default=-math.inf)
        upper = _parse_bound(upper_str, default=math.inf)

        left_closed = m.group("left_bracket") == "["
        right_closed = m.group("right_bracket") == "]"

        return (lower, upper, left_closed, right_closed)
    except (ValueError, TypeError):
        return None


def _parse_bound(raw: str, *, default: float) -> float:
    """Parse a single interval bound, handling inf/nan variants.

    Args:
        raw:     Raw string from the interval notation.
        default: Fallback when the string represents infinity.

    Returns:
        Parsed float value.
    """
    normalised = raw.lower().strip()
    if normalised in ("-inf", "-infinity", "−inf", "−infinity"):
        return -math.inf
    if normalised in ("inf", "+inf", "infinity", "+infinity"):
        return math.inf
    if normalised == "nan":
        return default
    return float(raw)


def _try_parse_numeric(value: str) -> float | None:
    """Attempt to parse a string as a plain numeric value.

    Returns ``None`` when the string does not represent a finite number.

    Args:
        value: Raw string.

    Returns:
        Parsed float or ``None``.
    """
    try:
        result = float(value)
        if math.isfinite(result):
            return result
        return None
    except (ValueError, TypeError):
        return None


def _apply_numeric_interval(
    values: np.ndarray,
    lower: float,
    upper: float,
    left_closed: bool,
    right_closed: bool,
) -> np.ndarray:
    """Apply an interval membership test to a numeric array.

    NaN values are treated as *not* belonging to any interval and
    produce 0.0 in the output.

    Args:
        values:       1-D numeric array.
        lower:        Lower bound.
        upper:        Upper bound.
        left_closed:  Whether the lower bound is inclusive.
        right_closed: Whether the upper bound is inclusive.

    Returns:
        1-D float64 array of 0.0 / 1.0.
    """
    arr = np.asarray(values, dtype=np.float64)

    # NaN guard: NaN comparisons return False in numpy, but we make
    # the intent explicit for clarity and safety.
    nan_mask = np.isnan(arr)

    if left_closed:
        left_mask = arr >= lower
    else:
        left_mask = arr > lower

    if right_closed:
        right_mask = arr <= upper
    else:
        right_mask = arr < upper

    result = (left_mask & right_mask & ~nan_mask).astype(np.float64)
    return result


def apply_construction_rule(
    rule: dict[str, Any],
    pattern_type: str,
    X: pd.DataFrame,
) -> np.ndarray:
    """Re-apply a construction rule to a (potentially resampled) DataFrame.

    This is the single source of truth for synthesising scores from a
    stored ``construction_rule``.  Used by both :class:`VariableSynthesizer`
    and :class:`~ive.construction.bootstrap_validator.BootstrapValidator`.

    Subgroup rules
    ~~~~~~~~~~~~~~
    * ``subgroup_type == "numeric_bin"`` → interval membership using
      ``lower``, ``upper``, ``left_closed``, ``right_closed``.
    * ``subgroup_type == "numeric_exact"`` → close-enough numeric equality
      against ``exact_value``.
    * ``subgroup_type == "categorical"`` (or absent) → string equality
      against ``value``.  As a fallback the function also attempts
      interval parsing from ``value`` if the column is numeric.

    Cluster rules
    ~~~~~~~~~~~~~
    Inverse-distance kernel as in :meth:`VariableSynthesizer._synthesize_cluster`.

    Args:
        rule:         The ``construction_rule`` dict from a candidate.
        pattern_type: ``"subgroup"`` or ``"cluster"``.
        X:            Feature DataFrame (may be a bootstrap resample).

    Returns:
        1-D ``np.ndarray`` of synthesised scores with length ``len(X)``.
    """
    n = len(X)

    if pattern_type == "subgroup":
        return _apply_subgroup_rule(rule, X, n)

    if pattern_type == "cluster":
        return _apply_cluster_rule(rule, X, n)

    # Unrecognised pattern type → zeros
    log.warning(
        "ive.apply_rule.unknown_pattern_type",
        pattern_type=pattern_type,
    )
    return np.zeros(n, dtype=np.float64)


def _apply_subgroup_rule(
    rule: dict[str, Any],
    X: pd.DataFrame,
    n: int,
) -> np.ndarray:
    """Apply a subgroup construction rule to a DataFrame.

    Dispatches to the correct reconstruction path based on
    ``subgroup_type``: numeric_bin, numeric_exact, or categorical.

    Args:
        rule: The subgroup ``construction_rule`` dict.
        X:    Feature DataFrame (may be a bootstrap resample).
        n:    Number of rows in X.

    Returns:
        1-D float64 array of scores.
    """
    col = rule.get("column", "")
    value = str(rule.get("value", ""))
    subgroup_type = rule.get("subgroup_type", "")

    if not col:
        log.warning("ive.apply_rule.empty_column", rule=rule)
        return np.zeros(n, dtype=np.float64)

    if col not in X.columns:
        log.debug(
            "ive.apply_rule.column_not_in_dataframe",
            column=col,
            available_columns=list(X.columns)[:10],
        )
        return np.zeros(n, dtype=np.float64)

    # ── Numeric interval path ──────────────────────────────────────────
    if subgroup_type == "numeric_bin":
        lower = rule.get("lower")
        upper = rule.get("upper")
        left_closed = bool(rule.get("left_closed", False))
        right_closed = bool(rule.get("right_closed", True))

        if lower is not None and upper is not None:
            try:
                col_values = X[col].values.astype(np.float64)
            except (ValueError, TypeError):
                log.warning(
                    "ive.apply_rule.numeric_cast_failed",
                    column=col,
                    subgroup_type=subgroup_type,
                )
                return np.zeros(n, dtype=np.float64)

            return _apply_numeric_interval(
                col_values,
                float(lower),
                float(upper),
                left_closed,
                right_closed,
            )

        # Bounds missing — attempt interval parsing from the value string
        # as a recovery path before giving up.
        log.warning(
            "ive.apply_rule.missing_bounds_attempting_parse",
            column=col,
            value=value,
        )
        interval = _parse_interval(value)
        if interval is not None:
            lower_f, upper_f, lc, rc = interval
            try:
                col_values = X[col].values.astype(np.float64)
                return _apply_numeric_interval(col_values, lower_f, upper_f, lc, rc)
            except (ValueError, TypeError):
                pass

        log.warning(
            "ive.apply_rule.numeric_bin_unrecoverable",
            column=col,
            value=value,
        )
        return np.zeros(n, dtype=np.float64)

    # ── Numeric exact match path ───────────────────────────────────────
    if subgroup_type == "numeric_exact":
        exact_value = rule.get("exact_value")
        if exact_value is not None:
            try:
                col_values = X[col].values.astype(np.float64)
                safe_values = np.nan_to_num(col_values, nan=np.inf)
                return np.isclose(safe_values, float(exact_value), atol=1e-9).astype(
                    np.float64,
                )
            except (ValueError, TypeError):
                log.warning(
                    "ive.apply_rule.numeric_exact_cast_failed",
                    column=col,
                    exact_value=exact_value,
                )
                return np.zeros(n, dtype=np.float64)

        # No exact_value stored — fall through to categorical
        log.warning(
            "ive.apply_rule.numeric_exact_missing_value",
            column=col,
        )

    # ── Categorical path ──────────────────────────────────────────────
    if subgroup_type in ("categorical", "numeric_exact", ""):
        # Before falling back to string equality, try interval parsing
        # in case this is a legacy rule without subgroup_type that
        # actually came from a numeric bin.
        if pd.api.types.is_numeric_dtype(X[col]):
            interval = _parse_interval(value)
            if interval is not None:
                lower_f, upper_f, lc, rc = interval
                try:
                    return _apply_numeric_interval(
                        X[col].values.astype(np.float64),
                        lower_f,
                        upper_f,
                        lc,
                        rc,
                    )
                except (ValueError, TypeError):
                    pass

            # Also try exact numeric comparison for plain numeric strings
            parsed = _try_parse_numeric(value)
            if parsed is not None:
                try:
                    col_values = X[col].values.astype(np.float64)
                    safe_values = np.nan_to_num(col_values, nan=np.inf)
                    return np.isclose(safe_values, parsed, atol=1e-9).astype(
                        np.float64,
                    )
                except (ValueError, TypeError):
                    pass

        # Pure string comparison
        return (X[col].astype(str) == value).astype(np.float64).values

    # Unrecognised subgroup_type — log and return zeros
    log.warning(
        "ive.apply_rule.unknown_subgroup_type",
        subgroup_type=subgroup_type,
        column=col,
    )
    return np.zeros(n, dtype=np.float64)


def _apply_cluster_rule(
    rule: dict[str, Any],
    X: pd.DataFrame,
    n: int,
) -> np.ndarray:
    """Apply a cluster construction rule to a DataFrame.

    Computes the inverse-distance kernel score for each row relative
    to the stored cluster center.

    Args:
        rule: The cluster ``construction_rule`` dict.
        X:    Feature DataFrame.
        n:    Number of rows in X.

    Returns:
        1-D float64 array of proximity scores ∈ (0, 1].
    """
    center: dict[str, float] = rule.get("center", {})
    usable_cols = [c for c in center if c in X.columns]

    if not usable_cols:
        return np.zeros(n, dtype=np.float64)

    center_vec = np.array([center[c] for c in usable_cols], dtype=np.float64)
    sample_matrix = X[usable_cols].values.astype(np.float64)
    np.nan_to_num(sample_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    diff = sample_matrix - center_vec
    distances = np.sqrt(np.sum(diff**2, axis=1))
    return 1.0 / (1.0 + distances)
