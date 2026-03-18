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
* **Cluster** pattern → continuous score ∈ (0, 1] via the inverse-distance
  kernel ``1 / (1 + d)`` where *d* is the Euclidean distance from the row's
  numeric features to the cluster center.

Output
------
A list of candidate dicts, each containing:

* ``name``              — human-readable label (e.g. ``Latent_Subgroup_day_of_week_weekend``)
* ``pattern_type``      — ``"subgroup"`` or ``"cluster"``
* ``construction_rule`` — dict describing how the variable was synthesised
* ``scores``            — ``np.ndarray`` of shape ``(n_samples,)``
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


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
            List of candidate dicts, one per pattern, each containing:

            * ``name``              — human-readable variable name
            * ``pattern_type``      — ``"subgroup"`` or ``"cluster"``
            * ``construction_rule`` — dict describing the synthesis logic
            * ``scores``            — ``np.ndarray`` of shape ``(n_samples,)``

            Patterns with an unrecognised ``pattern_type`` are silently
            skipped with a warning log entry.
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

        The indicator is 1 where ``str(X[column_name]) == str(bin_value)``
        and 0 otherwise.  String-casting both sides ensures safe comparison
        across mixed dtypes (numeric bins, categorical values, booleans).

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

        # Cast both sides to string for safe, type-agnostic comparison
        scores = (X[col].astype(str) == bin_value).astype(np.float64).values

        name = f"Latent_Subgroup_{_sanitise_name(col)}_{_sanitise_name(bin_value)}"

        return {
            "name": name,
            "pattern_type": "subgroup",
            "construction_rule": {"column": col, "value": bin_value},
            "scores": scores,
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


def apply_construction_rule(
    rule: dict[str, Any],
    pattern_type: str,
    X: pd.DataFrame,
) -> np.ndarray:
    """Re-apply a construction rule to a (potentially resampled) DataFrame.

    This is the single source of truth for synthesising scores from a
    stored ``construction_rule``.  Used by both :class:`VariableSynthesizer`
    and :class:`~ive.construction.bootstrap_validator.BootstrapValidator`.

    Args:
        rule:         The ``construction_rule`` dict from a candidate.
        pattern_type: ``"subgroup"`` or ``"cluster"``.
        X:            Feature DataFrame (may be a bootstrap resample).

    Returns:
        1-D ``np.ndarray`` of synthesised scores with length ``len(X)``.
    """
    n = len(X)

    if pattern_type == "subgroup":
        col = rule.get("column", "")
        value = str(rule.get("value", ""))

        if col not in X.columns:
            return np.zeros(n, dtype=np.float64)

        return (X[col].astype(str) == value).astype(np.float64).values

    if pattern_type == "cluster":
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

    # Unrecognised pattern type → zeros
    return np.zeros(n, dtype=np.float64)
