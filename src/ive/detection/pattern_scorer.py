"""
Pattern Scorer.

Ranks and filters the raw patterns discovered by all three detection
strategies (SHAP, clustering, subgroup discovery) into a unified
ranked list of latent variable candidates.

Scoring formula:
    score = 0.40 * effect_size + 0.35 * bootstrap_stability + 0.25 * coverage

Only patterns meeting minimum thresholds on all three sub-scores advance
to Phase 4 for variable construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from ive.utils.statistics import cohens_d

log = structlog.get_logger(__name__)

# Minimum thresholds for a pattern to survive scoring
_MIN_EFFECT_SIZE = 0.2  # Cohen's d
_MIN_COVERAGE = 0.03  # Must cover at least 3% of dataset
_MIN_STABILITY = 0.5  # Must be reproducible across bootstrap resamples


@dataclass
class ScoredPattern:
    """A pattern with its final composite score."""

    source: str  # 'shap' | 'cluster' | 'subgroup'
    raw_pattern: object  # Original pattern object
    effect_size: float = 0.0
    coverage: float = 0.0
    stability: float = 0.0
    composite_score: float = 0.0
    feature_references: list[str] = field(default_factory=list)
    sample_mask: np.ndarray[Any, Any] | None = None


class PatternScorer:
    """
    Scores, de-duplicates, and ranks detected patterns.

    Patterns from different sources are placed in a common scoring
    framework so that Phase 4 (Construct) works with a single ranked list.
    """

    WEIGHTS = {"effect_size": 0.60, "stability": 0.0, "coverage": 0.40}

    def score_and_rank(
        self,
        raw_patterns: list[object],
        residuals: np.ndarray[Any, Any],
        cluster_labels: np.ndarray[Any, Any] | None = None,
        top_k: int = 20,
    ) -> list[ScoredPattern]:
        """
        Score all raw patterns and return the top-k.

        Args:
            raw_patterns: Mixed list of SubgroupPattern or cluster-derived dicts.
            residuals: OOF residuals for effect size computation.
            cluster_labels: HDBSCAN labels (optional, for cluster-source patterns).
            top_k: Maximum number of scored patterns to return.

        Returns:
            List of ScoredPattern objects sorted by composite_score descending.

        TODO:
            - Iterate raw_patterns
            - Compute effect_size: Cohen's d of residuals inside vs outside pattern
            - Compute coverage: fraction of dataset matching pattern
            - Compute stability: placeholder 0.8 until bootstrap is available
            - Compute composite score using self.WEIGHTS
            - Filter by minimum thresholds
            - De-duplicate overlapping patterns (Jaccard index > 0.9)
            - Return[:top_k]
        """
        log.info("ive.pattern_scorer.start", n_raw=len(raw_patterns))

        scored: list[ScoredPattern] = []
        n = len(residuals)

        for pat_obj in raw_patterns:
            pat: dict[str, Any] = pat_obj if isinstance(pat_obj, dict) else getattr(pat_obj, "__dict__", {})
            if not pat:
                continue

            # --- Build boolean sample_mask ---
            sample_indices = pat.get("sample_indices")
            if sample_indices is not None:
                mask = np.zeros(n, dtype=bool)
                idx = np.asarray(sample_indices, dtype=int)
                idx = idx[(idx >= 0) & (idx < n)]
                mask[idx] = True
            else:
                # Fallback: cannot reconstruct mask — use stored values
                stored_es = pat.get("effect_size")
                sample_count = pat.get("sample_count")
                if stored_es is None or sample_count is None or sample_count < 5:
                    log.debug("ive.pattern_scorer.skip_no_mask", pattern=pat.get("pattern_type"))
                    continue
                abs_d = float(abs(stored_es))
                coverage = float(sample_count) / n if n > 0 else 0.0
                composite = self._compute_composite(abs_d, 0.0, coverage)
                source = "subgroup" if pat.get("pattern_type") == "subgroup" else "cluster"
                feature_refs: list[str] = []
                col = pat.get("column_name")
                if col:
                    feature_refs.append(str(col))
                scored.append(
                    ScoredPattern(
                        source=source,
                        raw_pattern=pat_obj,
                        effect_size=abs_d,
                        coverage=coverage,
                        stability=0.0,
                        composite_score=composite,
                        feature_references=feature_refs,
                        sample_mask=None,
                    )
                )
                continue

            # Skip patterns with very small sample counts
            if mask.sum() < 5:
                log.debug("ive.pattern_scorer.skip_small", count=int(mask.sum()))
                continue

            # --- Effect size (Cohen's d, absolute) ---
            abs_d = float(abs(cohens_d(residuals[mask], residuals[~mask])))

            # --- Coverage ---
            coverage = float(mask.sum()) / n if n > 0 else 0.0

            # --- Composite (stability = 0 until bootstrap wired) ---
            composite = self._compute_composite(abs_d, 0.0, coverage)

            # --- Source ---
            source = "subgroup" if pat.get("pattern_type") == "subgroup" else "cluster"

            # --- Feature references ---
            feature_refs = []
            col = pat.get("column_name")
            if col:
                feature_refs.append(str(col))

            scored.append(
                ScoredPattern(
                    source=source,
                    raw_pattern=pat_obj,
                    effect_size=abs_d,
                    coverage=coverage,
                    stability=0.0,
                    composite_score=composite,
                    feature_references=feature_refs,
                    sample_mask=mask,
                )
            )

        # --- Filter by minimum thresholds ---
        scored = [
            p for p in scored if p.effect_size >= _MIN_EFFECT_SIZE and p.coverage >= _MIN_COVERAGE
        ]

        # --- De-duplicate overlapping patterns (Jaccard > 0.9) ---
        scored.sort(key=lambda p: p.composite_score, reverse=True)
        deduplicated: list[ScoredPattern] = []
        for pat in scored:
            is_duplicate = False
            if pat.sample_mask is not None:
                for accepted in deduplicated:
                    if accepted.sample_mask is not None:
                        intersection = np.count_nonzero(pat.sample_mask & accepted.sample_mask)
                        union = np.count_nonzero(pat.sample_mask | accepted.sample_mask)
                        if union > 0 and intersection / union > 0.9:
                            is_duplicate = True
                            break
            if not is_duplicate:
                deduplicated.append(pat)
        scored = deduplicated

        scored.sort(key=lambda p: p.composite_score, reverse=True)
        log.info("ive.pattern_scorer.done", n_scored=len(scored))
        return scored[:top_k]

    def _compute_composite(
        self,
        effect_size: float,
        stability: float,
        coverage: float,
    ) -> float:
        """Compute the weighted composite score."""
        w = self.WEIGHTS
        return (
            w["effect_size"] * effect_size + w["stability"] * stability + w["coverage"] * coverage
        )
