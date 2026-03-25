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

    WEIGHTS = {"effect_size": 0.40, "stability": 0.35, "coverage": 0.25}

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

        # TODO: Implement pattern scoring loop

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
