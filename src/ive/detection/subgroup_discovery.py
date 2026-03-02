"""
Subgroup Discovery.

Discovers rule-based subgroups of the dataset where the model's residual
error is systematically high. Uses a beam search over conjunctive feature
conditions.

Quality metric: WRAcc (Weighted Relative Accuracy)
    WRAcc(rule) = P(cond) * (mean_residual(cond) - mean_residual(all))

A high WRAcc means the rule targets a large, highly-deviant subgroup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class SubgroupPattern:
    """A discovered subgroup pattern with its quality metrics."""

    rule: str  # Human-readable rule e.g. "age > 50 AND city == 'NYC'"
    conditions: list[dict]  # Machine-readable condition list
    coverage: float = 0.0  # Fraction of dataset matching the rule
    wracc: float = 0.0  # WRAcc quality metric
    mean_residual: float = 0.0  # Mean residual within the subgroup
    mean_residual_outside: float = 0.0
    effect_size: float = 0.0  # Cohen's d vs. complement
    sample_mask: np.ndarray | None = None  # Boolean mask for matched samples


class SubgroupDiscoverer:
    """
    Beam-search subgroup discovery on the residual space.

    Iteratively builds conjunctive rules by selecting conditions that
    maximise WRAcc within a beam of width W.
    """

    def __init__(
        self,
        beam_width: int = 10,
        search_depth: int = 3,
        min_coverage: float = 0.05,
        n_bins: int = 5,
    ) -> None:
        """
        Args:
            beam_width: Number of candidates kept at each search depth.
            search_depth: Maximum number of conditions per rule.
            min_coverage: Minimum fraction of dataset a subgroup must cover.
            n_bins: Number of bins for discretising continuous features.
        """
        self.beam_width = beam_width
        self.search_depth = search_depth
        self.min_coverage = min_coverage
        self.n_bins = n_bins

    def discover(
        self,
        df: object,  # pd.DataFrame
        residuals: np.ndarray,
        feature_columns: list[str],
        top_k: int = 20,
    ) -> list[SubgroupPattern]:
        """
        Discover top-k subgroups with high systematic residual error.

        Args:
            df: DataFrame containing feature_columns.
            residuals: OOF residuals array (n_samples,).
            feature_columns: Columns to use as condition candidates.
            top_k: Return only the top-k patterns by WRAcc.

        Returns:
            List of SubgroupPattern objects ordered by WRAcc descending.

        TODO:
            - Discretise continuous feature columns into bins
            - Build initial beam: all single-condition patterns
            - For each depth level:
                  expand beam by adding one more condition to each candidate
                  evaluate WRAcc for all expansions
                  prune to beam_width best
            - Filter by min_coverage and top_k
            - Compute effect_size (Cohen's d) for each pattern
        """
        log.info(
            "ive.subgroup_discovery.start",
            n_features=len(feature_columns),
            n_samples=len(residuals),
        )

        # TODO: Implement beam search
        patterns: list[SubgroupPattern] = []

        log.info("ive.subgroup_discovery.done", n_patterns=len(patterns))
        return patterns

    def _compute_wracc(
        self,
        mask: np.ndarray,
        residuals: np.ndarray,
    ) -> float:
        """
        Compute WRAcc for a given boolean mask.

        WRAcc = P(cond) * (mean_residual(cond) - mean_residual(all))

        TODO:
            - coverage = mask.mean()
            - return coverage * (residuals[mask].mean() - residuals.mean())
        """
        coverage = float(mask.mean())
        if coverage < self.min_coverage:
            return 0.0
        return coverage * (float(np.mean(residuals[mask])) - float(np.mean(residuals)))

    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size between two groups.

        TODO:
            - pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
            - return (mean1 - mean2) / pooled_std
        """
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        pooled_std = np.sqrt(
            ((n1 - 1) * group1.std() ** 2 + (n2 - 1) * group2.std() ** 2) / (n1 + n2 - 2)
        )
        if pooled_std == 0:
            return 0.0
        return float((group1.mean() - group2.mean()) / pooled_std)
