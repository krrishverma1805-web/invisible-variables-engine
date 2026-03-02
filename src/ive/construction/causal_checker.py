"""
Causal Plausibility Checker.

Filters latent variable candidates for causal plausibility using
heuristic directional checks. This does not claim to prove causality
but removes clearly spurious candidates before Phase 4 explanation.

Checks performed:
    1. Reverse causality: Does the LV proxy a consequence of Y rather than a cause?
    2. Confounding proxy: Is the LV merely a proxy for an already-included feature?
    3. Temporal ordering: If time columns exist, does the LV precede Y?
"""

from __future__ import annotations

import structlog

from ive.core.pipeline import LatentVariableCandidate

log = structlog.get_logger(__name__)


class CausalChecker:
    """
    Heuristic causal plausibility filter for LatentVariableCandidate objects.

    Operates as a filter — candidates failing causal checks are assigned
    a reduced confidence_score with a warning, not silently discarded.
    """

    def filter(
        self,
        candidates: list[LatentVariableCandidate],
        df: object,  # pd.DataFrame
        target_column: str | None = None,
    ) -> list[LatentVariableCandidate]:
        """
        Apply causal plausibility checks to all candidates.

        Args:
            candidates: List of LatentVariableCandidate objects from Phase 4.
            df: The original DataFrame for correlation and ordering checks.
            target_column: Name of the target column for reverse-causality check.

        Returns:
            Same list with confidence_score adjusted for failed checks.

        TODO:
            - For each candidate:
                  Check 1: Correlate candidate features with target
                           If correlation(feature, target) > 0.9 → flag reverse causality
                  Check 2: Compute max VIF of candidate_features vs. all features
                           If VIF > 10 → flag confounding proxy
                  Check 3: If datetime features present, verify proxy features
                           precede the target in time ordering
            - Apply confidence penalty multiplier (×0.5) for each failed check
        """
        log.info("ive.causal_checker.start", n_candidates=len(candidates))

        for candidate in candidates:
            # TODO: Implement checks (see docstring)
            pass

        return candidates

    def _is_reverse_causal(
        self,
        candidate: LatentVariableCandidate,
        df: object,
        target_column: str,
    ) -> bool:
        """
        Check if the candidate features are so correlated with the target
        that they likely represent a consequence rather than a cause.

        TODO:
            - Compute Pearson correlation of each candidate feature with the target
            - Return True if any correlation > 0.9
        """
        # TODO: Implement reverse causality check
        return False

    def _is_confounding_proxy(
        self,
        candidate: LatentVariableCandidate,
        df: object,
        all_feature_columns: list[str],
    ) -> bool:
        """
        Check if the candidate is simply a proxy for an already-present feature.

        TODO:
            - Check pairwise correlation between candidate_features and all_feature_columns
            - If max correlation > 0.95 with a non-candidate feature → return True
        """
        # TODO: Implement confounding proxy check
        return False
