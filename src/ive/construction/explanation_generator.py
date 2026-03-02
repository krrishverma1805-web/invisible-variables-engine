"""
Explanation Generator.

Produces human-readable natural language explanations for discovered
latent variables. Uses template-based generation as a baseline, with
placeholder for LLM-based generation.

Explanation structure:
    1. What the latent variable represents
    2. Which subgroup it affects
    3. What existing features it correlates with
    4. Why this matters for model improvement
"""

from __future__ import annotations

import structlog

from ive.core.pipeline import LatentVariableCandidate

log = structlog.get_logger(__name__)

_EXPLANATION_TEMPLATE = """
The model systematically {error_direction} predictions for samples where {subgroup_description}.
This suggests an unmeasured factor — tentatively named **{lv_name}** — that captures
{feature_description}.

Existing features most correlated with this factor: {feature_list}.

If this variable were measured and included, it could reduce model error by approximately
{coverage_pct:.1f}% of samples (estimated effect size: {effect_size:.2f}).

Statistical confidence: {confidence_pct:.0f}% (bootstrap stability: {stability:.2f},
p-value: {p_value:.4f}).
""".strip()


class ExplanationGenerator:
    """
    Generates natural language explanations for LatentVariableCandidate objects.

    Uses template-based generation by default. Can be extended to call
    an LLM (e.g., OpenAI API) for richer, more natural explanations.
    """

    def generate(
        self,
        candidate: LatentVariableCandidate,
        profile: dict,
    ) -> str:
        """
        Generate a natural language explanation for a latent variable.

        Args:
            candidate: The LatentVariableCandidate to explain.
            profile: Dataset profile dict from Phase 1.

        Returns:
            A multi-sentence explanation string.

        TODO:
            - Determine error direction (over/under-predicting)
            - Build subgroup_description from candidate's conditions/cluster
            - Build feature_description from top candidate_features
            - Fill in _EXPLANATION_TEMPLATE
            - Optional: POST to an LLM API for natural paraphrase
        """
        log.debug("ive.explanation.generate", candidate_name=candidate.name)

        error_direction = "under-estimates" if candidate.effect_size > 0 else "over-estimates"
        feature_list = ", ".join(candidate.candidate_features[:5]) or "unknown"
        subgroup_description = f"they belong to cluster #{candidate.rank}"
        feature_description = (
            "a combination of baseline conditions not captured by the current feature set"
        )
        stability = candidate.validation.get("bootstrap_stability", 0.0)
        p_value = candidate.validation.get("p_value", 1.0)

        explanation = _EXPLANATION_TEMPLATE.format(
            error_direction=error_direction,
            subgroup_description=subgroup_description,
            lv_name=candidate.name or "Unknown Latent Variable",
            feature_description=feature_description,
            feature_list=feature_list,
            coverage_pct=candidate.coverage_pct,
            effect_size=candidate.effect_size,
            confidence_pct=candidate.confidence_score * 100,
            stability=stability,
            p_value=p_value,
        )

        return explanation

    def generate_name(self, candidate: LatentVariableCandidate, profile: dict) -> str:
        """
        Generate a concise, descriptive name for the latent variable.

        TODO:
            - Use candidate_features and profile column semantics to generate a name
            - e.g., if top features are zip_code and commute_time → "Neighbourhood Quality Factor"
            - Fall back to generic name if no signal
        """
        if candidate.candidate_features:
            return f"Latent {candidate.candidate_features[0].replace('_', ' ').title()} Factor"
        return f"Latent Variable #{candidate.rank}"
