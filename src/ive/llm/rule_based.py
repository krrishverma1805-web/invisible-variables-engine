"""Rule-based fallback adapters for the LLM enrichment task.

Returns no-arg callables that wrap the existing rule-based prose
generators in :mod:`ive.construction.explanation_generator`.  Each
``generate_with_fallback`` call gets one of these as its
``rule_based`` parameter so the fallback path is never silent.

Plan reference: §A1 (fallback module), §107.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ive.construction.explanation_generator import ExplanationGenerator


def lv_rule_based(
    generator: ExplanationGenerator,
    candidate: dict[str, Any],
) -> Callable[[], str]:
    """Return a no-arg callable producing the rule-based LV explanation."""

    def _fn() -> str:
        try:
            return generator.generate_latent_variable_explanation(candidate)
        except Exception:  # pragma: no cover - never let the fallback throw
            return (
                f"Latent variable {candidate.get('name', 'unnamed')!r} was constructed "
                f"with an effect size of {candidate.get('effect_size', 'n/a')}."
            )

    return _fn


def experiment_rule_based(
    generator: ExplanationGenerator,
    *,
    headline: bool,
    payload: dict[str, Any],
) -> Callable[[], str]:
    """Return a no-arg callable for the experiment headline or narrative.

    The rule-based generator emits a full multi-paragraph narrative; for
    the ``headline`` variant we extract the first sentence so the shape
    matches the LLM-side expected output.
    """

    def _fn() -> str:
        try:
            full = generator.generate_experiment_summary(payload)
        except Exception:  # pragma: no cover
            full = "Experiment completed."
        if not headline:
            return full
        # Extract the first sentence as a fallback headline.
        for sep in (". ", "\n"):
            if sep in full:
                return full.split(sep, 1)[0].rstrip(".") + "."
        return full

    return _fn
