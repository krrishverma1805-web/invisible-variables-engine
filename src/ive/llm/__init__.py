"""IVE LLM enrichment package.

Post-processes pipeline output to produce business-readable explanations
via a hosted (Groq) or self-hosted OpenAI-compatible LLM.  Every output is
validated against the input facts; any failure falls back cleanly to the
rule-based prose generator.

Public surface:

    from ive.llm import (
        GroqClient,
        ChatResult,
        GenerationRequest,
        GenerationResult,
        ValidationReport,
        LLMUnavailable,
        composite_validate,
        sanitize_user_input,
        generate_with_fallback,
    )

Plan reference: §A1, §103 (batched calls), §95 (canonical residuals),
§107 (temp=0), §171 (cooperative cancellation).
"""

from __future__ import annotations

from ive.llm.client import ChatResult, GroqClient, LLMUnavailable
from ive.llm.fallback import generate_with_fallback
from ive.llm.types import GenerationRequest, GenerationResult, ValidationReport
from ive.llm.validators import composite_validate, sanitize_user_input

__all__ = [
    "ChatResult",
    "GenerationRequest",
    "GenerationResult",
    "GroqClient",
    "LLMUnavailable",
    "ValidationReport",
    "composite_validate",
    "generate_with_fallback",
    "sanitize_user_input",
]
