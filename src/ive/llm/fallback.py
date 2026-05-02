"""Graceful-degradation wrapper around LLM generation.

Resolution order:

    1. If LLM_EXPLANATIONS_ENABLED is False → skip Groq, return rule-based.
    2. If circuit breaker is open → skip Groq, return rule-based.
    3. Cache hit → return cached text.
    4. Call Groq → validate output → cache + return on success.
    5. On any failure (transient, validation, unexpected) → rule-based fallback.

The cache and circuit breaker are optional to keep this layer testable in
isolation; pass ``None`` for either to bypass.

Plan reference: §A1 (fallback module), §106 (sharpened retry — handled at
caller level, not here), §171 (cooperative cancellation via asyncio.Event).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ive.llm.cache import RedisLLMCache, make_key
from ive.llm.circuit_breaker import CircuitBreaker
from ive.llm.client import (
    ChatResult,
    GroqClient,
    LLMAuthError,
    LLMBadRequest,
    LLMUnavailable,
)
from ive.llm.prompts import OUTPUT_TOKEN_CAPS, render
from ive.llm.types import GenerationResult, ValidationReport
from ive.llm.validators import composite_validate

logger = logging.getLogger(__name__)


RuleBasedFn = Callable[[], str]


async def generate_with_fallback(
    *,
    function: str,
    prompt_version: str,
    facts: dict[str, Any],
    rule_based: RuleBasedFn,
    client: GroqClient | None,
    cache: RedisLLMCache | None,
    breaker: CircuitBreaker | None,
    enabled: bool,
    allow_causal: bool = False,
    cancel_event: asyncio.Event | None = None,
    entity_index: tuple[str, str] | None = None,
) -> GenerationResult:
    """Generate text with full fallback chain.

    ``rule_based`` is invoked synchronously as the fallback; it must not
    block on I/O (it should already be deterministic prose generation).
    """
    if cancel_event is not None and cancel_event.is_set():
        return _fallback(rule_based, reason="cancelled")

    if not enabled or client is None:
        return _fallback(rule_based, reason="llm_disabled")

    if breaker is not None and await breaker.is_open():
        return _fallback(rule_based, reason="circuit_breaker_open")

    cache_key = make_key(function, prompt_version, facts)

    # Cache lookup
    if cache is not None:
        cached = await cache.get(cache_key)
        if cached is not None:
            return GenerationResult(
                text=cached,
                source="cache",
                cache_status="hit",
            )

    if cancel_event is not None and cancel_event.is_set():
        return _fallback(rule_based, reason="cancelled")

    # Render prompt + call provider
    try:
        system, user = render(function, prompt_version, facts)
    except KeyError as exc:
        logger.error("llm.prompt.unregistered", extra={"function": function, "version": prompt_version})
        return _fallback(rule_based, reason=f"prompt_unregistered:{exc}")

    max_tokens = OUTPUT_TOKEN_CAPS.get(function)

    try:
        result = await _call_with_breaker(
            client=client,
            breaker=breaker,
            system=system,
            user=user,
            max_tokens=max_tokens,
        )
    except LLMAuthError as exc:
        # Auth errors do NOT trip the breaker (per §109) — but they are alarming.
        logger.error("llm.auth_error", extra={"error": str(exc)})
        return _fallback(rule_based, reason=f"auth_error:{exc}")
    except LLMBadRequest as exc:
        # Our payload is malformed — fix code, don't retry.
        logger.error("llm.bad_request", extra={"error": str(exc)})
        return _fallback(rule_based, reason=f"bad_request:{exc}")
    except LLMUnavailable as exc:
        return _fallback(rule_based, reason=f"unavailable:{exc}")
    except Exception as exc:  # defensive: never let a stray error break the pipeline
        logger.exception("llm.unexpected_error", extra={"error": str(exc)})
        return _fallback(rule_based, reason=f"unexpected:{exc.__class__.__name__}")

    # Validate output
    validation = composite_validate(result.text, facts, allow_causal=allow_causal)
    if not validation.passed:
        logger.warning(
            "llm.validation.failed",
            extra={
                "function": function,
                "rule": validation.rule,
                "failures": validation.failures,
                "request_id": result.request_id,
            },
        )
        return _fallback(
            rule_based,
            reason=f"validation_failed:{validation.rule}",
            validation=validation,
            tokens_in=result.prompt_tokens,
            tokens_out=result.completion_tokens,
            latency_ms=result.latency_ms,
        )

    # Cache success
    if cache is not None:
        await cache.set(cache_key, result.text, entity_index=entity_index)

    return GenerationResult(
        text=result.text,
        source="llm",
        validation=validation,
        tokens_in=result.prompt_tokens,
        tokens_out=result.completion_tokens,
        latency_ms=result.latency_ms,
        cache_status="miss",
    )


async def _call_with_breaker(
    *,
    client: GroqClient,
    breaker: CircuitBreaker | None,
    system: str,
    user: str,
    max_tokens: int | None,
) -> ChatResult:
    """Invoke the chat client and update the breaker on transient failures."""
    try:
        result = await client.chat(system=system, user=user, max_tokens=max_tokens)
    except LLMUnavailable:
        if breaker is not None:
            await breaker.record_failure()
        raise
    if breaker is not None:
        await breaker.record_success()
    return result


def _fallback(
    rule_based: RuleBasedFn,
    *,
    reason: str,
    validation: ValidationReport | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    latency_ms: int = 0,
) -> GenerationResult:
    """Build a fallback ``GenerationResult`` from the rule-based generator."""
    text = rule_based()
    # Bucket the reason to keep cardinality bounded — anything after `:` is
    # high-cardinality (exception text). The metric label keeps the prefix.
    bucket = reason.split(":", 1)[0]
    try:
        from ive.observability.metrics import (
            record_fallback,
            record_validation_failure,
        )

        record_fallback(reason=bucket)
        if validation is not None and not validation.passed:
            for failure in (validation.failures or [bucket]):
                record_validation_failure(reason=failure)
    except Exception:  # pragma: no cover - defensive
        pass
    return GenerationResult(
        text=text,
        source="fallback",
        validation=validation,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
        cache_status="bypass",
        failure_reason=reason,
    )


async def shielded_db_write(awaitable: Awaitable[None]) -> None:
    """Run a final DB write with ``asyncio.shield`` so cancellation doesn't drop it.

    Call this around the persistence step in cancel-aware tasks (per §171).
    """
    await asyncio.shield(awaitable)
