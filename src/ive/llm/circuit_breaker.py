"""Redis-backed circuit breaker for LLM calls.

Counts toward the breaker (per §109): timeouts, 5xx, 429-after-retries-
exhausted, network errors.  Does NOT count: 400 (our bug), 401/403 (auth),
validation failures (model misbehavior, not service health).

Plan reference: §A1, §109.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

BREAKER_PREFIX = "ive:llm:cb"


class _RedisLike(Protocol):
    async def get(self, name: str) -> bytes | str | None: ...

    async def set(
        self,
        name: str,
        value: bytes | str | int,
        *,
        ex: int | None = ...,
    ) -> bool | None: ...

    async def incr(self, name: str) -> int: ...

    async def expire(self, name: str, time: int) -> bool: ...

    async def delete(self, *names: str) -> int: ...


class CircuitBreaker:
    """Async circuit breaker keyed by a logical scope (e.g. ``"groq"``)."""

    def __init__(
        self,
        client: _RedisLike,
        *,
        scope: str = "groq",
        threshold: int,
        cooldown_seconds: int,
    ) -> None:
        self._client = client
        self._scope = scope
        self._fail_key = f"{BREAKER_PREFIX}:{scope}:fail"
        self._open_key = f"{BREAKER_PREFIX}:{scope}:open"
        self._threshold = threshold
        self._cooldown = cooldown_seconds

    async def is_open(self) -> bool:
        """Return True when the breaker is currently open (calls suppressed)."""
        try:
            return bool(await self._client.get(self._open_key))
        except Exception:  # pragma: no cover - defensive
            return False

    async def record_failure(self) -> bool:
        """Increment the failure counter; return True if the breaker opened.

        Counter has a TTL equal to the cooldown so isolated failures decay.
        """
        try:
            count = await self._client.incr(self._fail_key)
            await self._client.expire(self._fail_key, self._cooldown)
        except Exception:  # pragma: no cover
            return False
        if count >= self._threshold:
            await self._client.set(
                self._open_key,
                "1",
                ex=self._cooldown,
            )
            logger.warning(
                "llm.circuit_breaker.opened",
                extra={"failures": count, "cooldown_s": self._cooldown},
            )
            _emit_breaker_state(self._scope, "open")
            return True
        _emit_breaker_state(self._scope, "closed")
        return False

    async def record_success(self) -> None:
        """Reset the failure counter on a clean call."""
        try:
            await self._client.delete(self._fail_key)
        except Exception:  # pragma: no cover
            pass
        _emit_breaker_state(self._scope, "closed")


def _emit_breaker_state(scope: str, state: str) -> None:
    """Best-effort gauge update; insulated from observability failures."""
    try:
        from ive.observability.metrics import set_circuit_breaker_state

        set_circuit_breaker_state(service=scope, state=state)
    except Exception:  # pragma: no cover - defensive
        pass
