"""Redis-backed cache for LLM responses.

Cache key shape (per §110):
    ive:llm:{function}:{prompt_version}:{template_sha[:16]}:{payload_schema_hash}:{facts_hash[:32]}

The template SHA component auto-invalidates cache entries when a prompt
template is structurally edited within a version.  The payload schema hash
catches additions to the fact payload (e.g. when B4's effect_size_ci is
added) without requiring a prompt-version bump.

Plan reference: §A1 (cache module), §6 / §110 (template SHA in key).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Protocol

from ive.llm.prompts import template_sha

logger = logging.getLogger(__name__)

CACHE_PREFIX = "ive:llm"


class _RedisLike(Protocol):
    """Minimal Redis interface used by the cache (compatible with redis-py & fakeredis)."""

    async def get(self, name: str) -> bytes | str | None: ...

    async def set(
        self,
        name: str,
        value: bytes | str,
        *,
        ex: int | None = ...,
    ) -> bool | None: ...

    async def delete(self, *names: str) -> int: ...

    async def sadd(self, name: str, *values: str) -> int: ...

    async def smembers(self, name: str) -> set[bytes] | set[str]: ...


def _canonical_json(payload: dict[str, Any]) -> str:
    """Stable JSON serialization for hashing — sort keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_schema_hash(payload: dict[str, Any]) -> str:
    """Hash of the *shape* of a payload (sorted top-level keys)."""
    keys = sorted(payload.keys())
    h = hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()
    return h[:8]


def facts_hash(payload: dict[str, Any]) -> str:
    """Hash of the canonical facts JSON, truncated to 32 hex chars."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:32]


def make_key(function: str, prompt_version: str, payload: dict[str, Any]) -> str:
    """Build the canonical cache key for ``(function, version, payload)``."""
    return (
        f"{CACHE_PREFIX}:{function}:{prompt_version}:"
        f"{template_sha(function, prompt_version)}:"
        f"{payload_schema_hash(payload)}:"
        f"{facts_hash(payload)}"
    )


def entity_index_key(entity_type: str, entity_id: str) -> str:
    """SET key tracking every cache entry that belongs to ``(entity_type, entity_id)``.

    Used by ``delete_for_entity`` to clean up cache rows when a dataset,
    experiment, or LV is deleted (per §10 / §34).
    """
    return f"{CACHE_PREFIX}:idx:{entity_type}:{entity_id}"


class RedisLLMCache:
    """Async Redis cache wrapper for LLM responses.

    Constructed with any object satisfying the ``_RedisLike`` protocol; in
    production this is ``redis.asyncio.Redis``, in tests it's ``fakeredis``.
    """

    def __init__(self, client: _RedisLike, ttl_seconds: int) -> None:
        self._client = client
        self._ttl = ttl_seconds

    async def get(self, key: str) -> str | None:
        """Return cached text or None on miss."""
        try:
            raw = await self._client.get(key)
        except Exception:  # pragma: no cover - defensive: redis transient errors
            logger.warning("llm.cache.get_failed", extra={"key": key})
            _emit_cache_metric("bypass")
            return None
        if raw is None:
            _emit_cache_metric("miss")
            return None
        _emit_cache_metric("hit")
        return raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

    async def set(
        self,
        key: str,
        value: str,
        *,
        entity_index: tuple[str, str] | None = None,
    ) -> None:
        """Persist ``value`` at ``key`` with the configured TTL.

        When ``entity_index`` is provided as ``(entity_type, entity_id)``,
        the key is also added to that entity's index set so it can be bulk-
        deleted via ``delete_for_entity``.
        """
        try:
            await self._client.set(key, value, ex=self._ttl)
            if entity_index is not None:
                etype, eid = entity_index
                await self._client.sadd(entity_index_key(etype, eid), key)
        except Exception:  # pragma: no cover - defensive
            logger.warning("llm.cache.set_failed", extra={"key": key})

    async def delete_for_entity(self, entity_type: str, entity_id: str) -> int:
        """Delete every cache entry indexed under ``(entity_type, entity_id)``.

        Returns the number of entries removed.  Idempotent.
        """
        index = entity_index_key(entity_type, entity_id)
        try:
            members = await self._client.smembers(index)
        except Exception:  # pragma: no cover - defensive
            logger.warning("llm.cache.smembers_failed", extra={"entity": index})
            return 0
        if not members:
            return 0
        keys = [m.decode("utf-8") if isinstance(m, (bytes, bytearray)) else m for m in members]
        try:
            removed = await self._client.delete(*keys, index)
        except Exception:  # pragma: no cover - defensive
            logger.warning("llm.cache.delete_failed", extra={"entity": index})
            return 0
        # ``delete`` returns total keys removed (incl. the index itself)
        return max(0, removed - 1)


def _emit_cache_metric(outcome: str) -> None:
    """Best-effort cache event metric. Never raises."""
    try:
        from ive.observability.metrics import record_cache

        record_cache(outcome=outcome)
    except Exception:  # pragma: no cover - defensive
        pass
