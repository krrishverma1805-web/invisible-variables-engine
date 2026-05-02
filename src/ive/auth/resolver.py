"""API-key resolution: hashed DB lookup with env-CSV fallback.

Two paths in order:

1. **DB-managed keys** — incoming raw key hashed and looked up in the
   ``api_keys`` table. Returns full scopes, rate limit, identity.
2. **Env-CSV legacy keys** — when no DB row matches AND the key appears
   verbatim in ``Settings.valid_api_keys``, accept it with a default
   ``[read, write]`` scope set and ``api_key_id=None`` (un-managed). This
   preserves backwards compatibility with the legacy ``X-API-Key=dev-key-1``
   workflow while ops migrate to DB-managed keys.

Plan reference: §155 (multi-user), §47 (resolved).
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ive.auth.scopes import AuthContext, AuthOutcome, Scope
from ive.auth.utils import hash_api_key
from ive.config import Settings
from ive.db.models import APIKey

logger = logging.getLogger(__name__)


async def resolve_api_key(
    raw_key: str | None,
    session: AsyncSession | None,
    settings: Settings,
) -> AuthOutcome:
    """Resolve ``raw_key`` to an :class:`AuthContext`, or return a rejection.

    ``session`` may be ``None`` for the legacy env-CSV path (e.g. when the
    request fires before the DB is ready). The function is non-raising: any
    DB error degrades silently to env-CSV fallback.
    """
    if not raw_key:
        return AuthOutcome(error="missing_key", event_type="auth_missing")

    # 1. DB lookup
    if session is not None:
        try:
            ctx = await _resolve_from_db(raw_key, session)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ive.auth.db_lookup_failed", extra={"error": str(exc)})
            ctx = None
        if ctx is not None:
            return AuthOutcome(context=ctx, event_type="auth_success")

    # 2. Env-CSV fallback (legacy unmanaged keys)
    if raw_key in settings.api_keys_set:
        return AuthOutcome(
            context=AuthContext(
                api_key_id=None,
                api_key_name=f"env:{_truncate(raw_key)}",
                scopes=frozenset({Scope.READ, Scope.WRITE}),
                rate_limit=settings.rate_limit_requests,
            ),
            event_type="auth_success",
            extras={"resolution": "env_csv"},
        )

    return AuthOutcome(error="invalid_key", event_type="auth_failure")


async def _resolve_from_db(raw_key: str, session: AsyncSession) -> AuthContext | None:
    """DB-side resolution; returns None on miss, expired, or inactive."""
    digest = hash_api_key(raw_key)
    stmt = select(APIKey).where(APIKey.key_hash == digest)
    result = await session.execute(stmt)
    row: APIKey | None = result.scalar_one_or_none()
    if row is None:
        return None
    if not row.is_active:
        return None
    if row.is_expired():
        # Treated like a missing key by callers, but the audit-log
        # event_type is set to ``auth_expired`` for clarity in security
        # review. We surface that via a sentinel return — None — and let
        # the resolver layer record the event from the resolver's
        # caller side. (Auth middleware re-checks expiry to set the
        # right event_type.)
        return None
    return AuthContext(
        api_key_id=str(row.id),
        api_key_name=row.name,
        scopes=frozenset(Scope(s) for s in row.scopes if s in {"read", "write", "admin"}),
        rate_limit=row.rate_limit,
    )


def _truncate(value: str, n: int = 8) -> str:
    """Short, non-reversible label for log lines (never the full key)."""
    digest = hash_api_key(value)
    return digest[:n]
